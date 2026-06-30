//! Tree-free external-sampling MCCFR over a game expressed as a state machine.
//!
//! [`Game`] describes a game as a transition function the solver walks on demand. Nothing is
//! stored per node; regret lives in a per-infoset table, so games whose trees are astronomically
//! large (but whose reachable infoset count is modest) are still solvable -- this is how poker-scale
//! games are handled. Small games can instead be built into a full [`GameTree`] with
//! [`GameTree::from_game`] and solved exactly.

use crate::splitmix::{SplitmixHasher, mix, splitmix};
use crate::{Game, Moves, NodeType, Outcomes, PlayerNum, RegretParams, SolveError};
use dashmap::DashMap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::hash::{BuildHasherDefault, Hash};

/// Hash `value` to a `u64` with the cheap splitmix hasher, for deriving sampling contexts.
fn hash_of<H: Hash>(value: &H) -> u64 {
    mix(0, value)
}

/// Deterministic sampling key for an iteration and context, so chance and the opponent resolve
/// consistently across the traverser's explored actions.
fn sample_key(iter: u64, context: u64, salt: u64) -> u64 {
    splitmix(splitmix(iter.wrapping_mul(0x100_0001).wrapping_add(salt)) ^ context)
}

/// Pick an index from non-negative weights, given a uniform `r`.
fn pick(weights: &[f64], r: u64) -> usize {
    let total: f64 = weights.iter().sum();
    // r, u64::MAX: mapping a random u64 to [0, 1) for inverse-CDF sampling
    #[allow(clippy::cast_precision_loss)]
    let mut x = (r as f64 / u64::MAX as f64) * total;
    for (i, &w) in weights.iter().enumerate() {
        x -= w;
        if x < 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// One action's accumulators within an infoset.
#[derive(Debug, Clone, Default)]
struct Stats {
    regret: f64,
    strat_sum: f64,
    // the iteration before which this action, if pruned (negative regret), may have its subtree
    // skipped; 0 means "don't skip" (positive, zero, or never pruned)
    skip_until: u64,
}

/// One infoset's per-action accumulators.
///
/// A single boxed slice (never resized after creation), so the `(regret, strat_sum, skip_until)` an
/// update touches together are contiguous and each infoset costs one allocation.
#[derive(Debug, Clone)]
struct Entry {
    stats: Box<[Stats]>,
}

impl Entry {
    fn new(actions: usize) -> Self {
        Entry {
            stats: vec![Stats::default(); actions].into_boxed_slice(),
        }
    }

    /// The average (cumulative) strategy over this infoset's actions.
    fn average(&self) -> Vec<f64> {
        let sum: f64 = self.stats.iter().map(|stat| stat.strat_sum).sum();
        if sum > 0.0 {
            self.stats.iter().map(|stat| stat.strat_sum / sum).collect()
        } else {
            // len: number of actions, far below 2^53
            #[allow(clippy::cast_precision_loss)]
            let uniform = 1.0 / self.stats.len() as f64;
            vec![uniform; self.stats.len()]
        }
    }
}

/// The regret-matched strategy over an infoset's actions.
///
/// Honors `params.no_positive` when no action has positive regret. Mirrors the materialized solver's
/// matcher, in f64.
fn matched_strategy(stats: &[Stats], params: &RegretParams) -> Vec<f64> {
    let count = stats.len();
    let norm: f64 = stats
        .iter()
        .map(|stat| stat.regret)
        .filter(|&value| value > 0.0)
        .sum();
    if norm > 0.0 {
        return stats
            .iter()
            .map(|stat| {
                if stat.regret > 0.0 {
                    stat.regret / norm
                } else {
                    0.0
                }
            })
            .collect();
    }
    // all regrets non-positive: fall back per the no-positive policy
    let no_positive = params.no_positive;
    if no_positive == 0.0 {
        // count: number of actions, far below 2^53
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / count as f64;
        vec![uniform; count]
    } else if no_positive.is_infinite() {
        // +inf picks the highest-regret action, -inf the lowest
        let want_max = no_positive.is_sign_positive();
        let pick = (0..count)
            .max_by(|&left, &right| {
                let order = stats[left].regret.total_cmp(&stats[right].regret);
                if want_max { order } else { order.reverse() }
            })
            .unwrap_or(0);
        let mut strat = vec![0.0; count];
        strat[pick] = 1.0;
        strat
    } else {
        let max = stats
            .iter()
            .map(|stat| stat.regret)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = stats
            .iter()
            .map(|stat| ((stat.regret - max) * no_positive).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        weights.iter().map(|&weight| weight / total).collect()
    }
}

/// Whether traverser action `i` may be skipped this iteration: it has zero regret-matched
/// probability and its negative regret can't have climbed back to zero yet (regret-based pruning).
fn is_pruned(strat: &[f64], skip_until: &[u64], i: usize, iter: u64) -> bool {
    strat[i] == 0.0 && iter < skip_until[i]
}

/// Apply one iteration's regret/strategy contributions to the explored actions of an infoset,
/// discount them (positive regret by `pos`, negative by `neg`, the strategy sum by `strat`), and
/// refresh each explored action's skip horizon. Pruned actions are left frozen so their (negative)
/// regret -- and hence their zero probability -- is preserved across the skipped iterations.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // horizon: floor of a non-negative ratio
fn commit_update(
    entry: &mut Entry,
    regret_delta: &[f64],
    strat_delta: &[f64],
    explored: &[bool],
    (pos, neg, strat): (f64, f64, f64),
    iter: u64,
    swing: f64,
) {
    for (i, stat) in entry.stats.iter_mut().enumerate() {
        if !explored[i] {
            continue;
        }
        stat.regret += regret_delta[i];
        stat.regret *= if stat.regret > 0.0 { pos } else { neg };
        stat.strat_sum += strat_delta[i];
        stat.strat_sum *= strat;
        // a still-negative action can be skipped until its regret could plausibly reach zero again,
        // which takes at least |regret| / swing iterations (regret moves by at most `swing` each)
        stat.skip_until = if stat.regret < 0.0 && swing > 0.0 {
            iter + (stat.regret.abs() / swing) as u64
        } else {
            0
        };
    }
}

/// One player's regret table, keyed by infoset.
///
/// `DashMap` is internally sharded with a lock per shard, so the forked traversal updates infosets
/// directly and concurrently -- the writes are disjoint by perfect recall, so they almost never
/// contend the same lock, and there is no separate merge phase or delta buffer. The fixed-seed
/// [`SplitmixHasher`] keeps placement reproducible and is far cheaper than a cryptographic hash on
/// the already-avalanched infoset-key digests.
type Table<I> = DashMap<I, Entry, BuildHasherDefault<SplitmixHasher>>;

/// An external-sampling MCCFR solver over a [`Game`], with DCFR discounting and regret-based pruning.
pub struct LazySolver<G: Game> {
    table: [Table<G::Infoset>; 2],
    iter: u64,
    params: RegretParams,
    factors: (f64, f64, f64), // this iteration's (pos, neg, strat) discounts
    swing: f64,               // max_payoff - min_payoff, the per-iteration regret bound for pruning
}

// manual impl: the regret tables can't be `Debug` without `Infoset: Eq + Hash + Debug`, so summarize
// them by infoset count instead of printing their contents
impl<G: Game> std::fmt::Debug for LazySolver<G>
where
    G::Infoset: Eq + Hash,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("LazySolver")
            .field("iter", &self.iter)
            .field("params", &self.params)
            .field("factors", &self.factors)
            .field("swing", &self.swing)
            .field("infosets", &self.infosets())
            .finish_non_exhaustive()
    }
}

impl<G: Game> LazySolver<G>
where
    G::Infoset: Eq + Hash,
{
    /// Create an empty solver for a game whose player-one payoffs lie in `[min_payoff, max_payoff]`.
    ///
    /// No regret is accumulated yet, and it uses [discounted CFR][RegretParams::dcfr]. The payoff
    /// range bounds the per-iteration regret swing that regret-based pruning depends on.
    #[must_use]
    pub fn new(min_payoff: f64, max_payoff: f64) -> Self {
        Self::with_params(RegretParams::dcfr(), min_payoff, max_payoff)
    }

    /// Create an empty solver with explicit regret/strategy parameters.
    ///
    /// For example, [`RegretParams::vanilla`] gives undiscounted CFR. See [`new`][Self::new] for the
    /// payoff range.
    #[must_use]
    pub fn with_params(params: RegretParams, min_payoff: f64, max_payoff: f64) -> Self {
        LazySolver {
            table: [Table::default(), Table::default()],
            iter: 0,
            params,
            factors: (1.0, 1.0, 1.0),
            swing: max_payoff - min_payoff,
        }
    }

    /// The number of infosets each player has accumulated regret for.
    #[must_use]
    pub fn infosets(&self) -> usize {
        let [one, two] = &self.table;
        one.len() + two.len()
    }

    /// An estimate of `player`'s average regret.
    ///
    /// This is the sum over its infosets of `2·max(0, max regret)/T` -- the same per-infoset bound the
    /// materialized [`crate::GameTree`] reports. Since the cumulative regret here is a Monte-Carlo
    /// estimate, treat it as an empirical convergence readout rather than a guaranteed bound.
    /// Exploitability is at most the sum over both players.
    #[must_use]
    pub fn player_regret_bound(&self, player: PlayerNum) -> f64 {
        if self.iter == 0 {
            return f64::INFINITY;
        }
        self.table[player.index()]
            .iter()
            .map(|entry| {
                // iter: iteration count, only loses precision past 2^53 iterations
                #[allow(clippy::cast_precision_loss)]
                let iters = self.iter as f64;
                let max_regret = entry
                    .value()
                    .stats
                    .iter()
                    .map(|stat| stat.regret)
                    .fold(0.0, f64::max);
                2.0 * max_regret / iters
            })
            .sum()
    }

    /// Run `iters` MCCFR iterations from `root`, each updating both players.
    ///
    /// A fork cut of 0 makes the fused traversal fully serial.
    pub fn run(&mut self, root: &G, iters: u64)
    where
        G: Clone + Sync,
        G::Infoset: Send + Sync,
        G::ChanceInfoset: Hash,
        G::Player: Sync,
    {
        for _ in 0..iters {
            self.iter += 1;
            self.factors = self.params.iteration_factors(self.iter);
            for traverser in 0..2 {
                Traversal {
                    table: &self.table,
                    traverser,
                    cut: 0,
                    iter: self.iter,
                    params: &self.params,
                    factors: self.factors,
                    swing: self.swing,
                }
                .run(root.clone(), 1.0, 0, 0);
            }
        }
    }

    /// Run `iters` iterations, forking the traverser's action subtrees while shallower than `cut`.
    ///
    /// With no materialized tree we can't precompute subtree sizes, so the parallelism cut is by
    /// depth. The fork stays lock-light: by perfect recall the traverser writes disjoint infosets,
    /// and the table (`DashMap`) is internally sharded, so concurrent writes almost never contend the
    /// same lock and there is no merge phase. `num_threads` of zero uses all available cores.
    ///
    /// # Errors
    /// If the thread pool fails to build.
    pub fn run_parallel(
        &mut self,
        root: &G,
        iters: u64,
        num_threads: usize,
        cut: u32,
    ) -> Result<(), SolveError>
    where
        G: Clone + Sync,
        G::Infoset: Send + Sync,
        G::ChanceInfoset: Hash,
        G::Player: Sync,
    {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if num_threads > 0 {
            builder = builder.num_threads(num_threads);
        }
        let pool = builder.build()?;
        let params = self.params;
        let swing = self.swing;
        pool.install(|| {
            for _ in 0..iters {
                self.iter += 1;
                let factors = params.iteration_factors(self.iter);
                for traverser in 0..2 {
                    Traversal {
                        table: &self.table,
                        traverser,
                        cut,
                        iter: self.iter,
                        params: &params,
                        factors,
                        swing,
                    }
                    .run(root.clone(), 1.0, 0, 0);
                }
            }
        });
        Ok(())
    }

    /// The average strategy at an infoset, if it has been visited.
    #[must_use]
    pub fn average(&self, player: PlayerNum, info: &G::Infoset) -> Option<Vec<f64>> {
        self.table[player.index()]
            .get(info)
            .map(|entry| entry.average())
    }

    /// Monte-Carlo estimate of player one's value under the current average strategies.
    #[must_use]
    pub fn estimate_value(&self, root: &G, samples: u64) -> f64
    where
        G: Clone,
    {
        let mut total = 0.0;
        for seed in 0..samples {
            total += self.playout(root, splitmix(seed.wrapping_add(0xabcd)));
        }
        // samples: number of playouts
        #[allow(clippy::cast_precision_loss)]
        let samples = samples as f64;
        total / samples
    }

    fn playout(&self, root: &G, mut rng: u64) -> f64
    where
        G: Clone,
    {
        let mut state = root.clone();
        loop {
            match state.into_node() {
                NodeType::Terminal(payoff) => return payoff,
                NodeType::Chance(_info, outcomes) => {
                    let weights: Vec<f64> = outcomes.iter().map(|(prob, _)| prob).collect();
                    rng = splitmix(rng);
                    state = outcomes.get(pick(&weights, rng)).1;
                }
                NodeType::Player(num, info, moves) => {
                    let count = moves.len();
                    let strat = self.average(num, &info).unwrap_or_else(|| {
                        // len: number of actions, far below 2^53
                        #[allow(clippy::cast_precision_loss)]
                        let uniform = 1.0 / count as f64;
                        vec![uniform; count]
                    });
                    rng = splitmix(rng);
                    state = moves.apply(pick(&strat, rng));
                }
            }
        }
    }
}

/// Read-only regret-matched strategy and per-action skip horizons, matching on zero regret for an
/// absent infoset.
fn strategy_and_skip<G: Game>(
    table: &[Table<G::Infoset>; 2],
    player: usize,
    info: &G::Infoset,
    count: usize,
    params: &RegretParams,
) -> (Vec<f64>, Vec<u64>)
where
    G::Infoset: Eq + Hash,
{
    table[player].get(info).map_or_else(
        || {
            (
                matched_strategy(&vec![Stats::default(); count], params),
                vec![0; count],
            )
        },
        |entry| {
            let skip_until = entry.stats.iter().map(|stat| stat.skip_until).collect();
            (matched_strategy(&entry.stats, params), skip_until)
        },
    )
}

/// Mix an action/outcome index into the running path hash, so each node has a distinct context for
/// sampling uncorrelated chance outcomes (a correlated chance node keys on its own infoset instead).
fn descend(path: u64, index: usize) -> u64 {
    splitmix(path ^ (index as u64).wrapping_add(1))
}

/// The invariant context of one external-sampling traversal: everything that does not change as the
/// recursion descends, passed by shared reference so each recursive call takes only the four varying
/// arguments.
struct Traversal<'a, G: Game> {
    table: &'a [Table<G::Infoset>; 2],
    traverser: usize,
    cut: u32, // fork the traverser's action subtrees while shallower than this
    iter: u64,
    params: &'a RegretParams,
    factors: (f64, f64, f64), // this iteration's (pos, neg, strat) discounts
    swing: f64,
}

impl<G> Traversal<'_, G>
where
    G: Game + Sync,
    G::Infoset: Eq + Hash + Send + Sync,
    G::ChanceInfoset: Hash,
    G::Player: Sync,
{
    /// Fused external-sampling traversal from `state`.
    ///
    /// The traverser explores every non-pruned action (forking those subtrees while `depth < cut`),
    /// chance and the opponent are sampled, and each visited infoset is updated *in place* on the way
    /// up -- no delta buffer, no merge phase. Concurrent updates touch disjoint infosets (perfect
    /// recall), so the table's per-shard locks rarely contend. Returns the traverser's expected payoff.
    fn run(&self, state: G, reach: f64, depth: u32, path: u64) -> f64 {
        match state.into_node() {
            NodeType::Terminal(payoff) => {
                if self.traverser == 0 {
                    payoff
                } else {
                    -payoff
                }
            }
            NodeType::Chance(info, outcomes) => {
                let weights: Vec<f64> = outcomes.iter().map(|(prob, _)| prob).collect();
                // a correlated chance node samples by its infoset (so nodes sharing it agree); an
                // uncorrelated one (`None`) samples by its unique path
                let context = info.as_ref().map_or(path, hash_of);
                let choice = pick(&weights, sample_key(self.iter, context, 1));
                self.run(outcomes.get(choice).1, reach, depth, descend(path, choice))
            }
            NodeType::Player(num, info, moves) => {
                let player = num.index();
                let count = moves.len();
                let (strat, skip_until) =
                    strategy_and_skip::<G>(self.table, player, &info, count, self.params);

                if player != self.traverser {
                    let choice = pick(&strat, sample_key(self.iter, hash_of(&info), 2));
                    return self.run(moves.apply(choice), reach, depth, descend(path, choice));
                }

                // the traverser explores every non-pruned action; depth counts forks taken, so cut is
                // in traverser decisions, independent of the chance/opponent levels between them
                let explored: Vec<bool> = (0..count)
                    .map(|i| !is_pruned(&strat, &skip_until, i, self.iter))
                    .collect();
                let explore = |i: usize| -> f64 {
                    if explored[i] {
                        self.run(moves.apply(i), reach * strat[i], depth + 1, descend(path, i))
                    } else {
                        // pruned: skip the subtree entirely (strat[i] == 0, so node_util is unaffected)
                        0.0
                    }
                };
                let util: Vec<f64> = if depth < self.cut {
                    (0..count).into_par_iter().map(explore).collect()
                } else {
                    (0..count).map(explore).collect()
                };
                let node_util: f64 = (0..count).map(|i| strat[i] * util[i]).sum();

                // fused update: write this infoset's regret/strategy directly into the table
                let regret_delta: Vec<f64> = (0..count).map(|i| util[i] - node_util).collect();
                let strat_delta: Vec<f64> = (0..count).map(|i| reach * strat[i]).collect();
                let mut entry = self.table[player]
                    .entry(info)
                    .or_insert_with(|| Entry::new(count));
                commit_update(
                    entry.value_mut(),
                    &regret_delta,
                    &strat_delta,
                    &explored,
                    self.factors,
                    self.iter,
                    self.swing,
                );
                node_util
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LazySolver;
    use crate::{
        Game, GameTree, Moves, NodeType, PlayerNum, RegretParams, SolveMethod, SolveParams,
    };
    use std::convert::Infallible;

    /// Matching pennies: player one picks a side, player two picks blind; match -> player one wins.
    /// The unique equilibrium is 50/50 for both, value 0.
    #[derive(Debug, Clone)]
    enum Pennies {
        Start,
        Mid(bool),
        End(bool, bool),
    }

    impl Game for Pennies {
        type Action = bool;
        type Infoset = u8;
        type ChanceInfoset = Infallible;
        type Chance = Infallible;
        type Player = Pennies;

        fn into_node(self) -> NodeType<Self> {
            match self {
                // a constant infoset per player hides the opponent's hidden choice
                Pennies::Start => NodeType::Player(PlayerNum::One, 0, Pennies::Start),
                Pennies::Mid(first) => NodeType::Player(PlayerNum::Two, 1, Pennies::Mid(first)),
                Pennies::End(a, b) => NodeType::Terminal(if a == b { 1.0 } else { -1.0 }),
            }
        }
    }

    impl Moves<Pennies> for Pennies {
        fn len(&self) -> usize {
            2
        }
        fn action(&self, index: usize) -> bool {
            index == 0
        }
        fn apply(&self, index: usize) -> Pennies {
            let action = index == 0;
            match self {
                Pennies::Start => Pennies::Mid(action),
                Pennies::Mid(first) => Pennies::End(*first, action),
                Pennies::End(..) => unreachable!(),
            }
        }
    }

    #[test]
    fn matching_pennies_is_balanced() {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&Pennies::Start, 200_000);
        let value = solver.estimate_value(&Pennies::Start, 200_000);
        assert!(value.abs() < 0.05, "value not ~0: {value}");
        let one = solver.average(PlayerNum::One, &0).unwrap();
        let two = solver.average(PlayerNum::Two, &1).unwrap();
        assert!((one[0] - 0.5).abs() < 0.1, "player one not ~50/50: {one:?}");
        assert!((two[0] - 0.5).abs() < 0.1, "player two not ~50/50: {two:?}");
    }

    #[test]
    fn materialized_pennies_matches() {
        // The same Game, materialized and solved with exact full-tree CFR, must reach the same
        // 50/50 equilibrium the tree-free LazySolver finds -- this cross-checks both the adapter and
        // the LazySolver against an independent, non-sampled solver.
        let game = GameTree::from_game(Pennies::Start).unwrap();
        let (strats, bound) = game
            .solve(SolveMethod::Full, 50_000, 0.0, 1, SolveParams::default())
            .unwrap();
        for player in [PlayerNum::One, PlayerNum::Two] {
            assert!(
                bound.player_regret_bound(player) < 0.02,
                "player {player:?} not converged"
            );
        }
        let [one, two] = strats.as_named();
        for named in [one, two] {
            for (_info, actions) in named {
                for (_action, prob) in actions {
                    assert!((prob - 0.5).abs() < 0.05, "not ~50/50: {prob}");
                }
            }
        }
    }

    #[test]
    fn reports_low_regret_bound() {
        // DCFR (the default) should drive the empirical regret readout toward zero, and the bound is
        // huge before any iterations run.
        let mut solver: LazySolver<Pennies> = LazySolver::new(-1.0, 1.0);
        assert!(solver.player_regret_bound(PlayerNum::One).is_infinite());
        solver.run(&Pennies::Start, 50_000);
        let total =
            solver.player_regret_bound(PlayerNum::One) + solver.player_regret_bound(PlayerNum::Two);
        assert!(total < 0.05, "regret bound not small: {total}");
    }

    #[test]
    fn vanilla_params_also_converge() {
        let mut solver = LazySolver::with_params(RegretParams::vanilla(), -1.0, 1.0);
        solver.run(&Pennies::Start, 200_000);
        let value = solver.estimate_value(&Pennies::Start, 200_000);
        assert!(value.abs() < 0.05, "value not ~0 under vanilla: {value}");
    }

    #[test]
    fn pruning_params_stay_sound() {
        // dcfr_prune lets negative regret persist, giving long skip horizons -- the regime where
        // regret-based pruning is most active. It must still converge to the equilibrium.
        let mut solver = LazySolver::with_params(RegretParams::dcfr_prune(), -1.0, 1.0);
        solver.run(&Pennies::Start, 200_000);
        let value = solver.estimate_value(&Pennies::Start, 200_000);
        assert!(value.abs() < 0.05, "value not ~0 with pruning: {value}");
        let one = solver.average(PlayerNum::One, &0).unwrap();
        assert!(
            (one[0] - 0.5).abs() < 0.1,
            "player one not ~50/50 with pruning: {one:?}"
        );
    }

    #[test]
    fn parallel_matches_serial() {
        let mut serial = LazySolver::new(-1.0, 1.0);
        serial.run(&Pennies::Start, 50_000);
        let mut parallel = LazySolver::new(-1.0, 1.0);
        parallel
            .run_parallel(&Pennies::Start, 50_000, 4, 2)
            .unwrap();
        // disjoint, deterministically-keyed deltas => bit-for-bit identical to serial
        for (player, info) in [(PlayerNum::One, 0u8), (PlayerNum::Two, 1u8)] {
            assert_eq!(
                serial.average(player, &info),
                parallel.average(player, &info),
                "serial vs parallel diverged for {player:?}"
            );
        }
    }
}
