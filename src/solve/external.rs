//! External regret solving
// f32 cumulant storage casts f64 math down to f32 throughout; that truncation is intentional, so it
// is allowed module-wide. Lossy int->float casts (cast_precision_loss) are handled per-site instead.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
use super::data;
use super::data::{
    Action, Discounts, RegretInfoset, RegretParams, SampleKey, SampledChance, SolveInfo,
};
use super::multinomial;
use crate::{ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, SolveError, SolveParams};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cell::UnsafeCell;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};

/// A cell granting unsynchronized, unchecked access to its contents
///
/// The lazy external solver only ever forms a `&mut` to an infoset nothing else references in the
/// current sweep: by perfect recall each infoset is reached at most once, and the parallel solver
/// forks over the active player's actions into disjoint infoset sets (the opponent's are only read).
/// That disjointness is enforced at construction -- a tree only builds with perfect recall down to
/// the action taken out of each infoset -- so every tree the solver runs on satisfies it. The cell
/// exposes this with neither a lock nor a [`RefCell`]'s runtime check; soundness rests on the caller
/// never forming overlapping references to the same cell.
struct InfoCell<T> {
    cell: UnsafeCell<T>,
    /// debug-only guard: set when an active visit reaches this infoset within a sweep. A second
    /// visit in the same sweep means perfect recall was violated past construction -- exactly when
    /// the unchecked `&mut` could alias -- so it trips an assertion instead of risking UB.
    #[cfg(debug_assertions)]
    visited: AtomicBool,
}

// SAFETY: the solver only forms a `&mut` to a cell from the one thread that owns that infoset's
// action subtree in the current sweep, and never while another thread holds any reference to it.
unsafe impl<T: Send> Sync for InfoCell<T> {}

impl<T> InfoCell<T> {
    fn new(value: T) -> Self {
        InfoCell {
            cell: UnsafeCell::new(value),
            #[cfg(debug_assertions)]
            visited: AtomicBool::new(false),
        }
    }

    fn into_inner(self) -> T {
        self.cell.into_inner()
    }

    /// Borrow the contents mutably, unchecked.
    ///
    /// SAFETY: the caller must hold no other reference to this cell while the returned `&mut` lives;
    /// the solver guarantees this (see the type-level invariant).
    #[allow(clippy::mut_from_ref)]
    fn borrow_mut(&self) -> &mut T {
        unsafe { &mut *self.cell.get() }
    }

    /// Borrow the contents shared, unchecked.
    ///
    /// SAFETY: used only for opponent infosets, which are read but never written during a sweep, so
    /// concurrent shared reads never alias a `&mut`.
    fn borrow(&self) -> &T {
        unsafe { &*self.cell.get() }
    }

    /// Record (debug builds only) that an active visit reached this infoset this sweep, asserting it
    /// has not already. The flag is read and set through `&self` before any `borrow_mut`, so even the
    /// racing case it detects -- two threads reaching one infoset -- stays well defined.
    #[cfg(debug_assertions)]
    fn mark_visited(&self) {
        let already = self.visited.swap(true, Ordering::Relaxed);
        assert!(
            !already,
            "active infoset reached twice in one sweep -- perfect recall violated"
        );
    }

    /// Clear the per-sweep visit flag (debug builds only) before a sweep begins.
    #[cfg(debug_assertions)]
    fn clear_visited(&self) {
        self.visited.store(false, Ordering::Relaxed);
    }
}

/// A regret infoset for external sampling
#[derive(Debug)]
struct CachedInfoset {
    reg: RegretInfoset,
    /// The global `cum_log_pos` at this infoset's last cumulative-regret catch-up (lazy mode)
    base_log_pos: f64,
    /// The global `cum_log_neg` at this infoset's last cumulative-regret catch-up (lazy mode)
    base_log_neg: f64,
}

impl CachedInfoset {
    /// Create a new cached infoset
    fn new(num_actions: usize) -> Self {
        CachedInfoset {
            reg: RegretInfoset::new(num_actions),
            base_log_pos: 0.0,
            base_log_neg: 0.0,
        }
    }

    /// Deterministically sample an action from the current strategy
    ///
    /// Keying the draw on `(seed, iteration, sweep, infoset)` makes the opponent play consistently
    /// across every action the updating player explores -- without a shared cache -- because the
    /// same infoset draws the same way within a sweep.
    fn sample(&self, key: SampleKey, infoset: u32) -> usize {
        multinomial::sample(&mut key.rng(infoset), self.reg.strat())
    }

    /// Apply the deferred cumulative-regret discount for the iterations since the last catch-up
    ///
    /// `target_log_pos`/`target_log_neg` are the global cumulative log-discounts to bring this
    /// infoset up to. Cumulative regret keeps its sign between catch-ups (discounting by a positive
    /// factor is sign-preserving), so each element is scaled by the positive- or negative-regret
    /// catch-up factor according to its current sign.
    fn catch_up(&mut self, target_log_pos: f64, target_log_neg: f64) {
        let pos_factor = (target_log_pos - self.base_log_pos).exp();
        let neg_factor = (target_log_neg - self.base_log_neg).exp();
        for action in &mut *self.reg.actions {
            if action.regret() > 0.0 {
                action.set_regret(action.regret() * pos_factor);
            } else if action.regret() < 0.0 {
                action.set_regret(action.regret() * neg_factor);
            }
        }
        self.base_log_pos = target_log_pos;
        self.base_log_neg = target_log_neg;
    }

    /// Fold this iteration's per-action utilities into cumulative regret and the average strategy
    ///
    /// Shared tail of [`visit_lazy`] and [`finish_visit`]: `util_for(action, prob)` yields each
    /// action's expected utility -- computed inline by recursing (serial solver) or read back from
    /// the pre-forked subtree results (parallel solver). The infoset must already be caught up.
    fn accumulate_visit(
        &mut self,
        lazy: &LazyState,
        discounts: &Discounts,
        reach: f64,
        util_for: impl Fn(usize, f64) -> f64,
    ) -> f64 {
        // the average accumulates this player's own strategy weighted by reach times the deferred
        // lazy weight `t^γ` (the normalizer cancels at the end)
        let avg_weight = reach * lazy.strat_weight;
        let mut expected = 0.0;
        for (idx, action) in self.reg.actions.iter_mut().enumerate() {
            let prob = action.strat();
            let util = util_for(idx, prob);
            expected += prob * util;
            action.add_regret(util);
            action.cum_strat += prob * avg_weight;
        }
        for action in &mut *self.reg.actions {
            action.add_regret(-expected);
        }
        // regret match for the next iteration and apply this iteration's discount
        discounts.advance_infoset(&mut self.reg.actions);
        // this infoset is now caught up through the current iteration's discount
        self.base_log_pos = lazy.cum_log_pos + lazy.log_pos_step;
        self.base_log_neg = lazy.cum_log_neg + lazy.log_neg_step;
        expected
    }

    /// Accumulate an active-player visit, recursing into each action subtree in turn
    fn visit_lazy(
        &mut self,
        player: &Player,
        lazy: &LazyState,
        discounts: &Discounts,
        reach: f64,
        rec: impl Fn(&Node, f64) -> f64,
    ) -> f64 {
        // bring cumulative regret up to the start of this iteration
        self.catch_up(lazy.cum_log_pos, lazy.cum_log_neg);
        self.accumulate_visit(lazy, discounts, reach, |idx, prob| {
            rec(&player.actions[idx], reach * prob)
        })
    }

    /// Finish an active-player visit from already-computed child utilities
    ///
    /// `utils` holds one subtree utility per action, in order. The infoset must already be caught up;
    /// unlike [`visit_lazy`] this does not catch up.
    fn finish_visit(
        &mut self,
        lazy: &LazyState,
        discounts: &Discounts,
        reach: f64,
        utils: &[f64],
    ) -> f64 {
        self.accumulate_visit(lazy, discounts, reach, |idx, _prob| utils[idx])
    }

    /// Catch the infoset up through iteration `it` and return its regret-bound contribution
    fn catch_up_bound(&mut self, target_log_pos: f64, target_log_neg: f64, it: u64) -> f64 {
        self.catch_up(target_log_pos, target_log_neg);
        // it: iteration count, only loses precision past 2^53 iterations
        #[allow(clippy::cast_precision_loss)]
        let iters = it as f64;
        2.0 * f64::max(
            self.reg
                .actions
                .iter()
                .map(Action::regret)
                .fold(f64::NEG_INFINITY, f64::max),
            0.0,
        ) / iters
    }
}

/// Global deferred-discount state shared across a lazy external-sampling iteration
#[derive(Debug, Default)]
struct LazyState {
    /// sum of ln(positive-regret discount) over the iterations strictly before the current one
    cum_log_pos: f64,
    /// sum of ln(negative-regret discount) over the iterations strictly before the current one
    cum_log_neg: f64,
    /// ln of the current iteration's positive-regret discount
    log_pos_step: f64,
    /// ln of the current iteration's negative-regret discount
    log_neg_step: f64,
    /// Weight `t^γ` applied to the current iteration's average-strategy contribution
    strat_weight: f64,
}

/// The active/external/chance recursion for a lazy single-threaded external iteration
///
/// Mirrors [`recurse_regret`] but folds the active-player infoset update into the visit (so
/// unvisited infosets are never touched). The active player accumulates its own average strategy at
/// its own infosets; the opponent is only sampled, so the active player's `reach` carries through
/// opponent and chance nodes unchanged.
/// The invariant context of one external-sampling sweep.
///
/// Passed by shared reference so each recursive call takes only the node, reach, and sampling key
/// that actually change as the walk descends.
struct Sweep<'a> {
    chance_infosets: &'a [SampledChance],
    active_player_infosets: &'a [InfoCell<CachedInfoset>],
    external_player_infosets: &'a [InfoCell<CachedInfoset>],
    lazy: &'a LazyState,
    discounts: &'a Discounts<'a>,
    fork_depth: u32,
}

impl Sweep<'_> {
    /// Run one sweep for player `FIRST`: fork over the rayon pool when `parallel`, else walk serially.
    fn run<const FIRST: bool>(&self, parallel: bool, start: &Node, key: SampleKey) {
        #[cfg(debug_assertions)]
        for info_cell in self.active_player_infosets {
            info_cell.clear_visited();
        }
        if parallel {
            self.parallel::<FIRST>(start, 0, 1.0, key);
        } else {
            self.serial::<FIRST>(start, 1.0, key);
        }
    }

    /// Serial external-sampling walk from `node`, updating infosets in place on the way up.
    fn serial<const FIRST: bool>(&self, node: &Node, reach: f64, key: SampleKey) -> f64 {
        match node {
            Node::Terminal(payoff) => {
                if FIRST {
                    *payoff
                } else {
                    -payoff
                }
            }
            Node::Chance(chance) => {
                let next =
                    self.chance_infosets[chance.infoset].sample(&mut key.rng(chance.infoset as u32));
                self.serial::<FIRST>(&chance.outcomes[next], reach, key)
            }
            Node::Player(player) => match (player.num, FIRST) {
                (PlayerNum::One, true) | (PlayerNum::Two, false) => {
                    let info_cell = &self.active_player_infosets[player.infoset];
                    #[cfg(debug_assertions)]
                    info_cell.mark_visited();
                    info_cell.borrow_mut().visit_lazy(
                        player,
                        self.lazy,
                        self.discounts,
                        reach,
                        |next, child_reach| self.serial::<FIRST>(next, child_reach, key),
                    )
                }
                (PlayerNum::Two, true) | (PlayerNum::One, false) => {
                    let info = self.external_player_infosets[player.infoset].borrow();
                    let next = &player.actions[info.sample(key, player.infoset as u32)];
                    self.serial::<FIRST>(next, reach, key)
                }
            },
        }
    }

    /// Parallel external-sampling walk: fork the active player's action subtrees while `depth <
    /// fork_depth`, then fall back to [`serial`](Self::serial).
    ///
    /// The action subtrees of an active node touch disjoint infosets (see [`InfoCell`]), so they run
    /// in parallel without synchronization; `depth` counts the active forks already taken, bounding
    /// spawned tasks to roughly `branching`^`fork_depth`.
    fn parallel<const FIRST: bool>(
        &self,
        node: &Node,
        depth: u32,
        reach: f64,
        key: SampleKey,
    ) -> f64 {
        match node {
            Node::Terminal(payoff) => {
                if FIRST {
                    *payoff
                } else {
                    -payoff
                }
            }
            Node::Chance(chance) => {
                let next =
                    self.chance_infosets[chance.infoset].sample(&mut key.rng(chance.infoset as u32));
                self.parallel::<FIRST>(&chance.outcomes[next], depth, reach, key)
            }
            Node::Player(player) => match (player.num, FIRST) {
                (PlayerNum::One, true) | (PlayerNum::Two, false) => {
                    if depth >= self.fork_depth {
                        return self.serial::<FIRST>(node, reach, key);
                    }
                    let info_cell = &self.active_player_infosets[player.infoset];
                    #[cfg(debug_assertions)]
                    info_cell.mark_visited();
                    // catch up and snapshot the per-action reach weights before the fork, so no borrow
                    // of this infoset is held across the parallel section
                    let reaches: Vec<f64> = {
                        let info = info_cell.borrow_mut();
                        info.catch_up(self.lazy.cum_log_pos, self.lazy.cum_log_neg);
                        info.reg
                            .strat()
                            .map(|prob| reach * f64::from(prob))
                            .collect()
                    };
                    let utils: Vec<f64> = (0..player.actions.len())
                        .into_par_iter()
                        .map(|idx| {
                            self.parallel::<FIRST>(&player.actions[idx], depth + 1, reaches[idx], key)
                        })
                        .collect();
                    info_cell
                        .borrow_mut()
                        .finish_visit(self.lazy, self.discounts, reach, &utils)
                }
                (PlayerNum::Two, true) | (PlayerNum::One, false) => {
                    let info = self.external_player_infosets[player.infoset].borrow();
                    let next = &player.actions[info.sample(key, player.infoset as u32)];
                    self.parallel::<FIRST>(next, depth, reach, key)
                }
            },
        }
    }
}

/// External-sampling MCCFR over a materialized game tree
///
/// Deferring the per-iteration discount work means each iteration only touches the infosets the
/// sample actually reaches; the rest are caught up lazily when next visited, or in full on the
/// (periodic) check iterations when the regret bound is evaluated. With `num_threads > 1` each sweep
/// forks over the active player's action subtrees on a rayon pool; the result is identical to the
/// serial run for a given seed regardless of thread count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_external_single(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    sp: &SolveParams,
    num_threads: usize,
) -> Result<SolveInfo, SolveError> {
    let params = &sp.regret;
    let check_interval = sp.check_interval;
    let seed = sp.seed;
    let fork_depth = sp.fork_depth;
    let chance_infosets: Box<[_]> = chance_info
        .iter()
        .map(|info| SampledChance::new(info.probs()))
        .collect();
    let [player_one, player_two] = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| InfoCell::new(CachedInfoset::new(info.num_actions())))
            .collect::<Box<[_]>>()
    });
    let pool = (num_threads > 1)
        .then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
        })
        .transpose()?;
    let parallel = pool.is_some();
    let run = move || {
        let mut lazy = LazyState::default();
        let [mut reg_one, mut reg_two] = [f64::INFINITY; 2];
        for it in 1..=max_iter {
            let check = data::should_check(it, max_iter, check_interval);
            lazy.log_pos_step = RegretParams::gen_discount(it, params.pos_regret).ln();
            lazy.log_neg_step = RegretParams::gen_discount(it, params.neg_regret).ln();
            // discounting the average strategy by `(t/(t+1))^γ` each iteration is the same as
            // weighting iteration `t`'s contribution by `t^γ` and normalizing at the end
            // it: iteration count, only loses precision past 2^53 iterations
            #[allow(clippy::cast_precision_loss)]
            let iters = it as f64;
            lazy.strat_weight = iters.powf(params.strat);
            let discounts = Discounts::new(params, it, it);
            // the two sweeps sample independently (distinct sweep id in the key), but both read the
            // same pre-iteration cumulative discount, which advances once after both have run
            Sweep {
                chance_infosets: &chance_infosets,
                active_player_infosets: &player_one,
                external_player_infosets: &player_two,
                lazy: &lazy,
                discounts: &discounts,
                fork_depth,
            }
            .run::<true>(parallel, start, SampleKey::new(seed, it, 0));
            Sweep {
                chance_infosets: &chance_infosets,
                active_player_infosets: &player_two,
                external_player_infosets: &player_one,
                lazy: &lazy,
                discounts: &discounts,
                fork_depth,
            }
            .run::<false>(parallel, start, SampleKey::new(seed, it, 1));
            lazy.cum_log_pos += lazy.log_pos_step;
            lazy.cum_log_neg += lazy.log_neg_step;
            if check {
                // catch every infoset up through this iteration to evaluate the regret bound; no
                // sweep is in flight, so each borrow_mut is the only live reference to its cell
                reg_one = player_one
                    .iter()
                    .map(|info| {
                        info.borrow_mut()
                            .catch_up_bound(lazy.cum_log_pos, lazy.cum_log_neg, it)
                    })
                    .sum();
                reg_two = player_two
                    .iter()
                    .map(|info| {
                        info.borrow_mut()
                            .catch_up_bound(lazy.cum_log_pos, lazy.cum_log_neg, it)
                    })
                    .sum();
                if f64::max(reg_one, reg_two) < max_reg {
                    break;
                }
            }
        }
        let strats = [player_one, player_two].map(|player| {
            Vec::from(player)
                .into_iter()
                .flat_map(|info| Vec::from(info.into_inner().reg.into_avg_strat()))
                .collect()
        });
        ([reg_one, reg_two], strats)
    };
    match &pool {
        Some(pool) => Ok(pool.install(run)),
        None => Ok(run()),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, RegretParams, SolveParams,
    };

    mod eager {
        use super::super::CachedInfoset;
        use super::super::data::{
            Discounts, SampleKey, SampledChance, SolveInfo, should_check,
        };
        use crate::{ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, SolveParams};
        use std::cell::RefCell;

        impl CachedInfoset {
            /// Accumulate this iteration's counterfactual regret and average strategy (eager)
            ///
            /// `reach` is the active player's own reach probability to this infoset (the product of
            /// their action probabilities along the path); the average strategy accumulates this
            /// player's strategy weighted by `reach`.
            fn recurse_active(
                &mut self,
                player: &Player,
                reach: f64,
                rec: impl Fn(&Node, f64) -> f64,
            ) -> f64 {
                let mut expected = 0.0;
                for (next, action) in player.actions.iter().zip(self.reg.actions.iter_mut()) {
                    let prob = action.strat();
                    let util = rec(next, reach * prob);
                    expected += prob * util;
                    action.add_regret(util);
                    action.cum_strat += prob * reach;
                }
                // account for only adding utility to cum_regret
                for action in &mut *self.reg.actions {
                    action.add_regret(-expected);
                }
                expected
            }

            /// Regret match the next strategy and discount the cumulative regret (eager)
            fn advance(&mut self, discounts: &Discounts) -> f64 {
                let bound = discounts.advance_infoset(&mut self.reg.actions);
                discounts.discount_average_strat(&mut self.reg.actions);
                bound
            }
        }

        fn recurse_regret<const FIRST: bool>(
            node: &Node,
            chance_infosets: &[SampledChance],
            active_player_infosets: &[RefCell<CachedInfoset>],
            external_player_infosets: &[RefCell<CachedInfoset>],
            reach: f64,
            key: SampleKey,
        ) -> f64 {
            match node {
                Node::Terminal(payoff) => {
                    if FIRST {
                        *payoff
                    } else {
                        -payoff
                    }
                }
                Node::Chance(chance) => {
                    let ind =
                        chance_infosets[chance.infoset].sample(&mut key.rng(chance.infoset as u32));
                    recurse_regret::<FIRST>(
                        &chance.outcomes[ind],
                        chance_infosets,
                        active_player_infosets,
                        external_player_infosets,
                        reach,
                        key,
                    )
                }
                Node::Player(player) => match (player.num, FIRST) {
                    (PlayerNum::One, true) | (PlayerNum::Two, false) => active_player_infosets
                        [player.infoset]
                        .borrow_mut()
                        .recurse_active(player, reach, |next, child_reach| {
                            recurse_regret::<FIRST>(
                                next,
                                chance_infosets,
                                active_player_infosets,
                                external_player_infosets,
                                child_reach,
                                key,
                            )
                        }),
                    // the opponent is only sampled here -- their average strategy is accumulated during
                    // their own sweep, so the active player's reach carries through unchanged
                    (PlayerNum::One, false) | (PlayerNum::Two, true) => {
                        let next = {
                            let info = external_player_infosets[player.infoset].borrow();
                            &player.actions[info.sample(key, player.infoset as u32)]
                        };
                        recurse_regret::<FIRST>(
                            next,
                            chance_infosets,
                            active_player_infosets,
                            external_player_infosets,
                            reach,
                            key,
                        )
                    }
                },
            }
        }

        /// Advance every infoset of a player, returning the summed regret bound on a check iteration
        ///
        /// On a non-check iteration the advance still runs for its side effects, but the regret bound isn't
        /// needed, so we skip summing it and return infinity (which never satisfies an early-termination
        /// test).
        fn advance_player(
            player: &mut [RefCell<CachedInfoset>],
            discounts: &Discounts,
            check: bool,
        ) -> f64 {
            if check {
                player
                    .iter_mut()
                    .map(|info| info.get_mut().advance(discounts))
                    .sum()
            } else {
                for info in player.iter_mut() {
                    info.get_mut().advance(discounts);
                }
                f64::INFINITY
            }
        }

        /// The non-deferred (eager) variant: discount every infoset every iteration. Kept only to
        /// cross-check the lazy [`solve_external_single`] in tests -- the deferred catch-up must produce the
        /// same regret and strategy as discounting eagerly.
        pub(super) fn solve_external_eager(
            start: &Node,
            chance_info: &[impl ChanceInfoset],
            player_info: [&[impl PlayerInfoset]; 2],
            max_iter: u64,
            max_reg: f64,
            sp: &SolveParams,
        ) -> SolveInfo {
            let params = &sp.regret;
            let check_interval = sp.check_interval;
            let seed = sp.seed;
            let chance_infosets: Box<[_]> = chance_info
                .iter()
                .map(|info| SampledChance::new(info.probs()))
                .collect();
            let [mut player_one, mut player_two] = player_info.map(|infos| {
                infos
                    .iter()
                    .map(|info| RefCell::new(CachedInfoset::new(info.num_actions())))
                    .collect::<Box<[_]>>()
            });
            let [mut reg_one, mut reg_two] = [f64::INFINITY; 2];
            for it in 1..=max_iter {
                let check = should_check(it, max_iter, check_interval);
                // each player accumulates its own regret and average strategy during its own sweep, then
                // advances; the two sweeps sample independently via distinct sweep ids in the key
                recurse_regret::<true>(
                    start,
                    &chance_infosets,
                    &player_one,
                    &player_two,
                    1.0,
                    SampleKey::new(seed, it, 0),
                );
                let discounts = Discounts::new(params, it, it);
                reg_one = advance_player(&mut player_one, &discounts, check);
                // player two
                recurse_regret::<false>(
                    start,
                    &chance_infosets,
                    &player_two,
                    &player_one,
                    1.0,
                    SampleKey::new(seed, it, 1),
                );
                reg_two = advance_player(&mut player_two, &discounts, check);
                // check to terminate
                if check && f64::max(reg_one, reg_two) < max_reg {
                    break;
                }
            }
            let strats = [player_one, player_two].map(|player| {
                Vec::from(player)
                    .into_iter()
                    .flat_map(|info| Vec::from(info.into_inner().reg.into_avg_strat()))
                    .collect()
            });
            ([reg_one, reg_two], strats)
        }
    }

    #[derive(Debug, Clone)]
    struct Pinfo(usize);

    impl PlayerInfoset for Pinfo {
        fn num_actions(&self) -> usize {
            self.0
        }

        fn prev_infoset(&self) -> Option<usize> {
            None
        }
    }

    #[derive(Debug, Clone)]
    struct Cinfo(Box<[f64]>);

    impl ChanceInfoset for Cinfo {
        fn probs(&self) -> &[f64] {
            &self.0
        }
    }

    type GameTree = (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]);

    fn simple_game() -> GameTree {
        let root = Node::Chance(Chance {
            outcomes: vec![
                Node::Player(Player {
                    num: PlayerNum::One,
                    actions: vec![Node::Terminal(1.0), Node::Terminal(-1.0)].into(),
                    infoset: 0,
                }),
                Node::Player(Player {
                    num: PlayerNum::Two,
                    actions: vec![Node::Terminal(2.0), Node::Terminal(-2.0)].into(),
                    infoset: 0,
                }),
            ]
            .into(),
            infoset: 0,
        });
        let chance = vec![Cinfo(vec![0.5, 0.5].into())].into();
        let players = [vec![Pinfo(2)].into(), vec![Pinfo(2)].into()];
        (root, chance, players)
    }

    fn even_or_odd() -> GameTree {
        let root = Node::Player(Player {
            num: PlayerNum::One,
            actions: vec![
                Node::Player(Player {
                    num: PlayerNum::Two,
                    actions: vec![Node::Terminal(1.0), Node::Terminal(-1.0)].into(),
                    infoset: 0,
                }),
                Node::Player(Player {
                    num: PlayerNum::Two,
                    actions: vec![Node::Terminal(-1.0), Node::Terminal(1.0)].into(),
                    infoset: 0,
                }),
            ]
            .into(),
            infoset: 0,
        });
        let chance = [].into();
        let players = [vec![Pinfo(2)].into(), vec![Pinfo(2)].into()];
        (root, chance, players)
    }

    #[test]
    fn test_external_simple() {
        let params = SolveParams {
            regret: RegretParams::vanilla(),
            check_interval: 256,
            seed: 0,
            fork_depth: 3,
        };
        // the deferred (lazy) production solver and the eager reference must agree
        for eager in [false, true] {
            let (root, chance, [one, two]) = simple_game();
            let ([reg_one, reg_two], [strat_one, strat_two]) = if eager {
                eager::solve_external_eager(&root, &chance, [&*one, &*two], 1000, 0.0, &params)
            } else {
                super::solve_external_single(&root, &chance, [&*one, &*two], 1000, 0.0, &params, 1)
                    .unwrap()
            };
            assert!(strat_one[1] < 0.05, "eager={eager}");
            assert!(strat_two[0] < 0.05, "eager={eager}");
            assert!(reg_one < 0.05, "eager={eager}");
            assert!(reg_two < 0.05, "eager={eager}");
        }
    }

    #[test]
    fn test_external_even_or_odd() {
        let params = SolveParams {
            regret: RegretParams::vanilla(),
            check_interval: 256,
            seed: 0,
            fork_depth: 3,
        };
        for eager in [false, true] {
            let (root, chance, [one, two]) = even_or_odd();
            let ([reg_one, reg_two], [strat_one, strat_two]) = if eager {
                eager::solve_external_eager(&root, &chance, [&*one, &*two], 10_000, 0.005, &params)
            } else {
                super::solve_external_single(
                    &root,
                    &chance,
                    [&*one, &*two],
                    10_000,
                    0.005,
                    &params,
                    1,
                )
                .unwrap()
            };
            assert!(
                (strat_one[1] - 0.5).abs() < 0.05,
                "{strat_one:?} eager={eager}"
            );
            assert!(
                (strat_two[0] - 0.5).abs() < 0.05,
                "{strat_two:?} eager={eager}"
            );
            assert!(reg_one < 0.05, "eager={eager}");
            assert!(reg_two < 0.05, "eager={eager}");
        }
    }

    // a forked solve must reproduce the serial run bit-for-bit at any thread count and any fork
    // depth (the depth is a pure performance knob)
    #[test]
    // bit-identical determinism is exactly what's under test, so exact float equality is intended
    #[allow(clippy::float_cmp)]
    fn parallel_matches_serial() {
        let base = SolveParams {
            regret: RegretParams::dcfr(),
            check_interval: 256,
            seed: 7,
            fork_depth: 3,
        };
        let (root, chance, [one, two]) = even_or_odd();
        let serial =
            super::solve_external_single(&root, &chance, [&*one, &*two], 5000, 0.0, &base, 1)
                .unwrap();
        for threads in [2, 4] {
            for fork_depth in [1, 3, 5] {
                let params = SolveParams { fork_depth, ..base };
                let forked = super::solve_external_single(
                    &root,
                    &chance,
                    [&*one, &*two],
                    5000,
                    0.0,
                    &params,
                    threads,
                )
                .unwrap();
                assert_eq!(
                    serial.0, forked.0,
                    "regrets differ at {threads} threads, depth {fork_depth}"
                );
                assert_eq!(
                    serial.1, forked.1,
                    "strategies differ at {threads} threads, depth {fork_depth}"
                );
            }
        }
    }

    // The deferred (lazy) cumulative-regret catch-up must equal applying the per-iteration discount
    // every iteration. Cumulative regret keeps its sign while only being discounted, so we check the
    // log-space catch-up factor against the repeated product for both signs and every preset.
    #[test]
    fn lazy_catch_up_matches_repeated_discount() {
        let presets = [
            RegretParams::vanilla(),
            RegretParams::dcfr(),
            RegretParams::lcfr(),
            RegretParams::dcfr_prune(),
        ];
        for params in &presets {
            // run the global cumulative log-discount forward, mirroring solve_external_single
            let mut cum_log_pos = 0.0;
            let mut cum_log_neg = 0.0;
            // an infoset last caught up after iteration `t0`
            let t0 = 3_u64;
            for t in 1..=t0 {
                cum_log_pos += RegretParams::gen_discount(t, params.pos_regret).ln();
                cum_log_neg += RegretParams::gen_discount(t, params.neg_regret).ln();
            }
            let base_log_pos = cum_log_pos;
            let base_log_neg = cum_log_neg;
            // discount iterations t0+1..=t1 the eager way (repeated) and the lazy way (one catch-up)
            let t1 = 11_u64;
            let mut eager_pos = 5.0_f64;
            let mut eager_neg = -2.0_f64;
            for t in (t0 + 1)..=t1 {
                let pos = RegretParams::gen_discount(t, params.pos_regret);
                let neg = RegretParams::gen_discount(t, params.neg_regret);
                cum_log_pos += pos.ln();
                cum_log_neg += neg.ln();
                eager_pos *= pos;
                eager_neg *= neg;
            }
            let lazy_pos = 5.0 * (cum_log_pos - base_log_pos).exp();
            let lazy_neg = -2.0 * (cum_log_neg - base_log_neg).exp();
            assert!(
                (eager_pos - lazy_pos).abs() < 1e-9,
                "pos {params:?}: {eager_pos} vs {lazy_pos}"
            );
            assert!(
                (eager_neg - lazy_neg).abs() < 1e-9,
                "neg {params:?}: {eager_neg} vs {lazy_neg}"
            );
        }
    }
}
