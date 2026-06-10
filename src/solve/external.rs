//! External regret solving
// f32 cumulant storage casts f64 math down to f32 throughout; that truncation is intentional, so it
// is allowed module-wide. Lossy int->float casts (cast_precision_loss) are handled per-site instead.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
use super::data;
use super::data::{
    Discounts, RegretInfoset, RegretParams, SampleKey, SampledChance, SolveInfo,
};
use super::multinomial::Multinomial;
use crate::{ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, SolveParams};
use rand::distr::Distribution;
use std::cell::RefCell;

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
        Multinomial::new(&self.reg.strat).sample(&mut key.rng(infoset))
    }

    /// Add the current strategy into the average, weighted for discounting (lazy mode)
    ///
    /// Discounting the average strategy by `(t/(t+1))^γ` every iteration is equivalent to weighting
    /// iteration `t`'s contribution by `t^γ` and normalizing at the end, which avoids a per-iteration
    /// discount pass over every infoset. The shared `t^γ` and the global normalizer both cancel when
    /// the average is normalized.
    fn accumulate_strat_weighted(&mut self, weight: f64) {
        for (val, cum) in self.reg.strat.iter().zip(self.reg.cum_strat.iter_mut()) {
            *cum += f64::from(*val) * weight;
        }
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
        for reg in &mut *self.reg.cum_regret {
            if *reg > 0.0 {
                *reg = (f64::from(*reg) * pos_factor) as f32;
            } else if *reg < 0.0 {
                *reg = (f64::from(*reg) * neg_factor) as f32;
            }
        }
        self.base_log_pos = target_log_pos;
        self.base_log_neg = target_log_neg;
    }

    /// Run a full active-player visit in lazy mode and return the infoset's expected utility
    ///
    /// Cumulative regret is first caught up to the start of this iteration, then this iteration's
    /// counterfactual regret is accumulated, then regret matching and this iteration's discount are
    /// applied. Because the active infoset is reached at most once per traversal (perfect recall
    /// plus single-sampled chance and opponent), the whole update can happen here at the visit.
    fn visit_lazy(
        &mut self,
        player: &Player,
        lazy: &LazyState,
        discounts: &Discounts,
        rec: impl Fn(&Node) -> f64,
    ) -> f64 {
        // bring cumulative regret up to the start of this iteration
        self.catch_up(lazy.cum_log_pos, lazy.cum_log_neg);
        // accumulate this iteration's counterfactual regret
        let mut expected = 0.0;
        for ((next, prob), cum_reg) in player
            .actions
            .iter()
            .zip(self.reg.strat.iter())
            .zip(self.reg.cum_regret.iter_mut())
        {
            let util = rec(next);
            expected += f64::from(*prob) * util;
            *cum_reg += util as f32;
        }
        for cum_reg in &mut *self.reg.cum_regret {
            *cum_reg -= expected as f32;
        }
        // regret match for the next iteration and apply this iteration's discount
        discounts.advance_infoset(&mut self.reg.cum_regret, &mut self.reg.strat);
        // this infoset is now caught up through the current iteration's discount
        self.base_log_pos = lazy.cum_log_pos + lazy.log_pos_step;
        self.base_log_neg = lazy.cum_log_neg + lazy.log_neg_step;
        expected
    }

    /// Catch the infoset up through iteration `it` and return its regret-bound contribution
    fn catch_up_bound(&mut self, target_log_pos: f64, target_log_neg: f64, it: u64) -> f64 {
        self.catch_up(target_log_pos, target_log_neg);
        // it: iteration count, only loses precision past 2^53 iterations
        #[allow(clippy::cast_precision_loss)]
        let iters = it as f64;
        2.0 * f64::max(
            self.reg
                .cum_regret
                .iter()
                .map(|&reg| f64::from(reg))
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
/// unvisited infosets are never touched) and accumulates the opponent's average strategy with the
/// discount weight instead of discounting it.
fn recurse_lazy<const FIRST: bool>(
    node: &Node,
    chance_infosets: &[SampledChance],
    active_player_infosets: &[RefCell<CachedInfoset>],
    external_player_infosets: &[RefCell<CachedInfoset>],
    lazy: &LazyState,
    discounts: &Discounts,
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
            let next = chance_infosets[chance.infoset].sample(&mut key.rng(chance.infoset as u32));
            recurse_lazy::<FIRST>(
                &chance.outcomes[next],
                chance_infosets,
                active_player_infosets,
                external_player_infosets,
                lazy,
                discounts,
                key,
            )
        }
        Node::Player(player) => match (player.num, FIRST) {
            (PlayerNum::One, true) | (PlayerNum::Two, false) => active_player_infosets
                [player.infoset]
                .borrow_mut()
                .visit_lazy(player, lazy, discounts, |next| {
                    recurse_lazy::<FIRST>(
                        next,
                        chance_infosets,
                        active_player_infosets,
                        external_player_infosets,
                        lazy,
                        discounts,
                        key,
                    )
                }),
            (PlayerNum::Two, true) | (PlayerNum::One, false) => {
                let next = {
                    let mut info = external_player_infosets[player.infoset].borrow_mut();
                    info.accumulate_strat_weighted(lazy.strat_weight);
                    &player.actions[info.sample(key, player.infoset as u32)]
                };
                recurse_lazy::<FIRST>(
                    next,
                    chance_infosets,
                    active_player_infosets,
                    external_player_infosets,
                    lazy,
                    discounts,
                    key,
                )
            }
        },
    }
}

/// Lazy single-threaded external solving
///
/// Deferring the per-iteration discount work means each iteration only touches the infosets the
/// sample actually reaches; the rest are caught up lazily when next visited, or in full on the
/// (periodic) check iterations when the regret bound is evaluated.
pub(crate) fn solve_external_single(
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
    let [player_one, player_two] = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| RefCell::new(CachedInfoset::new(info.num_actions())))
            .collect::<Box<[_]>>()
    });
    let mut lazy = LazyState::default();
    let [mut reg_one, mut reg_two] = [f64::INFINITY; 2];
    for it in 1..=max_iter {
        let check = data::should_check(it, max_iter, check_interval);
        lazy.log_pos_step = RegretParams::gen_discount(it, params.pos_regret).ln();
        lazy.log_neg_step = RegretParams::gen_discount(it, params.neg_regret).ln();
        // discounting the average strategy by `(t/(t+1))^γ` each iteration is the same as weighting
        // iteration `t`'s contribution by `t^γ` and normalizing at the end
        // it: iteration count, only loses precision past 2^53 iterations
        #[allow(clippy::cast_precision_loss)]
        let iters = it as f64;
        lazy.strat_weight = iters.powf(params.strat);
        let discounts = Discounts::new(params, it, it);
        // the two sweeps sample independently (distinct sweep id in the key), but both read the same
        // pre-iteration cumulative discount, which advances once after both have run
        recurse_lazy::<true>(
            start,
            &chance_infosets,
            &player_one,
            &player_two,
            &lazy,
            &discounts,
            SampleKey::new(seed, it, 0),
        );
        recurse_lazy::<false>(
            start,
            &chance_infosets,
            &player_two,
            &player_one,
            &lazy,
            &discounts,
            SampleKey::new(seed, it, 1),
        );
        lazy.cum_log_pos += lazy.log_pos_step;
        lazy.cum_log_neg += lazy.log_neg_step;
        if check {
            // catch every infoset up through this iteration to evaluate the regret bound
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
}

#[cfg(test)]
mod tests {
    use crate::{
        Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, RegretParams, SolveParams,
    };

    mod eager {
        use super::super::CachedInfoset;
        use super::super::data::{Discounts, SampledChance, SampleKey, should_check, SolveInfo};
        use crate::{ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, SolveParams};
        use std::cell::RefCell;

        impl CachedInfoset {
            /// Accumulate this iteration's counterfactual regret over the active player's actions (eager)
            fn recurse_active(&mut self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64 {
                let mut expected = 0.0;
                for ((next, prob), cum_reg) in player
                    .actions
                    .iter()
                    .zip(self.reg.strat.iter())
                    .zip(self.reg.cum_regret.iter_mut())
                {
                    let util = rec(next);
                    expected += f64::from(*prob) * util;
                    *cum_reg += util as f32;
                }
                // account for only adding utility to cum_regret
                for cum_reg in &mut self.reg.cum_regret {
                    *cum_reg -= expected as f32;
                }
                expected
            }

            /// Regret match the next strategy and discount the cumulative regret (eager)
            fn advance(&mut self, discounts: &Discounts) -> f64 {
                let bound = discounts.advance_infoset(&mut self.reg.cum_regret, &mut self.reg.strat);
                discounts.discount_average_strat(&mut self.reg.cum_strat);
                bound
            }

            /// Add the current strategy into the average (eager)
            fn update_cum_strat(&mut self) {
                for (val, cum) in self.reg.strat.iter().zip(self.reg.cum_strat.iter_mut()) {
                    *cum += f64::from(*val);
                }
            }
        }

        fn recurse_regret<const FIRST: bool>(
            node: &Node,
            chance_infosets: &[SampledChance],
            active_player_infosets: &[RefCell<CachedInfoset>],
            external_player_infosets: &[RefCell<CachedInfoset>],
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
                        key,
                    )
                }
                Node::Player(player) => match (player.num, FIRST) {
                    (PlayerNum::One, true) | (PlayerNum::Two, false) => active_player_infosets
                        [player.infoset]
                        .borrow_mut()
                        .recurse_active(player, |next| {
                            recurse_regret::<FIRST>(
                                next,
                                chance_infosets,
                                active_player_infosets,
                                external_player_infosets,
                                key,
                            )
                        }),
                    (PlayerNum::One, false) | (PlayerNum::Two, true) => {
                        let next = {
                            let mut info = external_player_infosets[player.infoset].borrow_mut();
                            info.update_cum_strat();
                            &player.actions[info.sample(key, player.infoset as u32)]
                        };
                        recurse_regret::<FIRST>(
                            next,
                            chance_infosets,
                            active_player_infosets,
                            external_player_infosets,
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
                // player one (the two sweeps sample independently via distinct sweep ids in the key)
                recurse_regret::<true>(
                    start,
                    &chance_infosets,
                    &player_one,
                    &player_two,
                    SampleKey::new(seed, it, 0),
                );
                // the leading player has nothing accumulated on its first average-strat discount, so it
                // discounts against `it - 1`
                let discounts_one = Discounts::new(params, it, it - 1);
                reg_one = advance_player(&mut player_one, &discounts_one, check);
                // player two
                recurse_regret::<false>(
                    start,
                    &chance_infosets,
                    &player_two,
                    &player_one,
                    SampleKey::new(seed, it, 1),
                );
                let discounts_two = Discounts::new(params, it, it);
                reg_two = advance_player(&mut player_two, &discounts_two, check);
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

    type Game = (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]);

    fn simple_game() -> Game {
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

    fn even_or_odd() -> Game {
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
        };
        // the deferred (lazy) production solver and the eager reference must agree
        for eager in [false, true] {
            let (root, chance, [one, two]) = simple_game();
            let ([reg_one, reg_two], [strat_one, strat_two]) = if eager {
                eager::solve_external_eager(&root, &chance, [&*one, &*two], 1000, 0.0, &params)
            } else {
                super::solve_external_single(&root, &chance, [&*one, &*two], 1000, 0.0, &params)
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
        };
        for eager in [false, true] {
            let (root, chance, [one, two]) = even_or_odd();
            let ([reg_one, reg_two], [strat_one, strat_two]) = if eager {
                eager::solve_external_eager(&root, &chance, [&*one, &*two], 10_000, 0.005, &params)
            } else {
                super::solve_external_single(&root, &chance, [&*one, &*two], 10_000, 0.005, &params)
            };
            assert!((strat_one[1] - 0.5).abs() < 0.05, "{strat_one:?} eager={eager}");
            assert!((strat_two[0] - 0.5).abs() < 0.05, "{strat_two:?} eager={eager}");
            assert!(reg_one < 0.05, "eager={eager}");
            assert!(reg_two < 0.05, "eager={eager}");
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
            // run the global cumulative log-discount forward, mirroring solve_external_lazy
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
