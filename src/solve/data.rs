//! Common data structures for solving games
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
use crate::splitmix::{GAMMA, finalize, fold};
use core::convert::Infallible;
use logaddexp::LogAddExp;
use rand::TryRng;
use rand_distr::{Distribution, weighted::WeightedAliasIndex};

/// A deterministic, seedable counter-based RNG (splitmix64) for sampling
///
/// Driving the samplers with an RNG keyed by `(seed, iteration, sweep, infoset)` makes a solve
/// reproducible -- even in parallel, since a draw depends only on its key and not on thread
/// scheduling -- and gives external sampling its consistency (the same infoset draws the same way
/// within a sweep) without any shared cache.
#[derive(Debug)]
pub(super) struct DetRng(u64);

impl DetRng {
    fn step(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GAMMA);
        finalize(self.0)
    }
}

impl TryRng for DetRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok((self.step() >> 32) as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.step())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        let (chunks, rem) = dst.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = self.step().to_le_bytes();
        }
        if !rem.is_empty() {
            let bytes = self.step().to_le_bytes();
            rem.copy_from_slice(&bytes[..rem.len()]);
        }
        Ok(())
    }
}

/// The sampling key for one sweep: seed + iteration + sweep, producing a per-infoset [`DetRng`]
#[derive(Debug, Clone, Copy)]
pub(super) struct SampleKey(u64);

impl SampleKey {
    pub(super) fn new(seed: u64, iteration: u64, sweep: u64) -> Self {
        SampleKey(fold(fold(seed, iteration), sweep))
    }

    /// A fresh deterministic RNG for the given infoset
    pub(super) fn rng(self, infoset: u32) -> DetRng {
        DetRng(fold(self.0, u64::from(infoset)))
    }
}

/// A chance information set
#[derive(Debug)]
pub struct SampledChance {
    index: WeightedAliasIndex<f64>,
}

impl SampledChance {
    /// Create a new sampled chance from a set of probabilities
    pub fn new(probs: &[f64]) -> Self {
        SampledChance {
            index: WeightedAliasIndex::new(probs.to_vec()).unwrap(),
        }
    }

    /// Deterministically sample a chance outcome index from the given RNG
    pub(super) fn sample(&self, rng: &mut DetRng) -> usize {
        self.index.sample(rng)
    }
}

/// One action's accumulators within an infoset.
///
/// Field order `(f32, f32, f64)` packs to 16 bytes with no padding, so all three accumulators live
/// in one allocation and the ones an update touches together stay in the same cache line. Cumulative
/// regret and the current strategy are `f32` to halve their footprint (sums over them use `f64`); the
/// average-strategy accumulator stays `f64` because it grows large (the discount weighting can be
/// `tᵞ`) and is summed over every iteration.
#[derive(Debug, Clone, Default)]
pub(super) struct Action {
    // cumulative regret and the current strategy are f64 quantities kept as f32 for storage, reached
    // through the accessors below; the average-strategy accumulator is f64 throughout
    cum_regret: f32,
    strat: f32,
    pub(super) cum_strat: f64,
}

impl Action {
    /// An action at strategy probability `strat`, with regret and average strategy zeroed.
    fn with_strat(strat: f32) -> Action {
        Action {
            strat,
            ..Default::default()
        }
    }

    /// The cumulative regret (widened from its f32 storage).
    pub(super) fn regret(&self) -> f64 {
        self.cum_regret.into()
    }

    /// Store `value` as the cumulative regret, rounding to the f32 storage.
    pub(super) fn set_regret(&mut self, value: f64) {
        self.cum_regret = value as f32;
    }

    /// Add `value` to the cumulative regret in f64, rounding to f32 only on store.
    pub(super) fn add_regret(&mut self, value: f64) {
        self.set_regret(self.regret() + value);
    }

    /// The current strategy probability (widened from its f32 storage).
    pub(super) fn strat(&self) -> f64 {
        self.strat.into()
    }

    /// Store `value` as the current strategy probability, rounding to the f32 storage.
    pub(super) fn set_strat(&mut self, value: f64) {
        self.strat = value as f32;
    }
}

/// A player infoset for doing regret matching and tracking regret
///
/// The per-action accumulators live in one boxed slice of [`Action`]s.
#[derive(Debug)]
pub struct RegretInfoset {
    pub(super) actions: Box<[Action]>,
}

impl RegretInfoset {
    /// Create a new regret infoset given the number of actions
    pub fn new(num_actions: usize) -> Self {
        // (f32, f32, f64) packs to 16 bytes with no padding, so an infoset is one tight allocation
        debug_assert_eq!(size_of::<Action>(), 16, "Action should stay packed");
        let uniform = 1.0 / num_actions as f32;
        RegretInfoset {
            actions: (0..num_actions)
                .map(|_| Action::with_strat(uniform))
                .collect(),
        }
    }

    /// The current per-action strategy probabilities, for sampling.
    pub(super) fn strat(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.actions.iter().map(|action| action.strat)
    }

    /// Convert this into its average strategy
    pub fn into_avg_strat(self) -> Box<[f64]> {
        let mut avg: Box<[f64]> = self.actions.iter().map(|action| action.cum_strat).collect();
        avg_strat(&mut avg);
        avg
    }
}

/// Normalize a cumulative strategy
pub fn avg_strat(cum_strat: &mut [f64]) {
    let norm: f64 = cum_strat.iter().sum();
    if norm == 0.0 {
        cum_strat.fill(1.0 / cum_strat.len() as f64);
    } else {
        for prob in cum_strat.iter_mut() {
            *prob /= norm;
        }
    }
}

/// the result returned from solve functions
pub type SolveInfo = ([f64; 2], [Box<[f64]>; 2]);

/// Whether iteration `it` should evaluate the regret bound and test for early termination
///
/// Evaluating the regret bound is a reduction over every infoset, so rather than checking it every
/// iteration we only check every `check_interval` iterations (and always on the final iteration). As
/// a result solving runs at most `check_interval - 1` iterations past the point the regret target is
/// first met.
pub(super) fn should_check(it: u64, max_iter: u64, check_interval: u64) -> bool {
    it.is_multiple_of(check_interval) || it == max_iter
}

/// Advanced parameters for regret calculation
///
/// These parameters govern fine tuned details about how regrets and average strategies are
/// updated, and come mostly from[^dcfr]. In theory these only have constant factor effects on the
/// regret bounds, but in practice they can drastically speed up finding low regret strategies.
///
/// [^dcfr]: [Brown, Noam, and Tuomas Sandholm. "Solving imperfect-information games via discounted
///   regret minimization." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33.
///   No. 01. (2019)](https://ojs.aaai.org/index.php/AAAI/article/view/4007/3885)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegretParams {
    /// The discount factor for positive cumulative regret or `α`.
    ///
    /// Positive cumulative regrets are discounted by `tᵅ/(tᵅ + 1)` every iteration `t`. Setting
    /// alpha closer to infinity implies no discounting, while setting it at negative infinity
    /// means immediate forgetting. Note that any non-positive value is probably not desired.
    pub pos_regret: f64,
    /// The discount factor for negative cumulative regret or `β`
    ///
    /// Negative cumulative regrets are discounted by `tᵝ/(tᵝ + 1)` every iteration `t`. The
    /// values are the same as for positive regrets. Setting this to a non-positive value will
    /// prevent the cumulative regret of negative regret actions from approaching negative
    /// infinity, which can make pruning negative regret actions impossible.
    pub neg_regret: f64,
    /// The average strategy discount factor `γ`
    ///
    /// The average strategy is discounted by `(ᵗ⁄ₜ₊₁)ᵞ` every iteration t, which is equivalent to
    /// weighting each strategy update by `tᵞ`.
    pub strat: f64,
    /// The scale for picking a strategy when all regrets are negative
    ///
    /// If all actions have negative regret, the chosen strategy can be anything. We use the
    /// softmax of the regrets times this weight. Setting it to infinity is the same as always
    /// playing the strategy with the highest regret. Zero is equivalent to playing each action
    /// uniformly. No other values are recommend, but interpolate between those extremes.
    pub no_positive: f64,
}

impl RegretParams {
    /// Create a new arbitrary set of regret parameters
    ///
    /// # Panics
    ///
    /// If any values are nan, or `strat` is negative.
    #[must_use]
    pub fn new(pos_regret: f64, neg_regret: f64, strat: f64, no_positive: f64) -> Self {
        assert!(
            !pos_regret.is_nan(),
            "positive regret discounting can't be nan"
        );
        assert!(
            !neg_regret.is_nan(),
            "negative regret discounting can't be nan"
        );
        assert!(
            strat >= 0.0,
            "strategy discounting must be non-negative: {strat}"
        );
        assert!(
            strat != f64::INFINITY,
            "strategy discounting must be finite",
        );
        assert!(
            !no_positive.is_nan(),
            "no positive regret weight can't be nan"
        );
        RegretParams {
            pos_regret,
            neg_regret,
            strat,
            no_positive,
        }
    }

    /// Parameters for vanilla CFR
    ///
    /// Vanilla CFR has no discounting and picks the uniform strategy for negative regret infosets.
    #[must_use]
    pub fn vanilla() -> Self {
        RegretParams {
            pos_regret: f64::INFINITY,
            neg_regret: f64::INFINITY,
            strat: 0.0,
            no_positive: 0.0,
        }
    }

    /// Linear discounted CFR
    ///
    /// Regret and strategies are weighted proportional to the iteration number.
    #[must_use]
    pub fn lcfr() -> Self {
        RegretParams {
            pos_regret: 1.0,
            neg_regret: 1.0,
            strat: 1.0,
            no_positive: f64::INFINITY,
        }
    }

    /// CFR+
    ///
    /// Negative regrets are forgotten and strategies are weighted proportional to the square of
    /// the iteration number.
    #[must_use]
    pub fn cfr_plus() -> Self {
        RegretParams {
            pos_regret: f64::INFINITY,
            neg_regret: f64::NEG_INFINITY,
            strat: 2.0,
            no_positive: f64::INFINITY,
        }
    }

    /// Discounted CFR
    ///
    /// The loosely empirically optimal selection of parameters from the DCFR paper. (α: 1.5, β: 0, γ: 2)
    #[must_use]
    pub fn dcfr() -> Self {
        RegretParams {
            pos_regret: 1.5,
            neg_regret: 0.0,
            strat: 2.0,
            no_positive: f64::INFINITY,
        }
    }

    /// Discounted CFR with pruning
    ///
    /// A slight modification of DCFRs parameters that allows pruning negative regret actions. (α:
    /// 1.5, β: 0.5, γ: 2)
    #[must_use]
    pub fn dcfr_prune() -> Self {
        RegretParams {
            pos_regret: 1.5,
            neg_regret: 0.5,
            strat: 2.0,
            no_positive: f64::INFINITY,
        }
    }

    pub(super) fn regret_match(&self, actions: &mut [Action]) {
        let norm: f64 = actions
            .iter()
            .map(Action::regret)
            .filter(|value| *value > 0.0)
            .sum();
        if norm > 0.0 {
            for action in actions.iter_mut() {
                action.set_strat(if action.regret() > 0.0 {
                    action.regret() / norm
                } else {
                    0.0
                });
            }
        } else if self.no_positive == f64::INFINITY {
            let ind = actions
                .iter()
                .enumerate()
                .max_by(|(_, l), (_, r)| l.regret().partial_cmp(&r.regret()).unwrap())
                .unwrap()
                .0;
            for (i, action) in actions.iter_mut().enumerate() {
                action.set_strat(f64::from(i == ind));
            }
        } else if self.no_positive == 0.0 {
            let uniform = 1.0 / actions.len() as f64;
            for action in actions.iter_mut() {
                action.set_strat(uniform);
            }
        } else if self.no_positive == f64::NEG_INFINITY {
            let ind = actions
                .iter()
                .enumerate()
                .min_by(|(_, l), (_, r)| l.regret().partial_cmp(&r.regret()).unwrap())
                .unwrap()
                .0;
            for (i, action) in actions.iter_mut().enumerate() {
                action.set_strat(f64::from(i == ind));
            }
        } else {
            let max = actions.iter().map(Action::regret).reduce(f64::max).unwrap();
            let norm: f64 = actions
                .iter()
                .map(|action| ((action.regret() - max) * self.no_positive).exp())
                .sum();
            for action in actions.iter_mut() {
                action.set_strat(((action.regret() - max) * self.no_positive).exp() / norm);
            }
        }
    }

    pub(super) fn gen_discount(it: u64, discount: f64) -> f64 {
        if discount == f64::NEG_INFINITY {
            0.0
        } else if discount == 0.0 {
            0.5
        } else if discount == f64::INFINITY {
            1.0
        } else {
            let numer = discount * (it as f64).ln();
            let denom = numer.ln_add_exp(0.0);
            (numer - denom).exp()
        }
    }

    /// The `(positive-regret, negative-regret, average-strategy)` discount factors for iteration
    /// `it`. This lets the tree-free solver apply the same DCFR discounting the materialized solver
    /// does, without depending on its f32 storage internals.
    pub(crate) fn iteration_factors(&self, it: u64) -> (f64, f64, f64) {
        let strat = if self.strat == f64::INFINITY {
            0.0
        } else if self.strat > 0.0 {
            let float = it as f64;
            (float / (float + 1.0)).powf(self.strat)
        } else {
            1.0
        };
        (
            RegretParams::gen_discount(it, self.pos_regret),
            RegretParams::gen_discount(it, self.neg_regret),
            strat,
        )
    }
}

/// The average strategy discount for a single iteration
///
/// The discount applied to the average strategy depends only on the iteration, so we resolve it
/// once per sweep instead of branching on `strat` for every infoset.
#[derive(Debug)]
enum StratDiscount {
    /// Forget the entire accumulated average (`γ` is infinite)
    Forget,
    /// Scale the accumulated average by a factor (`γ > 0`)
    Scale(f64),
    /// Leave the accumulated average untouched (`γ == 0`)
    Keep,
}

/// Precomputed discount factors for a single update sweep
///
/// The regret and average-strategy discounts depend only on the iteration number, not on the
/// infoset, so they're computed once per sweep here rather than recomputing the underlying
/// transcendental functions ([`RegretParams::gen_discount`] and `powf`) for every infoset. A
/// borrowed [`RegretParams`] is carried along so callers can run regret matching through the same
/// handle.
#[derive(Debug)]
pub(super) struct Discounts<'a> {
    params: &'a RegretParams,
    it: u64,
    // factor applied to positive cumulative regret
    pos: f64,
    // factor applied to negative cumulative regret
    neg: f64,
    // factor applied to the accumulated average strategy
    strat: StratDiscount,
}

impl<'a> Discounts<'a> {
    /// Resolve the discounts for iteration `it`
    ///
    /// `strat_it` is the iteration used for the average-strategy discount, which can differ from
    /// `it` when players alternate updates (the lagging player hasn't accumulated anything yet).
    pub(super) fn new(params: &'a RegretParams, it: u64, strat_it: u64) -> Self {
        let strat = if params.strat == f64::INFINITY {
            StratDiscount::Forget
        } else if params.strat > 0.0 {
            let float = strat_it as f64;
            StratDiscount::Scale((float / (float + 1.0)).powf(params.strat))
        } else {
            StratDiscount::Keep
        };
        Discounts {
            params,
            it,
            pos: RegretParams::gen_discount(it, params.pos_regret),
            neg: RegretParams::gen_discount(it, params.neg_regret),
            strat,
        }
    }

    /// Run a full infoset update and return its regret-bound contribution
    ///
    /// This performs regret matching (writing the next strategy), discounts the cumulative regret,
    /// and computes the regret bound. The common case -- at least one action has positive
    /// cumulative regret -- is fused into a single pass over the cumulative regret that writes the
    /// strategy, applies the discount, and tracks the running maximum used for the bound. The rare
    /// all-non-positive case (where the strategy is chosen by the `no_positive` policy rather than
    /// proportionally to regret) falls back to the standalone primitives.
    pub(super) fn advance_infoset(&self, actions: &mut [Action]) -> f64 {
        let positive_norm: f64 = actions
            .iter()
            .map(Action::regret)
            .filter(|value| *value > 0.0)
            .sum();
        if positive_norm > 0.0 {
            let mut max = f64::NEG_INFINITY;
            for action in actions.iter_mut() {
                let cur = action.regret();
                action.set_strat(if cur > 0.0 { cur / positive_norm } else { 0.0 });
                let discounted = if cur > 0.0 {
                    cur * self.pos
                } else if cur < 0.0 {
                    cur * self.neg
                } else {
                    cur
                };
                action.set_regret(discounted);
                // read back the stored f32 so the bound matches the standalone primitives exactly
                max = f64::max(max, action.regret());
            }
            2.0 * f64::max(max, 0.0) / self.it as f64
        } else {
            self.params.regret_match(actions);
            self.discount_cum_regret(actions);
            self.regret_bound(actions)
        }
    }

    /// Discount cumulative regret with the precomputed factors
    pub(super) fn discount_cum_regret(&self, actions: &mut [Action]) {
        for action in actions.iter_mut() {
            if action.regret() > 0.0 {
                action.set_regret(action.regret() * self.pos);
            } else if action.regret() < 0.0 {
                action.set_regret(action.regret() * self.neg);
            }
        }
    }

    /// Discount the accumulated average strategy with the precomputed factor
    pub(super) fn discount_average_strat(&self, actions: &mut [Action]) {
        match self.strat {
            StratDiscount::Forget => {
                for action in actions.iter_mut() {
                    action.cum_strat = 0.0;
                }
            }
            StratDiscount::Scale(ratio) => {
                for action in actions.iter_mut() {
                    action.cum_strat *= ratio;
                }
            }
            StratDiscount::Keep => {}
        }
    }

    /// The regret bound contributed by an infoset given its cumulative regret
    pub(super) fn regret_bound(&self, actions: &[Action]) -> f64 {
        2.0 * f64::max(
            actions
                .iter()
                .map(Action::regret)
                .reduce(f64::max)
                .unwrap_or(0.0),
            0.0,
        ) / self.it as f64
    }
}

/// Defaults to [Discounted CFR (DCFR)][RegretParams::dcfr]
impl Default for RegretParams {
    fn default() -> Self {
        Self::dcfr()
    }
}

/// None of the values can be [nan][f64::NAN], so equality is well behaved
impl Eq for RegretParams {}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    unused_must_use
)]
mod tests {
    use super::{Action, Discounts, RegretParams};

    fn from_regret(regrets: &[f32]) -> Vec<Action> {
        regrets
            .iter()
            .map(|&cum_regret| Action {
                cum_regret,
                strat: 0.0,
                cum_strat: 0.0,
            })
            .collect()
    }

    fn from_cum_strat(cum_strats: &[f64]) -> Vec<Action> {
        cum_strats
            .iter()
            .map(|&cum_strat| Action {
                cum_regret: 0.0,
                strat: 0.0,
                cum_strat,
            })
            .collect()
    }

    fn strats(actions: &[Action]) -> Vec<f32> {
        actions.iter().map(|action| action.strat).collect()
    }

    fn regrets(actions: &[Action]) -> Vec<f32> {
        actions.iter().map(|action| action.cum_regret).collect()
    }

    fn cum_strats(actions: &[Action]) -> Vec<f64> {
        actions.iter().map(|action| action.cum_strat).collect()
    }

    #[test]
    fn avg_strat() {
        let mut strat = [1.0, 2.0, 1.0];
        super::avg_strat(&mut strat);
        assert_eq!(strat, [0.25, 0.5, 0.25]);
    }

    #[test]
    fn regret_match() {
        let mut acts = from_regret(&[1.0, 2.0, 1.0, -3.0]);
        RegretParams::new(0.0, 0.0, 0.0, 0.0).regret_match(&mut acts);
        assert_eq!(strats(&acts), [0.25, 0.5, 0.25, 0.0]);

        let mut acts = from_regret(&[-1.0, -2.0]);

        RegretParams::new(0.0, 0.0, 0.0, 0.0).regret_match(&mut acts);
        assert_eq!(strats(&acts), [0.5, 0.5]);

        RegretParams::new(0.0, 0.0, 0.0, f64::INFINITY).regret_match(&mut acts);
        assert_eq!(strats(&acts), [1.0, 0.0]);

        RegretParams::new(0.0, 0.0, 0.0, f64::NEG_INFINITY).regret_match(&mut acts);
        assert_eq!(strats(&acts), [0.0, 1.0]);

        RegretParams::new(0.0, 0.0, 0.0, 1.0).regret_match(&mut acts);
        let strat = strats(&acts);
        assert!((0.6..0.9).contains(&strat[0]));
        assert!((0.1..0.4).contains(&strat[1]));

        RegretParams::new(0.0, 0.0, 0.0, -1.0).regret_match(&mut acts);
        let strat = strats(&acts);
        assert!((0.1..0.4).contains(&strat[0]));
        assert!((0.6..0.9).contains(&strat[1]));
    }

    #[test]
    fn discount_regs() {
        let mut acts = from_regret(&[1.0, -1.0]);
        Discounts::new(&RegretParams::new(0.0, 0.0, 0.0, 0.0), 2, 2).discount_cum_regret(&mut acts);
        assert_eq!(regrets(&acts), [0.5, -0.5]);

        let mut acts = from_regret(&[1.0, -1.0]);
        Discounts::new(
            &RegretParams::new(f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0),
            2,
            2,
        )
        .discount_cum_regret(&mut acts);
        assert_eq!(regrets(&acts), [1.0, 0.0]);

        let mut acts = from_regret(&[1.0, -1.0]);
        Discounts::new(
            &RegretParams::new(f64::NEG_INFINITY, f64::INFINITY, 0.0, 0.0),
            2,
            2,
        )
        .discount_cum_regret(&mut acts);
        assert_eq!(regrets(&acts), [0.0, -1.0]);

        let mut acts = from_regret(&[1.0, -1.0]);
        Discounts::new(&RegretParams::new(1.0, 1.0, 0.0, 0.0), 2, 2).discount_cum_regret(&mut acts);
        let regret = regrets(&acts);
        assert!((0.6..0.9).contains(&regret[0]), "{}", regret[0]);
        assert!((-0.9..-0.6).contains(&regret[1]), "{}", regret[1]);
    }

    #[test]
    fn discount_average_strat() {
        let mut acts = from_cum_strat(&[1.0, 2.0]);
        Discounts::new(&RegretParams::new(0.0, 0.0, 0.0, 0.0), 1, 1)
            .discount_average_strat(&mut acts);
        assert_eq!(cum_strats(&acts), [1.0, 2.0]);

        let mut acts = from_cum_strat(&[1.0, 2.0]);
        Discounts::new(&RegretParams::lcfr(), 1, 1).discount_average_strat(&mut acts);
        assert_eq!(cum_strats(&acts), [0.5, 1.0]);

        let mut acts = from_cum_strat(&[1.0, 2.0]);
        Discounts::new(&RegretParams::cfr_plus(), 1, 1).discount_average_strat(&mut acts);
        assert_eq!(cum_strats(&acts), [0.25, 0.5]);

        let mut acts = from_cum_strat(&[1.0, 2.0]);
        Discounts::new(&RegretParams::lcfr(), 2, 2).discount_average_strat(&mut acts);
        let cum_strat = cum_strats(&acts);
        assert!((0.6..0.9).contains(&cum_strat[0]), "{}", cum_strat[0]);
        assert!((1.1..1.9).contains(&cum_strat[1]), "{}", cum_strat[1]);

        let params = RegretParams::lcfr();
        let mut acts = from_cum_strat(&[0.0]);
        for t in 1..=5 {
            acts[0].cum_strat += 1.0;
            Discounts::new(&params, t, t).discount_average_strat(&mut acts);
        }
        let expected = (1..=5).sum::<usize>() as f64 / (5 + 1) as f64;
        assert!((cum_strats(&acts)[0] - expected).abs() < 1e-6, "{}", cum_strats(&acts)[0]);
    }

    #[test]
    fn regret_bound() {
        let acts = from_regret(&[1.0, 2.0, 1.0, -3.0]);
        let res = Discounts::new(&RegretParams::vanilla(), 1, 1).regret_bound(&acts);
        assert_eq!(res, 4.0);
        let res = Discounts::new(&RegretParams::vanilla(), 2, 2).regret_bound(&acts);
        assert_eq!(res, 2.0);
        let res = Discounts::new(&RegretParams::vanilla(), 4, 4).regret_bound(&acts);
        assert_eq!(res, 1.0);
    }

    // The fused `advance_infoset` must produce exactly the same strategy, discounted cumulative
    // regret, and regret bound as running regret matching, regret discounting, and the regret bound
    // separately.
    #[test]
    fn advance_infoset_matches_unfused() {
        let cases = [
            vec![1.0, 2.0, 1.0, -3.0], // mixed signs, positive norm
            vec![5.0, 0.0, -1.0],      // includes an exact zero
            vec![-1.0, -2.0, -0.5],    // all non-positive, hits the fallback
            vec![0.0, 0.0],            // all zero, hits the fallback
            vec![3.0],                 // single positive action
        ];
        let params_list = [
            RegretParams::vanilla(),
            RegretParams::dcfr(),
            RegretParams::cfr_plus(),
            RegretParams::lcfr(),
        ];
        for params in &params_list {
            for it in [1_u64, 2, 7] {
                for case in &cases {
                    let discounts = Discounts::new(params, it, it);

                    let mut fused = from_regret(case);
                    let fused_bound = discounts.advance_infoset(&mut fused);

                    let mut acts = from_regret(case);
                    params.regret_match(&mut acts);
                    discounts.discount_cum_regret(&mut acts);
                    let bound = discounts.regret_bound(&acts);

                    assert_eq!(strats(&fused), strats(&acts), "strat {params:?} it={it} {case:?}");
                    assert_eq!(
                        regrets(&fused),
                        regrets(&acts),
                        "cum_regret {params:?} it={it} {case:?}"
                    );
                    assert_eq!(fused_bound, bound, "bound {params:?} it={it} {case:?}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "positive regret")]
    fn nan_pos() {
        RegretParams::new(f64::NAN, 0.0, 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "negative regret")]
    fn nan_neg() {
        RegretParams::new(0.0, f64::NAN, 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "strategy discounting")]
    fn nan_strat() {
        RegretParams::new(0.0, 0.0, f64::NAN, 0.0);
    }

    #[test]
    #[should_panic(expected = "strategy discounting")]
    fn neg_strat() {
        RegretParams::new(0.0, 0.0, -1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "strategy discounting")]
    fn inf_strat() {
        RegretParams::new(0.0, 0.0, f64::INFINITY, 0.0);
    }

    #[test]
    #[should_panic(expected = "no positive regret weight")]
    fn nan_no_pos() {
        RegretParams::new(0.0, 0.0, 0.0, f64::NAN);
    }
}
