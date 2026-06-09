//! Common data structures for solving games
#![allow(clippy::cast_precision_loss)]
use logaddexp::LogAddExp;
use rand::rng;
use rand_distr::{weighted::WeightedAliasIndex, Distribution};
use std::slice;

/// A chance information set for cached sampling
#[derive(Debug)]
pub struct SampledChance {
    index: WeightedAliasIndex<f64>,
    cached: usize,
}

impl SampledChance {
    /// Create a new sampled chance from a set of probabilities
    pub fn new(probs: &[f64]) -> Self {
        SampledChance {
            index: WeightedAliasIndex::new(probs.to_vec()).unwrap(),
            cached: 0,
        }
    }

    /// Sample a chance outcome index
    ///
    /// This will return the same value on successive calls until reset is called
    pub fn sample(&mut self) -> usize {
        if self.cached == 0 {
            let res = self.index.sample(&mut rng());
            self.cached = res + 1;
            res
        } else {
            self.cached - 1
        }
    }

    /// Reset the infoset allowing different samples
    pub fn reset(&mut self) {
        self.cached = 0;
    }
}

/// A player infoset for doing regret matching and tracking regret
#[derive(Debug)]
pub struct RegretInfoset {
    pub cum_regret: Box<[f64]>,
    pub cum_strat: Box<[f64]>,
    pub strat: Box<[f64]>,
}

impl RegretInfoset {
    /// Create a new regret infoset given the number of actions
    pub fn new(num_actions: usize) -> Self {
        RegretInfoset {
            cum_regret: vec![0.0; num_actions].into_boxed_slice(),
            cum_strat: vec![0.0; num_actions].into_boxed_slice(),
            strat: vec![1.0 / num_actions as f64; num_actions].into_boxed_slice(),
        }
    }

    /// Convert this into its average strategy
    pub fn into_avg_strat(mut self) -> Box<[f64]> {
        avg_strat(&mut self.cum_strat);
        self.cum_strat
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

/// Trait for `&mut [f64]` or `&mut [f64; N]`
///
/// This is mut because we have mut in all instances.
pub(super) trait IntoFloatsMut<'a> {
    type Floats: Iterator<Item = &'a mut f64>;

    fn into_floats_mut(self) -> Self::Floats;
}

impl<'a> IntoFloatsMut<'a> for &'a mut [f64] {
    type Floats = slice::IterMut<'a, f64>;

    fn into_floats_mut(self) -> Self::Floats {
        self.iter_mut()
    }
}

impl<'a, const N: usize> IntoFloatsMut<'a> for &'a mut [f64; N] {
    type Floats = slice::IterMut<'a, f64>;

    fn into_floats_mut(self) -> Self::Floats {
        self.iter_mut()
    }
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

    pub(super) fn regret_match<R: ?Sized>(&self, cum_reg: &mut R, strat: &mut [f64])
    where
        for<'a> &'a mut R: IntoFloatsMut<'a>,
    {
        debug_assert_eq!(cum_reg.into_floats_mut().count(), strat.len());
        let norm: f64 = cum_reg
            .into_floats_mut()
            .map(|&mut v| v)
            .filter(|v| v > &0.0)
            .sum();
        if norm > 0.0 {
            for (&mut reg, val) in cum_reg.into_floats_mut().zip(strat.iter_mut()) {
                *val = if reg > 0.0 { reg / norm } else { 0.0 }
            }
        } else if self.no_positive == f64::INFINITY {
            let (ind, _) = cum_reg
                .into_floats_mut()
                .enumerate()
                .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
                .unwrap();
            strat.fill(0.0);
            strat[ind] = 1.0;
        } else if self.no_positive == 0.0 {
            strat.fill(1.0 / strat.len() as f64);
        } else if self.no_positive == f64::NEG_INFINITY {
            let (ind, _) = cum_reg
                .into_floats_mut()
                .enumerate()
                .min_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
                .unwrap();
            strat.fill(0.0);
            strat[ind] = 1.0;
        } else {
            let max = cum_reg
                .into_floats_mut()
                .map(|&mut v| v)
                .reduce(f64::max)
                .unwrap();
            let norm: f64 = cum_reg
                .into_floats_mut()
                .map(|&mut reg| ((reg - max) * self.no_positive).exp())
                .sum();
            for (&mut reg, val) in cum_reg.into_floats_mut().zip(strat.iter_mut()) {
                *val = ((reg - max) * self.no_positive).exp() / norm;
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
    pub(super) fn advance_infoset<R: ?Sized>(&self, cum_reg: &mut R, strat: &mut [f64]) -> f64
    where
        for<'b> &'b mut R: IntoFloatsMut<'b>,
    {
        let positive_norm: f64 = cum_reg
            .into_floats_mut()
            .map(|&mut v| v)
            .filter(|v| *v > 0.0)
            .sum();
        if positive_norm > 0.0 {
            let mut max = f64::NEG_INFINITY;
            for (reg, val) in cum_reg.into_floats_mut().zip(strat.iter_mut()) {
                let cur = *reg;
                *val = if cur > 0.0 { cur / positive_norm } else { 0.0 };
                let discounted = if cur > 0.0 {
                    cur * self.pos
                } else if cur < 0.0 {
                    cur * self.neg
                } else {
                    cur
                };
                *reg = discounted;
                max = f64::max(max, discounted);
            }
            2.0 * f64::max(max, 0.0) / self.it as f64
        } else {
            self.params.regret_match(cum_reg, strat);
            self.discount_cum_regret(cum_reg);
            self.regret_bound(cum_reg)
        }
    }

    /// Discount cumulative regret with the precomputed factors
    pub(super) fn discount_cum_regret<R: ?Sized>(&self, cum_reg: &mut R)
    where
        for<'b> &'b mut R: IntoFloatsMut<'b>,
    {
        for reg in cum_reg.into_floats_mut() {
            if *reg > 0.0 {
                *reg *= self.pos;
            } else if *reg < 0.0 {
                *reg *= self.neg;
            }
        }
    }

    /// Discount the accumulated average strategy with the precomputed factor
    pub(super) fn discount_average_strat(&self, avg_strat: &mut [f64]) {
        match self.strat {
            StratDiscount::Forget => avg_strat.fill(0.0),
            StratDiscount::Scale(ratio) => {
                for avg in avg_strat.iter_mut() {
                    *avg *= ratio;
                }
            }
            StratDiscount::Keep => {}
        }
    }

    /// The regret bound contributed by an infoset given its cumulative regret
    pub(super) fn regret_bound<R: ?Sized>(&self, cum_reg: &mut R) -> f64
    where
        for<'b> &'b mut R: IntoFloatsMut<'b>,
    {
        2.0 * f64::max(
            cum_reg
                .into_floats_mut()
                .map(|&mut r| r)
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
    use super::{Discounts, RegretParams};

    #[test]
    fn avg_strat() {
        let mut strat = [1.0, 2.0, 1.0];
        super::avg_strat(&mut strat);
        assert_eq!(strat, [0.25, 0.5, 0.25]);
    }

    #[test]
    fn regret_match() {
        let mut regs = [1.0, 2.0, 1.0, -3.0];
        let mut strat = [0.0; 4];
        RegretParams::new(0.0, 0.0, 0.0, 0.0).regret_match(&mut regs, &mut strat);
        assert_eq!(strat, [0.25, 0.5, 0.25, 0.0]);

        let mut regs = [-1.0, -2.0];
        let mut strat = [0.0; 2];

        RegretParams::new(0.0, 0.0, 0.0, 0.0).regret_match(&mut regs, &mut strat);
        assert_eq!(strat, [0.5, 0.5]);

        RegretParams::new(0.0, 0.0, 0.0, f64::INFINITY).regret_match(&mut regs, &mut strat);
        assert_eq!(strat, [1.0, 0.0]);

        RegretParams::new(0.0, 0.0, 0.0, f64::NEG_INFINITY).regret_match(&mut regs, &mut strat);
        assert_eq!(strat, [0.0, 1.0]);

        RegretParams::new(0.0, 0.0, 0.0, 1.0).regret_match(&mut regs, &mut strat);
        let [a, b] = strat;
        assert!((0.6..0.9).contains(&a));
        assert!((0.1..0.4).contains(&b));

        RegretParams::new(0.0, 0.0, 0.0, -1.0).regret_match(&mut regs, &mut strat);
        let [a, b] = strat;
        assert!((0.1..0.4).contains(&a));
        assert!((0.6..0.9).contains(&b));
    }

    #[test]
    fn discount_regs() {
        let mut regs = [1.0, -1.0];
        Discounts::new(&RegretParams::new(0.0, 0.0, 0.0, 0.0), 2, 2).discount_cum_regret(&mut regs);
        assert_eq!(regs, [0.5, -0.5]);

        let mut regs = [1.0, -1.0];
        Discounts::new(&RegretParams::new(f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0), 2, 2)
            .discount_cum_regret(&mut regs);
        assert_eq!(regs, [1.0, 0.0]);

        let mut regs = [1.0, -1.0];
        Discounts::new(&RegretParams::new(f64::NEG_INFINITY, f64::INFINITY, 0.0, 0.0), 2, 2)
            .discount_cum_regret(&mut regs);
        assert_eq!(regs, [0.0, -1.0]);

        let mut regs = [1.0, -1.0];
        Discounts::new(&RegretParams::new(1.0, 1.0, 0.0, 0.0), 2, 2).discount_cum_regret(&mut regs);
        let [a, b] = regs;
        assert!((0.6..0.9).contains(&a), "{}", a);
        assert!((-0.9..-0.6).contains(&b), "{}", b);
    }

    #[test]
    fn discount_average_strat() {
        let mut cum_strat = [1.0, 2.0];
        Discounts::new(&RegretParams::new(0.0, 0.0, 0.0, 0.0), 1, 1)
            .discount_average_strat(&mut cum_strat);
        assert_eq!(cum_strat, [1.0, 2.0]);

        let mut cum_strat = [1.0, 2.0];
        Discounts::new(&RegretParams::lcfr(), 1, 1).discount_average_strat(&mut cum_strat);
        assert_eq!(cum_strat, [0.5, 1.0]);

        let mut cum_strat = [1.0, 2.0];
        Discounts::new(&RegretParams::cfr_plus(), 1, 1).discount_average_strat(&mut cum_strat);
        assert_eq!(cum_strat, [0.25, 0.5]);

        let mut cum_strat = [1.0, 2.0];
        Discounts::new(&RegretParams::lcfr(), 2, 2).discount_average_strat(&mut cum_strat);
        let [a, b] = cum_strat;
        assert!((0.6..0.9).contains(&a), "{}", a);
        assert!((1.1..1.9).contains(&b), "{}", b);

        let params = RegretParams::lcfr();
        let mut cum_strat = [0.0];
        for t in 1..=5 {
            cum_strat[0] += 1.0;
            Discounts::new(&params, t, t).discount_average_strat(&mut cum_strat);
        }
        let expected = (1..=5).sum::<usize>() as f64 / (5 + 1) as f64;
        assert!((cum_strat[0] - expected).abs() < 1e-6, "{}", cum_strat[0]);
    }

    #[test]
    fn regret_bound() {
        let mut regs = [1.0, 2.0, 1.0, -3.0];
        let res = Discounts::new(&RegretParams::vanilla(), 1, 1).regret_bound(&mut regs);
        assert_eq!(res, 4.0);
        let res = Discounts::new(&RegretParams::vanilla(), 2, 2).regret_bound(&mut regs);
        assert_eq!(res, 2.0);
        let res = Discounts::new(&RegretParams::vanilla(), 4, 4).regret_bound(&mut regs);
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

                    let mut fused_reg = case.clone();
                    let mut fused_strat = vec![0.0; case.len()];
                    let fused_bound =
                        discounts.advance_infoset(&mut fused_reg[..], &mut fused_strat);

                    let mut reg = case.clone();
                    let mut strat = vec![0.0; case.len()];
                    params.regret_match(&mut reg[..], &mut strat);
                    discounts.discount_cum_regret(&mut reg[..]);
                    let bound = discounts.regret_bound(&mut reg[..]);

                    assert_eq!(fused_strat, strat, "strat {params:?} it={it} {case:?}");
                    assert_eq!(fused_reg, reg, "cum_regret {params:?} it={it} {case:?}");
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
