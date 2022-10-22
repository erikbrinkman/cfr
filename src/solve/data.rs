//! Common data structures for solving games
use crate::Node;
use by_address::ByAddress;
use logaddexp::LogAddExp;
use portable_atomic::AtomicF64;
use rand::thread_rng;
use rand_distr::{Distribution, WeightedAliasIndex};
use std::collections::HashMap;
use std::iter::FusedIterator;
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

    /// Sample a chace outcome index
    ///
    /// This will return the same value on successive calls until reset is called
    pub fn sample(&mut self) -> usize {
        if self.cached == 0 {
            let res = self.index.sample(&mut thread_rng());
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

/// Normalize a cumulartive strategy
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

/// A trait for the option of cached payoffs
///
/// When doing multi threaded solving sometimes we'll have payoff data cached for a node when
/// bringing back the result of the computation from various threads. This trait represents that
/// wither with a node hashmap, or an empty tuple for when no cached payoffs exist.
pub(super) trait CachedPayoff {
    fn get_payoff(&self, node: &Node) -> Option<f64>;
}

impl CachedPayoff for HashMap<ByAddress<&Node>, f64> {
    fn get_payoff(&self, node: &Node) -> Option<f64> {
        self.get(&ByAddress(node)).copied()
    }
}

impl CachedPayoff for () {
    fn get_payoff(&self, _: &Node) -> Option<f64> {
        None
    }
}

/// the result returned from solve functions
pub type SolveInfo = ([f64; 2], [Box<[f64]>; 2]);

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
    /// means imediate forgetting. Note that any non-positive value is probably not desired.
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

/// Trait for &mut [f64] or &mut [AtomicF64]
///
/// This is mut because we have mut in all instances, and AtomicF64 can be more efficient because
/// we don't need to synchronize.
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

impl<'a> IntoFloatsMut<'a> for &'a mut [AtomicF64] {
    type Floats = AtomicIter<'a>;

    fn into_floats_mut(self) -> Self::Floats {
        AtomicIter(self.iter_mut())
    }
}

pub(super) struct AtomicIter<'a>(slice::IterMut<'a, AtomicF64>);

impl<'a> Iterator for AtomicIter<'a> {
    type Item = &'a mut f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(AtomicF64::get_mut)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.len();
        (len, Some(len))
    }
}

impl<'a> FusedIterator for AtomicIter<'a> {}

impl<'a> ExactSizeIterator for AtomicIter<'a> {}

impl RegretParams {
    /// Create a new arbitrary set of regret parameters
    ///
    /// # Panics
    ///
    /// If any values are nan, or `strat` is negative.
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
            "strategy discounting must be non-negative: {}",
            strat
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

    pub(super) fn discount_cum_regret<R: ?Sized>(&self, it: u64, cum_reg: &mut R)
    where
        for<'a> &'a mut R: IntoFloatsMut<'a>,
    {
        let pos = Self::gen_discount(it, self.pos_regret);
        let neg = Self::gen_discount(it, self.neg_regret);
        for reg in cum_reg.into_floats_mut() {
            if reg > &mut 0.0 {
                *reg *= pos;
            } else if reg < &mut 0.0 {
                *reg *= neg;
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

    pub(super) fn discount_average_strat(&self, it: u64, avg_strat: &mut [f64]) {
        if self.strat == f64::INFINITY {
            for avg in avg_strat.iter_mut() {
                *avg = 0.0;
            }
        } else if self.strat > 0.0 {
            let float = it as f64;
            let ratio = (float / (float + 1.0)).powf(self.strat);
            for avg in avg_strat.iter_mut() {
                *avg *= ratio;
            }
        }
    }

    pub(super) fn cum_regret<R: ?Sized>(&self, it: u64, cum_reg: &mut R) -> f64
    where
        for<'a> &'a mut R: IntoFloatsMut<'a>,
    {
        2.0 * f64::max(
            cum_reg
                .into_floats_mut()
                .map(|&mut r| r)
                .reduce(f64::max)
                .unwrap_or(0.0),
            0.0,
        ) / it as f64
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
mod tests {
    use super::RegretParams;

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
        RegretParams::new(0.0, 0.0, 0.0, 0.0).discount_cum_regret(2, &mut regs);
        assert_eq!(regs, [0.5, -0.5]);

        let mut regs = [1.0, -1.0];
        RegretParams::new(f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0)
            .discount_cum_regret(2, &mut regs);
        assert_eq!(regs, [1.0, 0.0]);

        let mut regs = [1.0, -1.0];
        RegretParams::new(f64::NEG_INFINITY, f64::INFINITY, 0.0, 0.0)
            .discount_cum_regret(2, &mut regs);
        assert_eq!(regs, [0.0, -1.0]);

        let mut regs = [1.0, -1.0];
        RegretParams::new(1.0, 1.0, 0.0, 0.0).discount_cum_regret(2, &mut regs);
        let [a, b] = regs;
        assert!((0.6..0.9).contains(&a), "{}", a);
        assert!((-0.9..-0.6).contains(&b), "{}", b);
    }

    #[test]
    fn discount_average_strat() {
        let mut cum_strat = [1.0, 2.0];
        RegretParams::new(0.0, 0.0, 0.0, 0.0).discount_average_strat(1, &mut cum_strat);
        assert_eq!(cum_strat, [1.0, 2.0]);

        let mut cum_strat = [1.0, 2.0];
        RegretParams::lcfr().discount_average_strat(1, &mut cum_strat);
        assert_eq!(cum_strat, [0.5, 1.0]);

        let mut cum_strat = [1.0, 2.0];
        RegretParams::cfr_plus().discount_average_strat(1, &mut cum_strat);
        assert_eq!(cum_strat, [0.25, 0.5]);

        let mut cum_strat = [1.0, 2.0];
        RegretParams::lcfr().discount_average_strat(2, &mut cum_strat);
        let [a, b] = cum_strat;
        assert!((0.6..0.9).contains(&a), "{}", a);
        assert!((1.1..1.9).contains(&b), "{}", b);

        let params = RegretParams::lcfr();
        let mut cum_strat = [0.0];
        for t in 1..=5 {
            cum_strat[0] += 1.0;
            params.discount_average_strat(t, &mut cum_strat);
        }
        let expected = (1..=5).sum::<usize>() as f64 / (5 + 1) as f64;
        assert!((cum_strat[0] - expected).abs() < 1e-6, "{}", cum_strat[0]);
    }

    #[test]
    fn cum_regret() {
        let mut regs = [1.0, 2.0, 1.0, -3.0];
        let res = RegretParams::default().cum_regret(1, &mut regs);
        assert_eq!(res, 4.0);
        let res = RegretParams::default().cum_regret(2, &mut regs);
        assert_eq!(res, 2.0);
        let res = RegretParams::default().cum_regret(4, &mut regs);
        assert_eq!(res, 1.0);
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
