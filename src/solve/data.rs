//! Common data structures for solving games
use crate::Node;
use by_address::ByAddress;
use rand::thread_rng;
use rand_distr::{Distribution, WeightedAliasIndex};
use std::collections::HashMap;

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

    /// Performe regret matching to update the current strategy
    pub fn regret_match(&mut self, temp: f64) {
        regret_match(self.cum_regret.iter().copied(), temp, &mut self.strat);
    }

    /// Get the cumulative regret encountered so far
    pub fn cum_regret(&self) -> f64 {
        f64::max(
            self.cum_regret
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap_or(0.0),
            0.0,
        )
    }

    /// Convert this into its average strategy
    pub fn into_avg_strat(mut self) -> Box<[f64]> {
        avg_strat(&mut self.cum_strat);
        self.cum_strat
    }
}

/// Perform regret matching given cumulative regret and a temperature
pub fn regret_match(cum_reg: impl Iterator<Item = f64> + Clone, temp: f64, dest: &mut [f64]) {
    debug_assert!(temp >= 0.0);
    debug_assert_eq!(cum_reg.clone().count(), dest.len());
    let norm: f64 = cum_reg.clone().filter(|v| v > &0.0).sum();
    if norm > 0.0 {
        for (reg, val) in cum_reg.zip(dest.iter_mut()) {
            *val = if reg > 0.0 { reg / norm } else { 0.0 }
        }
    } else if temp == f64::INFINITY {
        dest.fill(1.0 / dest.len() as f64);
    } else if temp == 0.0 {
        let (ind, _) = cum_reg
            .enumerate()
            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
            .unwrap();
        dest.fill(0.0);
        dest[ind] = 1.0;
    } else {
        let max = cum_reg.clone().reduce(f64::max).unwrap();
        let norm: f64 = cum_reg.clone().map(|reg| ((reg - max) / temp).exp()).sum();
        for (reg, val) in cum_reg.zip(dest.iter_mut()) {
            *val = ((reg - max) / temp).exp() / norm;
        }
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
