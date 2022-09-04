use rand::thread_rng;
use rand_distr::{Distribution, WeightedAliasIndex};
use std::cell::RefCell;

#[derive(Debug)]
pub struct SampledChance {
    index: WeightedAliasIndex<f64>,
    cached: RefCell<usize>,
}

impl SampledChance {
    pub fn new(probs: &[f64]) -> Self {
        SampledChance {
            index: WeightedAliasIndex::new(probs.to_vec()).unwrap(),
            cached: RefCell::new(0),
        }
    }

    pub fn sample(&self) -> usize {
        let mut cached = self.cached.borrow_mut();
        if *cached == 0 {
            let res = self.index.sample(&mut thread_rng());
            *cached = res + 1;
            res
        } else {
            *cached - 1
        }
    }

    pub fn reset(&mut self) {
        *self.cached.get_mut() = 0;
    }
}

#[derive(Debug)]
pub struct RegretInfoset<const AVG: bool> {
    pub cum_regret: Box<[f64]>,
    pub cum_strat: Box<[f64]>,
    pub strat: Box<[f64]>,
}

impl<const AVG: bool> RegretInfoset<AVG> {
    pub fn new(num_actions: usize) -> RegretInfoset<AVG> {
        RegretInfoset {
            cum_regret: vec![0.0; num_actions].into_boxed_slice(),
            cum_strat: vec![0.0; num_actions].into_boxed_slice(),
            strat: vec![1.0 / num_actions as f64; num_actions].into_boxed_slice(),
        }
    }

    pub fn regret_match(&mut self) {
        let norm: f64 = self.cum_regret.iter().filter(|v| v > &&0.0).sum();
        if norm > 0.0 {
            for (reg, val) in self.cum_regret.iter().zip(self.strat.iter_mut()) {
                *val = if reg > &0.0 { reg / norm } else { 0.0 }
            }
        } else if AVG {
            self.strat.fill(1.0 / self.strat.len() as f64);
        } else {
            let (ind, _) = self
                .cum_regret
                .iter()
                .enumerate()
                .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
                .unwrap();
            self.strat.fill(0.0);
            self.strat[ind] = 1.0;
        }
    }

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

    pub fn into_avg_strat(mut self) -> Box<[f64]> {
        let norm: f64 = self.cum_strat.iter().sum();
        if norm == 0.0 {
            self.cum_strat.fill(1.0 / self.cum_strat.len() as f64);
        } else {
            for prob in self.cum_strat.iter_mut() {
                *prob /= norm;
            }
        }
        self.cum_strat
    }
}
