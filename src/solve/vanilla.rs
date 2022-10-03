//! Vanilla and sampled cfr implementations
use super::data;
use super::data::{CachedPayoff, RegretInfoset, SampledChance, SolveInfo};
use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
use by_address::ByAddress;
use portable_atomic::AtomicF64;
use rayon::iter::{ParallelDrainRange, ParallelExtend, ParallelIterator};
use rayon::{ThreadPoolBuildError, ThreadPoolBuilder};
use std::cell::RefCell;
use std::collections::HashMap;
use std::iter;
use std::iter::Zip;
use std::mem;
use std::num::NonZeroUsize;
use std::slice;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

type ChanceIter<'a, 'b> = Zip<slice::Iter<'a, f64>, slice::Iter<'b, Node>>;

// NOTE ideally this trait would define it's own iterator type, but without GAT we can't do that
// niesly
trait ChanceRecurse: Send {
    fn next_nodes<'a>(&self, chance: &'a Chance) -> ChanceIter<'_, 'a>;

    fn advance(&mut self);
}

#[derive(Debug)]
struct FullChance<'a>(&'a [f64]);

impl<'a> ChanceRecurse for FullChance<'a> {
    fn next_nodes<'b>(&self, chance: &'b Chance) -> ChanceIter<'_, 'b> {
        self.0.iter().zip(chance.outcomes.iter())
    }

    fn advance(&mut self) {}
}

impl ChanceRecurse for RefCell<SampledChance> {
    fn next_nodes<'b>(&self, chance: &'b Chance) -> ChanceIter<'_, 'b> {
        let ind = self.borrow_mut().sample();
        [1.0].iter().zip(chance.outcomes[ind..=ind].iter())
    }

    fn advance(&mut self) {
        self.get_mut().reset()
    }
}

impl ChanceRecurse for Mutex<SampledChance> {
    fn next_nodes<'b>(&self, chance: &'b Chance) -> ChanceIter<'_, 'b> {
        let ind = self.lock().unwrap().sample();
        [1.0].iter().zip(chance.outcomes[ind..=ind].iter())
    }

    fn advance(&mut self) {
        self.get_mut().unwrap().reset()
    }
}

trait PlayerRecurse {
    fn update_cum_strat(&mut self, prob: f64);

    fn advance(&mut self, temp: f64) -> f64;
}

impl PlayerRecurse for RegretInfoset {
    fn update_cum_strat(&mut self, prob: f64) {
        for (val, cum) in self.strat.iter().zip(self.cum_strat.iter_mut()) {
            *cum += prob * val;
        }
    }

    fn advance(&mut self, temp: f64) -> f64 {
        self.regret_match(temp);
        self.cum_regret()
    }
}

#[derive(Debug)]
struct MutexRegretInfoset {
    pub cum_regret: Box<[AtomicF64]>,
    pub cum_strat: Mutex<Box<[f64]>>,
    pub strat: Box<[f64]>,
}

impl MutexRegretInfoset {
    fn new(num_actions: usize) -> Self {
        MutexRegretInfoset {
            cum_regret: iter::repeat(())
                .take(num_actions)
                .map(|_| AtomicF64::new(0.0))
                .collect(),
            cum_strat: Mutex::new(vec![0.0; num_actions].into()),
            strat: vec![1.0 / num_actions as f64; num_actions].into(),
        }
    }

    fn regret_match(&mut self, temp: f64) {
        data::regret_match(
            self.cum_regret
                .iter()
                .map(|atomic| atomic.load(Ordering::Relaxed)),
            temp,
            &mut self.strat,
        )
    }

    fn cum_regret(&self) -> f64 {
        f64::max(
            self.cum_regret
                .iter()
                .map(|atomic| atomic.load(Ordering::Relaxed))
                .reduce(f64::max)
                .unwrap_or(0.0),
            0.0,
        )
    }

    fn into_avg_strat(self) -> Box<[f64]> {
        let mut cum_strat = self.cum_strat.into_inner().unwrap();
        data::avg_strat(&mut cum_strat);
        cum_strat
    }
}

trait MutexPlayerRecurse {
    fn update_cum_strat(&self, prob: f64);

    fn advance(&mut self, temp: f64) -> f64;
}

impl MutexPlayerRecurse for MutexRegretInfoset {
    fn update_cum_strat(&self, prob: f64) {
        for (val, cum) in self
            .strat
            .iter()
            .zip(self.cum_strat.lock().unwrap().iter_mut())
        {
            *cum += prob * val;
        }
    }

    fn advance(&mut self, temp: f64) -> f64 {
        self.regret_match(temp);
        self.cum_regret()
    }
}

// TODO there's a lot of duplication between this and recurse_multi, but the lack of GAT and
// complexities of the atomic cumulative regret make it hard to generalize. We try a bit with
// `recurse_player` but even that has its issues
fn recurse_single(
    node: &Node,
    chance_infosets: &[impl ChanceRecurse],
    player_infosets: [&[RefCell<RegretInfoset>]; 2],
    p_chance: f64,
    p_player: [f64; 2],
) -> f64 {
    match node {
        Node::Terminal(payoff) => *payoff,
        Node::Chance(chance) => {
            let mut expected = 0.0;
            for (prob, next) in chance_infosets[chance.infoset].next_nodes(chance) {
                let payoff = recurse_single(
                    next,
                    chance_infosets,
                    player_infosets,
                    p_chance * prob,
                    p_player,
                );
                expected += prob * payoff;
            }
            expected
        }
        Node::Player(player) => {
            // get infoset
            let mut info = player.num.ind(&player_infosets)[player.infoset].borrow_mut();
            info.update_cum_strat(*player.num.ind(&p_player));
            let RegretInfoset {
                strat, cum_regret, ..
            } = &mut *info;
            let (res, sub) = recurse_player(
                player,
                p_chance,
                p_player,
                strat,
                &mut **cum_regret,
                |next, p_next| {
                    recurse_single(next, chance_infosets, player_infosets, p_chance, p_next)
                },
            );
            for val in info.cum_regret.iter_mut() {
                *val -= sub;
            }
            res
        }
    }
}

trait Add {
    fn add(self, other: f64);
}

impl<'a> Add for &'a AtomicF64 {
    fn add(self, other: f64) {
        self.fetch_add(other, Ordering::Relaxed);
    }
}

impl<'a> Add for &'a mut f64 {
    fn add(self, other: f64) {
        *self += other;
    }
}

fn recurse_player(
    player: &Player,
    p_chance: f64,
    p_player: [f64; 2],
    strat: &[f64],
    cum_regret: impl IntoIterator<Item = impl Add>,
    rec: impl Fn(&Node, [f64; 2]) -> f64,
) -> (f64, f64) {
    let mult = match (player.num, p_player) {
        (PlayerNum::One, [_, two]) => p_chance * two,
        (PlayerNum::Two, [one, _]) => -one * p_chance,
    };

    let mut expected_one = 0.0;
    let mut expected = 0.0;
    for ((next, prob), cum_reg) in player
        .actions
        .iter()
        .zip(strat.iter())
        .zip(cum_regret.into_iter())
    {
        let mut p_next = p_player;
        *player.num.ind_mut(&mut p_next) *= prob;
        let util_one = rec(next, p_next);
        let util = util_one * mult;
        expected_one += prob * util_one;
        expected += util * prob;
        cum_reg.add(util);
    }
    (expected_one, expected)
}

fn recurse_multi(
    node: &Node,
    chance_infosets: &[impl ChanceRecurse],
    player_infosets: [&[MutexRegretInfoset]; 2],
    p_chance: f64,
    p_player: [f64; 2],
    cached: &impl CachedPayoff,
) -> f64 {
    match node {
        Node::Terminal(payoff) => *payoff,
        Node::Chance(chance) => {
            let mut expected = 0.0;
            for (prob, next) in chance_infosets[chance.infoset].next_nodes(chance) {
                let payoff = recurse_multi(
                    next,
                    chance_infosets,
                    player_infosets,
                    p_chance * prob,
                    p_player,
                    cached,
                );
                expected += prob * payoff;
            }
            expected
        }
        Node::Player(player) => {
            // get infoset
            let info = &player.num.ind(&player_infosets)[player.infoset];
            info.update_cum_strat(*player.num.ind(&p_player));
            let (res, sub) = recurse_player(
                player,
                p_chance,
                p_player,
                &info.strat,
                &*info.cum_regret,
                |next, p_next| {
                    recurse_multi(
                        next,
                        chance_infosets,
                        player_infosets,
                        p_chance,
                        p_next,
                        cached,
                    )
                },
            );
            for val in info.cum_regret.iter() {
                val.fetch_sub(sub, Ordering::Relaxed);
            }
            res
        }
    }
}

fn solve_generic_single(
    start: &Node,
    mut chance_infosets: Box<[impl ChanceRecurse]>,
    mut player_infosets: [Box<[RefCell<RegretInfoset>]>; 2],
    iter: u64,
    max_reg: f64,
    temp: f64,
) -> SolveInfo {
    let mut regs = [f64::INFINITY; 2];
    for it in 1..=iter {
        let [player_one, player_two] = &player_infosets;
        recurse_single(
            start,
            &chance_infosets,
            [player_one, player_two],
            1.0,
            [1.0; 2],
        );
        chance_infosets.iter_mut().for_each(ChanceRecurse::advance);
        for (reg, infos) in regs.iter_mut().zip(player_infosets.iter_mut()) {
            let total: f64 = infos
                .iter_mut()
                .map(|info| info.get_mut().advance(temp))
                .sum();
            *reg = 2.0 * total / it as f64;
        }
        let [reg_one, reg_two] = regs;
        if f64::max(reg_one, reg_two) < max_reg {
            break;
        }
    }
    let strats = player_infosets.map(|player| {
        Vec::from(player)
            .into_iter()
            .flat_map(|info| Vec::from(info.into_inner().into_avg_strat()))
            .collect()
    });
    (regs, strats)
}

/// Explore out from root returning a large enough frontier to make multi-threaded recursion fast
fn thread_threshold<'a>(
    root: &'a Node,
    chance_infosets: &[impl ChanceRecurse],
    mut player_infosets: [&mut [MutexRegretInfoset]; 2],
    target: NonZeroUsize,
    queue: &mut Vec<(&'a Node, f64, [f64; 2])>,
    work: &mut Vec<(&'a Node, f64, [f64; 2])>,
) {
    queue.push((root, 1.0, [1.0; 2]));
    while !(queue.is_empty() && work.is_empty()) && queue.len() + work.len() < target.get() {
        match queue.pop() {
            Some((Node::Terminal(_), _, _)) => {}
            Some((Node::Chance(chance), p_chance, p_player)) => {
                let info = &chance_infosets[chance.infoset];
                work.extend(
                    info.next_nodes(chance)
                        .map(|(prob, node)| (node, p_chance * prob, p_player)),
                );
            }
            Some((Node::Player(player), p_chance, p_player)) => {
                // NOTE get_mut is much faster than locking, so when possible we prefer it
                let probs = &player.num.ind_mut(&mut player_infosets)[player.infoset].strat;
                for (prob, next) in probs.iter().zip(player.actions.iter()) {
                    let mut next_probs = p_player;
                    *player.num.ind_mut(&mut next_probs) *= prob;
                    work.push((next, p_chance, next_probs));
                }
            }
            None => {
                mem::swap(queue, work);
            }
        }
    }
}

fn solve_generic_multi(
    start: &Node,
    mut chance_infosets: Box<[impl ChanceRecurse + Sync]>,
    mut player_infosets: [Box<[MutexRegretInfoset]>; 2],
    iter: u64,
    max_reg: f64,
    temp: f64,
    thread_info: (NonZeroUsize, NonZeroUsize),
) -> Result<SolveInfo, ThreadPoolBuildError> {
    let mut regs = [f64::INFINITY; 2];
    let (num_threads, target) = thread_info;
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads.get())
        .build()?;
    pool.scope(|_| {
        let mut queue = Vec::with_capacity(target.get());
        let mut work = Vec::with_capacity(target.get());
        let mut payoffs = HashMap::with_capacity(target.get());
        for it in 1..=iter {
            // compute threadding threshold
            let [player_one, player_two] = &mut player_infosets;
            thread_threshold(
                start,
                &chance_infosets,
                [player_one, player_two],
                target,
                &mut queue,
                &mut work,
            );
            // send threshold to threads for computation
            let [player_one, player_two] = &player_infosets;
            payoffs.par_extend(queue.par_drain(..).map(|(node, p_chance, p_player)| {
                let payoff = recurse_multi(
                    node,
                    &chance_infosets,
                    [player_one, player_two],
                    p_chance,
                    p_player,
                    &(),
                );
                (ByAddress(node), payoff)
            }));
            // search full from there
            recurse_multi(
                start,
                &chance_infosets,
                [player_one, player_two],
                1.0,
                [1.0; 2],
                &payoffs,
            );
            chance_infosets.iter_mut().for_each(ChanceRecurse::advance);
            for (reg, infos) in regs.iter_mut().zip(player_infosets.iter_mut()) {
                let total: f64 = infos.iter_mut().map(|info| info.advance(temp)).sum();
                *reg = 2.0 * total / it as f64;
            }
            let [reg_one, reg_two] = regs;
            if f64::max(reg_one, reg_two) < max_reg {
                break;
            }
        }
    });
    let strats = player_infosets.map(|player| {
        Vec::from(player)
            .into_iter()
            .flat_map(|info| Vec::from(info.into_avg_strat()))
            .collect()
    });
    Ok((regs, strats))
}

pub(crate) fn solve_full_single(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    temp: f64,
) -> SolveInfo {
    let player_infosets = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| RefCell::new(RegretInfoset::new(info.num_actions())))
            .collect()
    });
    let chance_infosets = chance_info
        .iter()
        .map(|info| FullChance(info.probs()))
        .collect();
    solve_generic_single(
        start,
        chance_infosets,
        player_infosets,
        max_iter,
        max_reg,
        temp,
    )
}

pub(crate) fn solve_full_multi(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    temp: f64,
    thread_info: (NonZeroUsize, NonZeroUsize),
) -> Result<SolveInfo, ThreadPoolBuildError> {
    let player_infosets = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| MutexRegretInfoset::new(info.num_actions()))
            .collect()
    });
    let chance_infosets = chance_info
        .iter()
        .map(|info| FullChance(info.probs()))
        .collect();
    solve_generic_multi(
        start,
        chance_infosets,
        player_infosets,
        max_iter,
        max_reg,
        temp,
        thread_info,
    )
}

pub(crate) fn solve_sampled_single(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    temp: f64,
) -> SolveInfo {
    let player_infosets = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| RefCell::new(RegretInfoset::new(info.num_actions())))
            .collect()
    });
    let chance_infosets = chance_info
        .iter()
        .map(|info| RefCell::new(SampledChance::new(info.probs())))
        .collect();
    solve_generic_single(
        start,
        chance_infosets,
        player_infosets,
        max_iter,
        max_reg,
        temp,
    )
}

pub(crate) fn solve_sampled_multi(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    temp: f64,
    thread_info: (NonZeroUsize, NonZeroUsize),
) -> Result<SolveInfo, ThreadPoolBuildError> {
    let player_infosets = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| MutexRegretInfoset::new(info.num_actions()))
            .collect()
    });
    let chance_infosets = chance_info
        .iter()
        .map(|info| Mutex::new(SampledChance::new(info.probs())))
        .collect();
    solve_generic_multi(
        start,
        chance_infosets,
        player_infosets,
        max_iter,
        max_reg,
        temp,
        thread_info,
    )
}

#[cfg(test)]
mod tests {
    use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
    use std::num::NonZeroUsize;

    #[derive(Clone, Debug)]
    struct Pinfo(usize);

    impl PlayerInfoset for Pinfo {
        fn num_actions(&self) -> usize {
            self.0
        }

        fn prev_infoset(&self) -> Option<usize> {
            None
        }
    }

    #[derive(Clone, Debug)]
    struct Cinfo(Box<[f64]>);

    impl ChanceInfoset for Cinfo {
        fn probs(&self) -> &[f64] {
            &*self.0
        }
    }

    fn new_game() -> (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]) {
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

    #[test]
    fn test_full_single() {
        let (root, chance, [one, two]) = new_game();
        let ([reg_one, reg_two], [strat_one, strat_two]) =
            super::solve_full_single(&root, &*chance, [&*one, &*two], 100, 0.0, 0.0);
        assert_eq!(*strat_one, [0.995, 0.005]);
        assert_eq!(*strat_two, [0.005, 0.995]);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }

    #[test]
    fn test_sampled_single() {
        let (root, chance, [one, two]) = new_game();
        let ([reg_one, reg_two], [strat_one, strat_two]) =
            super::solve_sampled_single(&root, &*chance, [&*one, &*two], 100, 0.0, 0.0);
        assert!(strat_one[1] < 0.05);
        assert!(strat_two[0] < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }

    fn recurse_simple_node(steps: usize, payoff: f64) -> Node {
        match (steps, steps % 3) {
            (0, _) => Node::Terminal(payoff),
            (_, 2) => Node::Chance(Chance {
                outcomes: vec![
                    recurse_simple_node(steps - 1, payoff),
                    recurse_simple_node(steps - 2, payoff),
                ]
                .into(),
                infoset: steps / 3,
            }),
            (_, num @ (0 | 1)) => Node::Player(Player {
                num: if num == 0 {
                    PlayerNum::One
                } else {
                    PlayerNum::Two
                },
                actions: [
                    Node::Terminal(0.0),
                    recurse_simple_node(steps - 1, payoff * -1.0),
                ]
                .into(),
                infoset: (steps - 1) / 3,
            }),
            _ => panic!(),
        }
    }

    fn large_game(num: usize) -> (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]) {
        let root = recurse_simple_node(num, 1.0);
        let chance = vec![Cinfo(vec![0.5, 0.5].into()); (num + 1) / 3].into();
        let players = [
            vec![Pinfo(2); num / 3].into(),
            vec![Pinfo(2); (num + 2) / 3].into(),
        ];
        (root, chance, players)
    }

    #[test]
    fn test_full_multi() {
        let (root, chance, [one, two]) = large_game(10);
        let ([reg_one, reg_two], _) = super::solve_full_multi(
            &root,
            &*chance,
            [&*one, &*two],
            10000,
            0.0,
            0.0,
            (NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(1).unwrap()),
        )
        .unwrap();
        assert!(f64::max(reg_one, reg_two) < 0.1, "{} {}", reg_one, reg_two);
    }

    #[test]
    fn test_sampled_multi() {
        let (root, chance, [one, two]) = large_game(10);
        let ([reg_one, reg_two], _) = super::solve_sampled_multi(
            &root,
            &*chance,
            [&*one, &*two],
            10000,
            0.0,
            0.0,
            (NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(1).unwrap()),
        )
        .unwrap();
        assert!(f64::max(reg_one, reg_two) < 0.1, "{} {}", reg_one, reg_two);
    }
}
