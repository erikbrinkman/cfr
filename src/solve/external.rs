//! External regret solving
use super::data::{CachedPayoff, RegretInfoset, SampledChance, SolveInfo};
use super::multinomial::Multinomial;
use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
use by_address::ByAddress;
use rand::thread_rng;
use rand_distr::Distribution;
use rayon::iter::{
    IntoParallelRefMutIterator, ParallelDrainRange, ParallelExtend, ParallelIterator,
};
use rayon::{ThreadPoolBuildError, ThreadPoolBuilder};
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem;
use std::num::NonZeroUsize;
use std::sync::Mutex;

/// A variant of the standard regret infoset that caches the last selected external sampled strat
#[derive(Debug)]
struct CachedInfoset {
    reg: RegretInfoset,
    cached: usize,
}

impl CachedInfoset {
    /// Create a new cached infoset
    fn new(num_actions: usize) -> Self {
        CachedInfoset {
            reg: RegretInfoset::new(num_actions),
            cached: 0,
        }
    }

    /// Sample an action from the current strategy, caches between resets
    fn sample(&mut self) -> usize {
        if self.cached == 0 {
            let res = Multinomial::new(&self.reg.strat).sample(&mut thread_rng());
            self.cached = res + 1;
            res
        } else {
            self.cached - 1
        }
    }
}

/// Extra implementations for chance infosets for external sampling
trait ChanceInfo {
    fn next<'a>(&mut self, chance: &'a Chance) -> &'a Node;

    fn advance(&mut self);
}

impl ChanceInfo for SampledChance {
    fn next<'a>(&mut self, chance: &'a Chance) -> &'a Node {
        &chance.outcomes[self.sample()]
    }

    fn advance(&mut self) {
        self.reset()
    }
}

/// Abstraction over wrappers of ChanceInfo
///
/// This allows recursing over RefCells or Mutex
trait ChanceRecurse {
    fn next<'a>(&self, chance: &'a Chance) -> &'a Node;
}

impl<T: ChanceInfo> ChanceRecurse for RefCell<T> {
    fn next<'a>(&self, chance: &'a Chance) -> &'a Node {
        self.borrow_mut().next(chance)
    }
}

impl<T: ChanceInfo> ChanceRecurse for Mutex<T> {
    fn next<'a>(&self, chance: &'a Chance) -> &'a Node {
        self.lock().unwrap().next(chance)
    }
}

trait ActiveInfo {
    fn recurse(&mut self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64;

    fn advance(&mut self, temp: f64) -> f64;
}

trait ActiveRecurse {
    fn recurse(&self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64;
}

impl<T: ActiveInfo> ActiveRecurse for RefCell<T> {
    fn recurse(&self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64 {
        self.borrow_mut().recurse(player, rec)
    }
}

impl<T: ActiveInfo> ActiveRecurse for Mutex<T> {
    fn recurse(&self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64 {
        // NOTE we technically don't need to lock here as the perfect recall guarantees ensure that
        // this is the unique visit to this infoset this iteration, however in practice switching
        // to unsafe rust didn't actually improve performance, likely because the locking isn't a
        // huge bottleneck
        self.try_lock().unwrap().recurse(player, rec)
    }
}

trait ExternalInfo {
    fn next<'a>(&mut self, player: &'a Player) -> &'a Node;

    fn update_cum_strat(&mut self);

    fn next_update<'a>(&mut self, player: &'a Player) -> &'a Node {
        self.update_cum_strat();
        self.next(player)
    }
}

trait ExternalRecurse {
    fn next_update<'a>(&self, player: &'a Player) -> &'a Node;
}

impl<T: ExternalInfo> ExternalRecurse for RefCell<T> {
    fn next_update<'a>(&self, player: &'a Player) -> &'a Node {
        self.borrow_mut().next_update(player)
    }
}

impl<T: ExternalInfo> ExternalRecurse for Mutex<T> {
    fn next_update<'a>(&self, player: &'a Player) -> &'a Node {
        self.lock().unwrap().next_update(player)
    }
}

impl ActiveInfo for CachedInfoset {
    fn recurse(&mut self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64 {
        // recurse and get expected utility
        let mut expected = 0.0;
        for ((next, prob), cum_reg) in player
            .actions
            .iter()
            .zip(self.reg.strat.iter())
            .zip(self.reg.cum_regret.iter_mut())
        {
            let util = rec(next);
            expected += prob * util;
            *cum_reg += util;
        }

        // account for only adding utility to cum_regret
        for cum_reg in self.reg.cum_regret.iter_mut() {
            *cum_reg -= expected;
        }
        expected
    }

    fn advance(&mut self, temp: f64) -> f64 {
        self.cached = 0;
        self.reg.regret_match(temp);
        self.reg.cum_regret()
    }
}

impl ExternalInfo for CachedInfoset {
    fn next<'a>(&mut self, player: &'a Player) -> &'a Node {
        &player.actions[self.sample()]
    }

    fn update_cum_strat<'a>(&mut self) {
        for (val, cum) in self.reg.strat.iter().zip(self.reg.cum_strat.iter_mut()) {
            *cum += val;
        }
    }
}

fn next_nodes<'a, const FIRST: bool>(
    mut node: &'a Node,
    chance_infosets: &mut [Mutex<SampledChance>],
    external_player_infosets: &mut [Mutex<CachedInfoset>],
) -> Option<impl IntoIterator<Item = &'a Node>> {
    loop {
        match node {
            Node::Terminal(_) => return None,
            Node::Chance(chance) => {
                node = chance_infosets[chance.infoset]
                    .get_mut()
                    .unwrap()
                    .next(chance);
            }
            Node::Player(player) => match (player.num, FIRST) {
                (PlayerNum::One, true) | (PlayerNum::Two, false) => {
                    return Some(&*player.actions);
                }
                (PlayerNum::Two, true) | (PlayerNum::One, false) => {
                    node = external_player_infosets[player.infoset]
                        .get_mut()
                        .unwrap()
                        .next(player);
                }
            },
        }
    }
}

fn recurse_regret<const FIRST: bool>(
    node: &Node,
    chance_infosets: &[impl ChanceRecurse],
    active_player_infosets: &[impl ActiveRecurse],
    external_player_infosets: &[impl ExternalRecurse],
    cached: &impl CachedPayoff,
) -> f64 {
    if let Some(pay) = cached.get_payoff(node) {
        pay
    } else {
        match node {
            Node::Terminal(payoff) => {
                if FIRST {
                    *payoff
                } else {
                    -payoff
                }
            }
            Node::Chance(chance) => recurse_regret::<FIRST>(
                chance_infosets[chance.infoset].next(chance),
                chance_infosets,
                active_player_infosets,
                external_player_infosets,
                cached,
            ),
            Node::Player(player) => match (player.num, FIRST) {
                (PlayerNum::One, true) | (PlayerNum::Two, false) => {
                    active_player_infosets[player.infoset].recurse(player, |next| {
                        recurse_regret::<FIRST>(
                            next,
                            chance_infosets,
                            active_player_infosets,
                            external_player_infosets,
                            cached,
                        )
                    })
                }
                (PlayerNum::One, false) | (PlayerNum::Two, true) => recurse_regret::<FIRST>(
                    external_player_infosets[player.infoset].next_update(player),
                    chance_infosets,
                    active_player_infosets,
                    external_player_infosets,
                    cached,
                ),
            },
        }
    }
}

/// Explore out from root returning a large enough frontier to make multi-threaded recursion fast
fn thread_threshold<'a, const FIRST: bool>(
    root: &'a Node,
    chance_infosets: &mut [Mutex<SampledChance>],
    external_player_infosets: &mut [Mutex<CachedInfoset>],
    target: NonZeroUsize,
    queue: &mut Vec<&'a Node>,
    work: &mut Vec<&'a Node>,
) {
    queue.push(root);
    while !(queue.is_empty() && work.is_empty()) && queue.len() + work.len() < target.get() {
        if let Some(node) = queue.pop() {
            if let Some(nexts) =
                next_nodes::<FIRST>(node, chance_infosets, external_player_infosets)
            {
                work.extend(nexts);
            }
        } else {
            mem::swap(queue, work);
        }
    }
}

/// workspace needed between iterations to avoid excess allocation
struct Workspace<'a> {
    queue: Vec<&'a Node>,
    work: Vec<&'a Node>,
    payoffs: HashMap<ByAddress<&'a Node>, f64>,
}

impl<'a> Workspace<'a> {
    fn with_capacity(capacity: usize) -> Self {
        Workspace {
            queue: Vec::with_capacity(capacity),
            work: Vec::with_capacity(capacity),
            payoffs: HashMap::with_capacity(capacity),
        }
    }
}

/// solve for a single player returning their regret
///
/// When doing single threaded we can just call recurse immediately, but we need to do some extra
/// prep here
fn single_player_iter<'a, const FIRST: bool>(
    root: &'a Node,
    chance_infosets: &mut [Mutex<SampledChance>],
    active_player_infosets: &mut [Mutex<CachedInfoset>],
    external_player_infosets: &mut [Mutex<CachedInfoset>],
    target: NonZeroUsize,
    temp: f64,
    work: &mut Workspace<'a>,
) -> f64 {
    // compute threashold of `target` nodes for efficient multi threading
    thread_threshold::<FIRST>(
        root,
        chance_infosets,
        external_player_infosets,
        target,
        &mut work.queue,
        &mut work.work,
    );
    // send threshold to threads for computation
    work.payoffs
        .par_extend(work.queue.par_drain(..).map(|node| {
            let payoff = recurse_regret::<FIRST>(
                node,
                chance_infosets,
                active_player_infosets,
                external_player_infosets,
                &(),
            );
            (ByAddress(node), payoff)
        }));
    // now actually recurse, having cached results from threaded computation
    recurse_regret::<FIRST>(
        root,
        chance_infosets,
        active_player_infosets,
        external_player_infosets,
        &work.payoffs,
    );

    // update all infosets
    work.payoffs.clear();
    chance_infosets
        .iter_mut()
        .for_each(|info| info.get_mut().unwrap().advance());
    active_player_infosets
        .par_iter_mut()
        .map(|info| info.get_mut().unwrap().advance(temp))
        .sum()
}

pub(crate) fn solve_external_multi(
    root: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    temp: f64,
    thread_info: (NonZeroUsize, NonZeroUsize),
) -> Result<SolveInfo, ThreadPoolBuildError> {
    let (num_threads, target) = thread_info;
    // NOTE these are Box's not Arcs so that at the end of the threaded computation we can move
    // them out
    let mut chance_infosets: Box<[_]> = chance_info
        .iter()
        .map(|info| Mutex::new(SampledChance::new(info.probs())))
        .collect();
    let [mut player_one, mut player_two] = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| Mutex::new(CachedInfoset::new(info.num_actions())))
            .collect::<Box<[_]>>()
    });
    let [mut reg_one, mut reg_two] = [f64::INFINITY; 2];

    // create channels
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads.get())
        .build()?;
    pool.scope(|_| {
        // initialize workspace
        let mut work = Workspace::with_capacity(target.get());

        // loop through iters, these will send data to to the threads
        for it in 1..=max_iter {
            reg_one = single_player_iter::<true>(
                root,
                &mut chance_infosets,
                &mut player_one,
                &mut player_two,
                target,
                temp,
                &mut work,
            );
            reg_one *= 2.0 / it as f64;
            reg_two = single_player_iter::<false>(
                root,
                &mut chance_infosets,
                &mut player_two,
                &mut player_one,
                target,
                temp,
                &mut work,
            );
            reg_two *= 2.0 / it as f64;
            // check to terminate
            if f64::max(reg_one, reg_two) < max_reg {
                break;
            }
        }
    });

    let strats = [player_one, player_two].map(|player| {
        Vec::from(player)
            .into_iter()
            .flat_map(|info| Vec::from(info.into_inner().unwrap().reg.into_avg_strat()))
            .collect()
    });
    Ok(([reg_one, reg_two], strats))
}

pub(crate) fn solve_external_single(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    temp: f64,
) -> SolveInfo {
    let mut chance_infosets: Box<[_]> = chance_info
        .iter()
        .map(|info| RefCell::new(SampledChance::new(info.probs())))
        .collect();
    let [mut player_one, mut player_two] = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| RefCell::new(CachedInfoset::new(info.num_actions())))
            .collect::<Box<[_]>>()
    });
    let [mut reg_one, mut reg_two] = [f64::INFINITY; 2];
    for it in 1..=max_iter {
        // player one
        recurse_regret::<true>(start, &chance_infosets, &player_one, &player_two, &());
        chance_infosets
            .iter_mut()
            .for_each(|info| info.get_mut().advance());
        reg_one = player_one
            .iter_mut()
            .map(|info| info.get_mut().advance(temp))
            .sum();
        reg_one *= 2.0 / it as f64;
        // player two
        recurse_regret::<false>(start, &chance_infosets, &player_two, &player_one, &());
        chance_infosets
            .iter_mut()
            .for_each(|info| info.get_mut().advance());
        reg_two = player_two
            .iter_mut()
            .map(|info| info.get_mut().advance(temp))
            .sum();
        reg_two *= 2.0 / it as f64;
        // check to terminate
        if f64::max(reg_one, reg_two) < max_reg {
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

#[cfg(test)]
mod tests {
    use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
    use std::num::NonZeroUsize;

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
            &*self.0
        }
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
    fn test_multi() {
        let (root, cinfo, [pone, ptwo]) = large_game(10);
        let ([reg_one, reg_two], _) = super::solve_external_multi(
            &root,
            &cinfo,
            [&pone, &ptwo],
            1000,
            0.0,
            0.0,
            (NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(1).unwrap()),
        )
        .unwrap();
        assert!(f64::max(reg_one, reg_two) < 0.1, "{} {}", reg_one, reg_two);
    }

    fn simple_game() -> (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]) {
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

    fn even_or_odd() -> (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]) {
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
        let (root, chance, [one, two]) = simple_game();
        let ([reg_one, reg_two], [strat_one, strat_two]) =
            super::solve_external_single(&root, &*chance, [&*one, &*two], 1000, 0.0, 0.0);
        assert!(strat_one[1] < 0.05);
        assert!(strat_two[0] < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }

    #[test]
    fn test_external_even_or_odd() {
        let (root, chance, [one, two]) = even_or_odd();
        let ([reg_one, reg_two], [strat_one, strat_two]) =
            super::solve_external_single(&root, &*chance, [&*one, &*two], 10000, 0.05, 0.0);
        assert!((strat_one[1] - 0.5).abs() < 0.05);
        assert!((strat_two[0] - 0.5).abs() < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }
}
