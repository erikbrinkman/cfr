//! private module for computing regret
use super::{ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
use std::mem;

// NOTE Some of these methods could be written to use thread pools, but it's not clear that this is
// a large bottleneck so it's not worth the complexity

pub(super) fn expected(
    node: &Node,
    chance_info: &[impl ChanceInfoset],
    strat_info: [&[impl AsRef<[f64]>]; 2],
) -> f64 {
    let mut queue = vec![(node, 1.0)];
    let mut expected = 0.0;
    while let Some((node, reach)) = queue.pop() {
        match node {
            Node::Terminal(payoff) => {
                expected += reach * payoff;
            }
            Node::Chance(chance) => {
                let probs = chance_info[chance.infoset].probs();
                for (prob, next) in probs.iter().zip(chance.outcomes.iter()) {
                    queue.push((next, prob * reach));
                }
            }
            Node::Player(player) => {
                let probs = player.num.ind(&strat_info)[player.infoset].as_ref();
                for (prob, next) in probs.iter().zip(player.actions.iter()) {
                    if prob > &0.0 {
                        queue.push((next, prob * reach));
                    }
                }
            }
        }
    }
    expected
}

#[derive(Default, Debug)]
struct DeviationInfo<'a> {
    future_nodes: usize,
    prob_nodes: Vec<(&'a Player, f64)>,
    max_utility: f64,
}

fn optimal_deviations<const PLAYER_ONE: bool>(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: &[impl PlayerInfoset],
    strat_info: &[impl AsRef<[f64]>],
) -> f64 {
    let mut infosets: Box<[_]> = player_info
        .iter()
        .map(|_| DeviationInfo::default())
        .collect();
    let mut search_queue = vec![(start, 1.0)];
    while let Some((node, reach)) = search_queue.pop() {
        match node {
            Node::Terminal(_) => (),
            Node::Chance(chance) => {
                let probs = chance_info[chance.infoset].probs();
                for (prob, next) in probs.iter().zip(chance.outcomes.iter()) {
                    search_queue.push((next, prob * reach));
                }
            }
            Node::Player(player) => match (player.num, PLAYER_ONE) {
                (PlayerNum::One, true) | (PlayerNum::Two, false) => {
                    infosets[player.infoset].prob_nodes.push((player, reach));
                    if let Some(prev) = player_info[player.infoset].prev_infoset() {
                        infosets[prev].future_nodes += 1;
                    }
                    for next in player.actions.iter() {
                        search_queue.push((next, reach));
                    }
                }
                (PlayerNum::One, false) | (PlayerNum::Two, true) => {
                    let probs = strat_info[player.infoset].as_ref();
                    for (prob, next) in probs.iter().zip(player.actions.iter()) {
                        if prob > &0.0 {
                            search_queue.push((next, prob * reach));
                        }
                    }
                }
            },
        }
    }

    let mut info_queue: Vec<_> = infosets
        .iter()
        .enumerate()
        .filter(|(_, dev)| dev.future_nodes == 0 && !dev.prob_nodes.is_empty())
        .map(|(info, _)| info)
        .collect();
    while let Some(info) = info_queue.pop() {
        // get iteration nodes and compute total probability of reach for normalization
        let nodes = mem::take(&mut infosets[info].prob_nodes);
        let total_reach: f64 = nodes.iter().map(|(_, p)| p).sum();

        // check if finishing this infoset will allow us to evaluate a new infoset
        if let Some(prev) = player_info[info].prev_infoset() {
            let futs = &mut infosets[prev].future_nodes;
            *futs -= nodes.len();
            // if so, add it to the queue
            if futs == &mut 0 {
                info_queue.push(prev);
            }
        }

        // get the expected payoff of each action
        let mut payoffs = vec![0.0; player_info[info].num_actions()];
        for (player, prob) in nodes {
            for (next, res) in player.actions.iter().zip(payoffs.iter_mut()) {
                *res += next_infoset_search::<PLAYER_ONE>(
                    next,
                    &mut search_queue,
                    &*infosets,
                    chance_info,
                    strat_info,
                ) * prob;
            }
        }

        // set the max utility of playing to reach an infoset
        infosets[info].max_utility = payoffs.into_iter().reduce(f64::max).unwrap() / total_reach;
    }
    next_infoset_search::<PLAYER_ONE>(
        start,
        &mut search_queue,
        &*infosets,
        chance_info,
        strat_info,
    )
}

fn next_infoset_search<'a, const PLAYER_ONE: bool>(
    start: &'a Node,
    search_queue: &mut Vec<(&'a Node, f64)>,
    infosets: &[DeviationInfo],
    chance_info: &[impl ChanceInfoset],
    strat_info: &[impl AsRef<[f64]>],
) -> f64 {
    let mut res = 0.0;
    search_queue.push((start, 1.0));
    while let Some((node, reach)) = search_queue.pop() {
        match node {
            Node::Terminal(payoff) => {
                if PLAYER_ONE {
                    res += payoff * reach;
                } else {
                    res -= payoff * reach;
                }
            }
            Node::Chance(chance) => {
                let probs = chance_info[chance.infoset].probs();
                for (prob, next) in probs.iter().zip(chance.outcomes.iter()) {
                    search_queue.push((next, prob * reach));
                }
            }
            Node::Player(player) => match (player.num, PLAYER_ONE) {
                (PlayerNum::One, true) | (PlayerNum::Two, false) => {
                    let info = &infosets[player.infoset];
                    debug_assert_eq!(info.prob_nodes.len(), 0);
                    debug_assert_eq!(info.future_nodes, 0);
                    res += info.max_utility * reach;
                }
                (PlayerNum::One, false) | (PlayerNum::Two, true) => {
                    let probs = strat_info[player.infoset].as_ref();
                    for (prob, next) in probs.iter().zip(player.actions.iter()) {
                        if prob > &0.0 {
                            search_queue.push((next, prob * reach));
                        }
                    }
                }
            },
        }
    }
    res
}

pub(super) fn regret(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    strat_info: [&[impl AsRef<[f64]>]; 2],
) -> (f64, [f64; 2]) {
    let expected = expected(start, chance_info, strat_info);
    let one = optimal_deviations::<true>(start, chance_info, player_info[0], strat_info[1]);
    let two = optimal_deviations::<false>(start, chance_info, player_info[1], strat_info[0]);
    (expected, [one - expected, two + expected])
}
