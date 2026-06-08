//! Vanilla and sampled cfr implementations
#![allow(clippy::cast_precision_loss)]
use super::data::{RegretInfoset, RegretParams, SampledChance, SolveInfo};
use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
use std::cell::RefCell;
use std::iter::Zip;
use std::slice;

type ChanceIter<'a, 'b> = Zip<slice::Iter<'a, f64>, slice::Iter<'b, Node>>;

// NOTE ideally this trait would define it's own iterator type, but without GAT we can't do that
// nicely
trait ChanceRecurse {
    fn next_nodes<'a>(&self, chance: &'a Chance) -> ChanceIter<'_, 'a>;

    fn advance(&mut self);
}

#[derive(Debug)]
struct FullChance<'a>(&'a [f64]);

impl ChanceRecurse for FullChance<'_> {
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
        self.get_mut().reset();
    }
}

trait PlayerRecurse {
    fn update_cum_strat(&mut self, prob: f64);

    fn advance(&mut self, it: u64, params: &RegretParams) -> f64;
}

impl PlayerRecurse for RegretInfoset {
    fn update_cum_strat(&mut self, prob: f64) {
        for (val, cum) in self.strat.iter().zip(self.cum_strat.iter_mut()) {
            *cum += prob * val;
        }
    }

    fn advance(&mut self, it: u64, params: &RegretParams) -> f64 {
        params.regret_match(&mut *self.cum_regret, &mut self.strat);
        params.discount_cum_regret(it, &mut *self.cum_regret);
        params.discount_average_strat(it, &mut self.cum_strat);
        RegretParams::cum_regret(it, &mut *self.cum_regret)
    }
}

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
            for val in &mut info.cum_regret {
                *val -= sub;
            }
            res
        }
    }
}

trait Add {
    fn add(self, other: f64);
}

impl Add for &mut f64 {
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
        .zip(cum_regret)
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

fn solve_generic_single(
    start: &Node,
    mut chance_infosets: Box<[impl ChanceRecurse]>,
    mut player_infosets: [Box<[RefCell<RegretInfoset>]>; 2],
    iter: u64,
    max_reg: f64,
    params: &RegretParams,
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
            *reg = infos
                .iter_mut()
                .map(|info| info.get_mut().advance(it, params))
                .sum();
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

pub(crate) fn solve_full_single(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    params: &RegretParams,
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
        params,
    )
}

pub(crate) fn solve_sampled_single(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    params: &RegretParams,
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
        params,
    )
}

#[cfg(test)]
mod tests {
    use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, RegretParams};

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
            &self.0
        }
    }

    type Game = (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]);

    fn new_game() -> Game {
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
        let ([reg_one, reg_two], [strat_one, strat_two]) = super::solve_full_single(
            &root,
            &chance,
            [&*one, &*two],
            100,
            0.0,
            &RegretParams::vanilla(),
        );
        assert_eq!(*strat_one, [0.995, 0.005]);
        assert_eq!(*strat_two, [0.005, 0.995]);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }

    #[test]
    fn test_sampled_single() {
        let (root, chance, [one, two]) = new_game();
        let ([reg_one, reg_two], [strat_one, strat_two]) = super::solve_sampled_single(
            &root,
            &chance,
            [&*one, &*two],
            100,
            0.0,
            &RegretParams::vanilla(),
        );
        assert!(strat_one[1] < 0.05);
        assert!(strat_two[0] < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }
}
