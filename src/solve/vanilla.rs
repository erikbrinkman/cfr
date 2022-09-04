use super::data::{RegretInfoset, SampledChance};
use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
use std::cell::RefCell;

trait ChanceRecurse: Sized {
    fn recurse(
        &self,
        chance: &Chance,
        chance_infosets: &[Self],
        player_infosets: [&[impl PlayerRecurse]; 2],
        p_chance: f64,
        p_player: [f64; 2],
    ) -> f64;

    fn advance(&mut self) {}
}

#[derive(Debug)]
struct FullChance<'a>(&'a [f64]);

impl<'a> ChanceRecurse for FullChance<'a> {
    fn recurse(
        &self,
        chance: &Chance,
        chance_infosets: &[Self],
        player_infosets: [&[impl PlayerRecurse]; 2],
        p_chance: f64,
        p_player: [f64; 2],
    ) -> f64 {
        let mut expected = 0.0;
        for (prob, next) in self.0.iter().zip(chance.outcomes.iter()) {
            expected += prob
                * recurse(
                    next,
                    chance_infosets,
                    player_infosets,
                    p_chance * prob,
                    p_player,
                );
        }
        expected
    }
}

impl ChanceRecurse for SampledChance {
    fn recurse(
        &self,
        chance: &Chance,
        chance_infosets: &[Self],
        player_infosets: [&[impl PlayerRecurse]; 2],
        p_chance: f64,
        p_player: [f64; 2],
    ) -> f64 {
        recurse(
            &chance.outcomes[self.sample()],
            chance_infosets,
            player_infosets,
            p_chance,
            p_player,
        )
    }

    fn advance(&mut self) {
        self.reset()
    }
}

trait PlayerRecurse: Sized {
    fn recurse(
        &self,
        player: &Player,
        chance_infosets: &[impl ChanceRecurse],
        player_infosets: [&[Self]; 2],
        p_chance: f64,
        p_player: [f64; 2],
    ) -> f64;

    fn advance(&mut self) -> f64;

    fn into_strat(self) -> Box<[f64]>;
}

impl<const AVG: bool> PlayerRecurse for RefCell<RegretInfoset<AVG>> {
    fn recurse(
        &self,
        player: &Player,
        chance_infosets: &[impl ChanceRecurse],
        player_infosets: [&[Self]; 2],
        p_chance: f64,
        p_player: [f64; 2],
    ) -> f64 {
        // unpack self as mutable : this is okay because of perfect recall
        let RegretInfoset {
            strat,
            cum_regret,
            cum_strat,
        } = &mut *self.borrow_mut();

        // update cumulative strategy
        let prob = player.num.ind(&p_player);
        for (val, cum) in strat.iter().zip(cum_strat.iter_mut()) {
            *cum += prob * val;
        }

        // get constant multiple of utility
        let mult = match (player.num, p_player) {
            (PlayerNum::One, [_, two]) => p_chance * two,
            (PlayerNum::Two, [one, _]) => -one * p_chance,
        };

        // recurse and get expected utility
        let mut expected_one = 0.0;
        let mut expected = 0.0;
        for ((next, prob), cum_reg) in player
            .actions
            .iter()
            .zip(strat.iter())
            .zip(cum_regret.iter_mut())
        {
            let mut p_next = p_player;
            *player.num.ind_mut(&mut p_next) *= prob;
            let util_one = recurse(next, chance_infosets, player_infosets, p_chance, p_next);
            let util = util_one * mult;
            expected_one += prob * util_one;
            expected += util * prob;
            *cum_reg += util;
        }

        // account for only adding utility to cum_regret
        for reg in cum_regret.iter_mut() {
            *reg -= expected;
        }
        expected_one
    }

    fn advance(&mut self) -> f64 {
        let borrowed = self.get_mut();
        borrowed.regret_match();
        borrowed.cum_regret()
    }

    fn into_strat(self) -> Box<[f64]> {
        self.into_inner().into_avg_strat()
    }
}

fn recurse(
    node: &Node,
    chance_infosets: &[impl ChanceRecurse],
    player_infosets: [&[impl PlayerRecurse]; 2],
    p_chance: f64,
    p_player: [f64; 2],
) -> f64 {
    match node {
        Node::Terminal(payoff) => *payoff,
        Node::Chance(chance) => chance_infosets[chance.infoset].recurse(
            chance,
            chance_infosets,
            player_infosets,
            p_chance,
            p_player,
        ),
        Node::Player(player) => player.num.ind(&player_infosets)[player.infoset].recurse(
            player,
            chance_infosets,
            player_infosets,
            p_chance,
            p_player,
        ),
    }
}

fn solve_generic(
    start: &Node,
    mut chance_infosets: Box<[impl ChanceRecurse]>,
    mut player_infosets: [Box<[impl PlayerRecurse]>; 2],
    iter: u64,
    max_reg: f64,
) -> ([f64; 2], [Box<[f64]>; 2]) {
    let mut regs = [f64::INFINITY; 2];
    for it in 1..=iter {
        let [player_one, player_two] = &player_infosets;
        recurse(
            start,
            &chance_infosets,
            [player_one, player_two],
            1.0,
            [1.0; 2],
        );
        chance_infosets.iter_mut().for_each(ChanceRecurse::advance);
        for (reg, infos) in regs.iter_mut().zip(player_infosets.iter_mut()) {
            let total: f64 = infos.iter_mut().map(PlayerRecurse::advance).sum();
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
            .flat_map(|info| Vec::from(info.into_strat()))
            .collect()
    });
    (regs, strats)
}

pub(crate) fn solve_full(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
) -> ([f64; 2], [Box<[f64]>; 2]) {
    let player_infosets = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| RefCell::new(RegretInfoset::<false>::new(info.num_actions())))
            .collect()
    });
    let chance_infosets = chance_info
        .iter()
        .map(|info| FullChance(info.probs()))
        .collect();
    solve_generic(start, chance_infosets, player_infosets, max_iter, max_reg)
}

pub(crate) fn solve_sampled(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
) -> ([f64; 2], [Box<[f64]>; 2]) {
    let player_infosets = player_info.map(|infos| {
        infos
            .iter()
            .map(|info| RefCell::new(RegretInfoset::<false>::new(info.num_actions())))
            .collect()
    });
    let chance_infosets = chance_info
        .iter()
        .map(|info| SampledChance::new(info.probs()))
        .collect();
    solve_generic(start, chance_infosets, player_infosets, max_iter, max_reg)
}

// FIXME implement multi threaded version

#[cfg(test)]
mod tests {
    use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};

    struct Pinfo(usize);

    impl PlayerInfoset for Pinfo {
        fn num_actions(&self) -> usize {
            self.0
        }

        fn prev_infoset(&self) -> Option<usize> {
            None
        }
    }

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
    fn test_full() {
        let (root, chance, [one, two]) = new_game();
        let ([reg_one, reg_two], [strat_one, strat_two]) =
            super::solve_full(&root, &*chance, [&*one, &*two], 100, 0.0);
        assert_eq!(*strat_one, [0.995, 0.005]);
        assert_eq!(*strat_two, [0.005, 0.995]);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }

    #[test]
    fn test_sampled() {
        let (root, chance, [one, two]) = new_game();
        let ([reg_one, reg_two], [strat_one, strat_two]) =
            super::solve_sampled(&root, &*chance, [&*one, &*two], 100, 0.0);
        assert!(strat_one[1] < 0.05);
        assert!(strat_two[0] < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }
}
