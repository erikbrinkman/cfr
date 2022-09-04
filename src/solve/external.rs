use super::data::{RegretInfoset, SampledChance};
use super::multinomial::Multinomial;
use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
use rand::thread_rng;
use rand_distr::Distribution;
use std::cell::RefCell;

trait ChanceRecurse: Sized {
    fn recurse<const FIRST: bool>(
        &self,
        chance: &Chance,
        chance_infosets: &[Self],
        active_player_infosets: &[impl ActivePlayerRecurse],
        external_player_infosets: &[impl ExternalPlayerRecurse],
    ) -> f64;

    fn advance(&mut self);
}

impl ChanceRecurse for SampledChance {
    fn recurse<const FIRST: bool>(
        &self,
        chance: &Chance,
        chance_infosets: &[Self],
        active_player_infosets: &[impl ActivePlayerRecurse],
        external_player_infosets: &[impl ExternalPlayerRecurse],
    ) -> f64 {
        recurse::<FIRST>(
            &chance.outcomes[self.sample()],
            chance_infosets,
            active_player_infosets,
            external_player_infosets,
        )
    }

    fn advance(&mut self) {
        self.reset()
    }
}

trait PlayerRecurse {
    fn advance(&mut self) -> f64;

    fn into_strat(self) -> Box<[f64]>;
}

trait ActivePlayerRecurse: Sized {
    fn recurse<const FIRST: bool>(
        &self,
        player: &Player,
        chance_infosets: &[impl ChanceRecurse],
        active_player_infosets: &[Self],
        external_player_infosets: &[impl ExternalPlayerRecurse],
    ) -> f64;
}

trait ExternalPlayerRecurse: Sized {
    fn recurse<const FIRST: bool>(
        &self,
        player: &Player,
        chance_infosets: &[impl ChanceRecurse],
        active_player_infosets: &[impl ActivePlayerRecurse],
        external_player_infosets: &[Self],
    ) -> f64;
}

#[derive(Debug)]
struct CachedInfoset<const AVG: bool> {
    reg: RegretInfoset<AVG>,
    cached: usize,
}

impl<const AVG: bool> CachedInfoset<AVG> {
    fn new(num_actions: usize) -> Self {
        CachedInfoset {
            reg: RegretInfoset::new(num_actions),
            cached: 0,
        }
    }

    fn sample(&mut self) -> usize {
        if self.cached == 0 {
            let res = Multinomial::new(&self.reg.strat).sample(&mut thread_rng());
            self.cached = res + 1;
            res
        } else {
            self.cached - 1
        }
    }

    fn advance(&mut self) -> f64 {
        self.cached = 0;
        self.reg.regret_match();
        self.reg.cum_regret()
    }
}

impl<const AVG: bool> PlayerRecurse for RefCell<CachedInfoset<AVG>> {
    fn advance(&mut self) -> f64 {
        self.get_mut().advance()
    }

    fn into_strat(self) -> Box<[f64]> {
        self.into_inner().reg.into_avg_strat()
    }
}

impl<const AVG: bool> ActivePlayerRecurse for RefCell<CachedInfoset<AVG>> {
    fn recurse<const FIRST: bool>(
        &self,
        player: &Player,
        chance_infosets: &[impl ChanceRecurse],
        active_player_infosets: &[Self],
        external_player_infosets: &[impl ExternalPlayerRecurse],
    ) -> f64 {
        // unpack self as mutable : this is okay because of perfect recall
        let RegretInfoset {
            strat, cum_regret, ..
        } = &mut self.borrow_mut().reg;

        // recurse and get expected utility
        let mut expected = 0.0;
        for ((next, prob), cum_reg) in player
            .actions
            .iter()
            .zip(strat.iter())
            .zip(cum_regret.iter_mut())
        {
            let util = recurse::<FIRST>(
                next,
                chance_infosets,
                active_player_infosets,
                external_player_infosets,
            );
            expected += prob * util;
            *cum_reg += util;
        }

        // account for only adding utility to cum_regret
        for cum_reg in cum_regret.iter_mut() {
            *cum_reg -= expected;
        }
        expected
    }
}

impl<const AVG: bool> ExternalPlayerRecurse for RefCell<CachedInfoset<AVG>> {
    fn recurse<const FIRST: bool>(
        &self,
        player: &Player,
        chance_infosets: &[impl ChanceRecurse],
        active_player_infosets: &[impl ActivePlayerRecurse],
        external_player_infosets: &[Self],
    ) -> f64 {
        // update cumulative strategy
        let mut borrowed = self.borrow_mut();
        let RegretInfoset {
            strat, cum_strat, ..
        } = &mut borrowed.reg;
        for (val, cum) in strat.iter().zip(cum_strat.iter_mut()) {
            *cum += val;
        }

        // then iterate
        recurse::<FIRST>(
            &player.actions[borrowed.sample()],
            chance_infosets,
            active_player_infosets,
            external_player_infosets,
        )
    }
}

fn recurse<const FIRST: bool>(
    node: &Node,
    chance_infosets: &[impl ChanceRecurse],
    active_player_infosets: &[impl ActivePlayerRecurse],
    external_player_infosets: &[impl ExternalPlayerRecurse],
) -> f64 {
    match node {
        Node::Terminal(payoff) => {
            if FIRST {
                *payoff
            } else {
                -payoff
            }
        }
        Node::Chance(chance) => chance_infosets[chance.infoset].recurse::<FIRST>(
            chance,
            chance_infosets,
            active_player_infosets,
            external_player_infosets,
        ),
        Node::Player(player) => match (player.num, FIRST) {
            (PlayerNum::One, true) | (PlayerNum::Two, false) => {
                active_player_infosets[player.infoset].recurse::<FIRST>(
                    player,
                    chance_infosets,
                    active_player_infosets,
                    external_player_infosets,
                )
            }
            (PlayerNum::One, false) | (PlayerNum::Two, true) => {
                external_player_infosets[player.infoset].recurse::<FIRST>(
                    player,
                    chance_infosets,
                    active_player_infosets,
                    external_player_infosets,
                )
            }
        },
    }
}

// FIXME multi threaded

pub(crate) fn solve_external(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
) -> ([f64; 2], [Box<[f64]>; 2]) {
    let mut chance_infosets: Box<[_]> = chance_info
        .iter()
        .map(|info| SampledChance::new(info.probs()))
        .collect();
    let [mut player_one, mut player_two] = player_info.map(|infos| -> Box<[_]> {
        infos
            .iter()
            .map(|info| RefCell::new(CachedInfoset::<false>::new(info.num_actions())))
            .collect()
    });
    let [mut reg_one, mut reg_two] = [f64::INFINITY; 2];
    for it in 1..=max_iter {
        // player one
        recurse::<true>(start, &chance_infosets, &player_one, &player_two);
        chance_infosets.iter_mut().for_each(ChanceRecurse::advance);
        reg_one = player_one.iter_mut().map(PlayerRecurse::advance).sum();
        reg_one *= 2.0 / it as f64;
        // player two
        recurse::<false>(start, &chance_infosets, &player_two, &player_one);
        chance_infosets.iter_mut().for_each(ChanceRecurse::advance);
        reg_two = player_two.iter_mut().map(PlayerRecurse::advance).sum();
        reg_two *= 2.0 / it as f64;
        // check to terminate
        if f64::max(reg_one, reg_two) < max_reg {
            break;
        }
    }
    let strats = [player_one, player_two].map(|player| {
        Vec::from(player)
            .into_iter()
            .flat_map(|info| Vec::from(info.into_strat()))
            .collect()
    });
    ([reg_one, reg_two], strats)
}

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
            super::solve_external(&root, &*chance, [&*one, &*two], 1000, 0.0);
        assert!(strat_one[1] < 0.05);
        assert!(strat_two[0] < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }

    #[test]
    fn test_external_even_or_odd() {
        let (root, chance, [one, two]) = even_or_odd();
        let ([reg_one, reg_two], [strat_one, strat_two]) =
            super::solve_external(&root, &*chance, [&*one, &*two], 10000, 0.05);
        assert!((strat_one[1] - 0.5).abs() < 0.05);
        assert!((strat_two[0] - 0.5).abs() < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }
}
