//! External regret solving
use super::data;
use super::data::{CachedPayoff, Discounts, RegretInfoset, RegretParams, SampledChance, SolveInfo};
use super::multinomial::Multinomial;
use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum};
use rand::distr::Distribution;
use rand::rng;
use std::cell::RefCell;

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
            let res = Multinomial::new(&self.reg.strat).sample(&mut rng());
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
        self.reset();
    }
}

/// Abstraction over wrappers of `ChanceInfo`
trait ChanceRecurse {
    fn next<'a>(&self, chance: &'a Chance) -> &'a Node;
}

impl<T: ChanceInfo> ChanceRecurse for RefCell<T> {
    fn next<'a>(&self, chance: &'a Chance) -> &'a Node {
        self.borrow_mut().next(chance)
    }
}

trait ActiveInfo {
    fn recurse(&mut self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64;

    fn advance(&mut self, discounts: &Discounts) -> f64;
}

trait ActiveRecurse {
    fn recurse(&self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64;
}

impl<T: ActiveInfo> ActiveRecurse for RefCell<T> {
    fn recurse(&self, player: &Player, rec: impl Fn(&Node) -> f64) -> f64 {
        self.borrow_mut().recurse(player, rec)
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
        for cum_reg in &mut self.reg.cum_regret {
            *cum_reg -= expected;
        }
        expected
    }

    fn advance(&mut self, discounts: &Discounts) -> f64 {
        self.cached = 0;
        let bound = discounts.advance_infoset(&mut *self.reg.cum_regret, &mut self.reg.strat);
        discounts.discount_average_strat(&mut self.reg.cum_strat);
        bound
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

/// Advance every infoset of a player, returning the summed regret bound on a check iteration
///
/// On a non-check iteration the advance still runs for its side effects, but the regret bound isn't
/// needed, so we skip summing it and return infinity (which never satisfies an early-termination
/// test).
fn advance_player(
    player: &mut [RefCell<CachedInfoset>],
    discounts: &Discounts,
    check: bool,
) -> f64 {
    if check {
        player
            .iter_mut()
            .map(|info| info.get_mut().advance(discounts))
            .sum()
    } else {
        for info in player.iter_mut() {
            info.get_mut().advance(discounts);
        }
        f64::INFINITY
    }
}

pub(crate) fn solve_external_single(
    start: &Node,
    chance_info: &[impl ChanceInfoset],
    player_info: [&[impl PlayerInfoset]; 2],
    max_iter: u64,
    max_reg: f64,
    params: &RegretParams,
    check_interval: u64,
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
        let check = data::should_check(it, max_iter, check_interval);
        // player one
        recurse_regret::<true>(start, &chance_infosets, &player_one, &player_two, &());
        chance_infosets
            .iter_mut()
            .for_each(|info| info.get_mut().advance());
        // the leading player has nothing accumulated on its first average-strat discount, so it
        // discounts against `it - 1` (see `single_player_iter`)
        let discounts_one = Discounts::new(params, it, it - 1);
        reg_one = advance_player(&mut player_one, &discounts_one, check);
        // player two
        recurse_regret::<false>(start, &chance_infosets, &player_two, &player_one, &());
        chance_infosets
            .iter_mut()
            .for_each(|info| info.get_mut().advance());
        let discounts_two = Discounts::new(params, it, it);
        reg_two = advance_player(&mut player_two, &discounts_two, check);
        // check to terminate
        if check && f64::max(reg_one, reg_two) < max_reg {
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
    use crate::{Chance, ChanceInfoset, Node, Player, PlayerInfoset, PlayerNum, RegretParams};

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
            &self.0
        }
    }

    type Game = (Node, Box<[Cinfo]>, [Box<[Pinfo]>; 2]);

    fn simple_game() -> Game {
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

    fn even_or_odd() -> Game {
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
        let ([reg_one, reg_two], [strat_one, strat_two]) = super::solve_external_single(
            &root,
            &chance,
            [&*one, &*two],
            1000,
            0.0,
            &RegretParams::vanilla(),
            256,
        );
        assert!(strat_one[1] < 0.05);
        assert!(strat_two[0] < 0.05);
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }

    #[test]
    fn test_external_even_or_odd() {
        let (root, chance, [one, two]) = even_or_odd();
        let ([reg_one, reg_two], [strat_one, strat_two]) = super::solve_external_single(
            &root,
            &chance,
            [&*one, &*two],
            10_000,
            0.005,
            &RegretParams::vanilla(),
            256,
        );
        assert!((strat_one[1] - 0.5).abs() < 0.05, "{strat_one:?}");
        assert!((strat_two[0] - 0.5).abs() < 0.05, "{strat_two:?}");
        assert!(reg_one < 0.05);
        assert!(reg_two < 0.05);
    }
}
