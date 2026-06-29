//! An example implementation of the [`cfr::Game`] trait for Kuhn Poker
use cfr::{
    Game, GameTree, Moves, NodeType, Outcomes, PlayerNum, SolveMethod, SolveParams,
};
use clap::Parser;
use std::convert::Infallible;

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
enum Action {
    Fold,
    Call,  // check or call
    Raise, // bet or raise
}

/// A node in Kuhn poker. Chance deals player one's card, then -- after player one's first action --
/// player two's card from the remaining cards; a player's infoset is `(their card, raise pending)`.
/// Each variant carries exactly its node's data, so the carriers never inspect a phase.
#[derive(Debug, Clone, Copy)]
enum Kuhn {
    DealOne(usize),              // num_cards
    OneFirst(usize, usize),      // num_cards, one_card
    DealTwo(usize, usize, bool), // num_cards, one_card, one_raised
    TwoAct(usize, usize, bool),  // one_card, two_card, one_raised
    OneSecond(usize, usize),     // one_card, two_card
    Done(f64),
}

/// The terminal where the higher card wins `stake` for player one.
fn showdown(one_card: usize, two_card: usize, stake: f64) -> Kuhn {
    Kuhn::Done(if one_card > two_card { stake } else { -stake })
}

/// A deal chance node's outcomes (the carrier for [`Game::Chance`]).
#[derive(Debug)]
enum Deal {
    One(usize),              // num_cards
    Two(usize, usize, bool), // num_cards, one_card, one_raised
}

/// A player decision's moves (the carrier for [`Game::Player`]).
#[derive(Debug)]
enum Decide {
    OneFirst(usize, usize),     // num_cards, one_card
    TwoAct(usize, usize, bool), // one_card, two_card, one_raised
    OneSecond(usize, usize),    // one_card, two_card
}

impl Game for Kuhn {
    type Action = Action;
    type Infoset = (usize, bool);
    type ChanceInfoset = Infallible; // deals are never correlated, so chance nodes are unique
    type Chance = Deal;
    type Player = Decide;

    fn into_node(self) -> NodeType<Self> {
        match self {
            Kuhn::Done(payoff) => NodeType::Terminal(payoff),
            Kuhn::DealOne(num_cards) => NodeType::Chance(None, Deal::One(num_cards)),
            Kuhn::DealTwo(num_cards, one_card, one_raised) => {
                NodeType::Chance(None, Deal::Two(num_cards, one_card, one_raised))
            }
            Kuhn::OneFirst(num_cards, one_card) => {
                NodeType::Player(PlayerNum::One, (one_card, false), Decide::OneFirst(num_cards, one_card))
            }
            Kuhn::TwoAct(one_card, two_card, one_raised) => NodeType::Player(
                PlayerNum::Two,
                (two_card, one_raised),
                Decide::TwoAct(one_card, two_card, one_raised),
            ),
            Kuhn::OneSecond(one_card, two_card) => {
                NodeType::Player(PlayerNum::One, (one_card, true), Decide::OneSecond(one_card, two_card))
            }
        }
    }
}

impl Outcomes<Kuhn> for Deal {
    fn len(&self) -> usize {
        match *self {
            Deal::One(num_cards) => num_cards,
            Deal::Two(num_cards, ..) => num_cards - 1, // player one's card is excluded
        }
    }

    fn get(&self, index: usize) -> (f64, Kuhn) {
        match *self {
            Deal::One(num_cards) => (1.0 / num_cards as f64, Kuhn::OneFirst(num_cards, index)),
            Deal::Two(num_cards, one_card, one_raised) => {
                let two_card = if index < one_card { index } else { index + 1 };
                (
                    1.0 / (num_cards - 1) as f64,
                    Kuhn::TwoAct(one_card, two_card, one_raised),
                )
            }
        }
    }
}

impl Moves<Kuhn> for Decide {
    fn len(&self) -> usize {
        2
    }

    fn action(&self, index: usize) -> Action {
        match self {
            Decide::OneFirst(..) | Decide::TwoAct(_, _, false) => [Action::Call, Action::Raise][index],
            Decide::TwoAct(_, _, true) | Decide::OneSecond(..) => [Action::Call, Action::Fold][index],
        }
    }

    fn apply(&self, index: usize) -> Kuhn {
        let action = self.action(index);
        match *self {
            Decide::OneFirst(num_cards, one_card) => {
                Kuhn::DealTwo(num_cards, one_card, action == Action::Raise)
            }
            Decide::TwoAct(one_card, two_card, one_raised) => {
                if one_raised {
                    // facing the raise: call to a showdown at 2, or fold and concede 1
                    if action == Action::Call {
                        showdown(one_card, two_card, 2.0)
                    } else {
                        Kuhn::Done(1.0)
                    }
                } else if action == Action::Call {
                    showdown(one_card, two_card, 1.0) // checked down
                } else {
                    Kuhn::OneSecond(one_card, two_card) // raised back
                }
            }
            Decide::OneSecond(one_card, two_card) => {
                // call the re-raise (showdown at 2) or fold (lose 1)
                if action == Action::Call {
                    showdown(one_card, two_card, 2.0)
                } else {
                    Kuhn::Done(-1.0)
                }
            }
        }
    }
}

fn create_kuhn(num_cards: usize) -> GameTree<(usize, bool), Action> {
    assert!(num_cards > 1);
    GameTree::from_game(Kuhn::DealOne(num_cards)).unwrap()
}

/// Use cfr to find a kuhn poker strategy
#[derive(Debug, Parser)]
struct Args {
    /// The number of cards used for Kuhn Poker
    #[arg(default_value_t = 3)]
    num_cards: usize,

    /// The number of iterations to run
    #[arg(short, long, default_value_t = 10000)]
    iterations: u64,

    /// The number of threads to use for solving
    #[arg(short, long, default_value_t = 0)]
    parallel: usize,
}

const CARDS: [&str; 13] = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A",
];

fn print_card(card: usize, num_cards: usize) {
    if num_cards > CARDS.len() {
        print!("{card}");
    } else {
        print!("{}", CARDS[CARDS.len() - num_cards + card]);
    }
}

fn print_card_strat<'a>(
    mut card_strat: Vec<(usize, impl IntoIterator<Item = (&'a Action, f64)>)>,
    num_cards: usize,
) {
    card_strat.sort_by_key(|&(card, _)| card);
    for (card, actions) in card_strat {
        print!("holding ");
        print_card(card, num_cards);
        print!(":");
        for (action, prob) in actions {
            if prob == 1.0 {
                print!(" always {action:?}");
            } else if prob > 0.0 {
                print!(" {prob:.2} {action:?}");
            }
        }
        println!();
    }
    println!();
}

fn main() {
    let args = Args::parse();
    let game = create_kuhn(args.num_cards);
    let (mut strats, _) = game
        .solve(
            SolveMethod::External,
            args.iterations,
            0.0,
            args.parallel,
            SolveParams::default(),
        )
        .unwrap();
    strats.truncate(5e-3); // not visible
    let [player_one_strat, player_two_strat] = strats.as_named();
    println!("Player One");
    println!("==========");
    let mut init = Vec::with_capacity(3);
    let mut raised = Vec::with_capacity(3);
    for (&(card, raise), actions) in player_one_strat {
        if raise { &mut raised } else { &mut init }.push((card, actions));
    }
    println!("Initial Action");
    println!("--------------");
    print_card_strat(init, args.num_cards);
    println!("If Player Two Raised");
    println!("--------------------");
    print_card_strat(raised, args.num_cards);
    println!("Player Two");
    println!("==========");
    let mut called = Vec::with_capacity(3);
    let mut raised = Vec::with_capacity(3);
    for (&(card, raise), actions) in player_two_strat {
        if raise { &mut raised } else { &mut called }.push((card, actions));
    }
    println!("If Player One Called");
    println!("--------------------");
    print_card_strat(called, args.num_cards);
    println!("If Player One Raised");
    println!("--------------------");
    print_card_strat(raised, args.num_cards);
}

#[cfg(test)]
mod tests {
    use super::Action;
    use cfr::{PlayerNum, RegretParams, SolveMethod, SolveParams, Strategies};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    fn infer_alpha(strat: &Strategies<(usize, bool), Action>) -> f64 {
        let mut alpha = 0.0;
        let [one, _] = strat.as_named();
        for (info, actions) in one {
            match info {
                (0, false) => {
                    let (_, prob) = actions
                        .into_iter()
                        .find(|(act, _)| act == &&Action::Raise)
                        .unwrap();
                    alpha += prob;
                }
                (1, true) => {
                    let (_, prob) = actions
                        .into_iter()
                        .find(|(act, _)| act == &&Action::Call)
                        .unwrap();
                    alpha += prob - 1.0 / 3.0;
                }
                (2, false) => {
                    let (_, prob) = actions
                        .into_iter()
                        .find(|(act, _)| act == &&Action::Raise)
                        .unwrap();
                    alpha += prob / 3.0;
                }
                _ => (),
            }
        }
        alpha
    }

    fn create_equilibrium(alpha: f64) -> [Vec<((usize, bool), Vec<(Action, f64)>)>; 2] {
        assert!(
            (-1e-2..=1.01).contains(&alpha),
            "alpha not in proper range: {}",
            alpha
        );
        let alpha = f64::min(f64::max(0.0, alpha), 1.0);
        let one = vec![
            (
                (0, false),
                vec![(Action::Call, 3.0 - alpha), (Action::Raise, alpha)],
            ),
            ((1, false), vec![(Action::Call, 1.0)]),
            (
                (2, false),
                vec![(Action::Call, 1.0 - alpha), (Action::Raise, alpha)],
            ),
            ((0, true), vec![(Action::Fold, 1.0)]),
            (
                (1, true),
                vec![(Action::Fold, 2.0 - alpha), (Action::Call, 1.0 + alpha)],
            ),
            ((2, true), vec![(Action::Call, 1.0)]),
        ];
        let two = vec![
            ((0, false), vec![(Action::Call, 2.0), (Action::Raise, 1.0)]),
            ((1, false), vec![(Action::Call, 1.0)]),
            ((2, false), vec![(Action::Raise, 1.0)]),
            ((0, true), vec![(Action::Fold, 1.0)]),
            ((1, true), vec![(Action::Call, 1.0), (Action::Fold, 2.0)]),
            ((2, true), vec![(Action::Call, 1.0)]),
        ];
        [one, two]
    }

    #[test]
    fn test_equilibrium() {
        let game = super::create_kuhn(3);

        let eqm = game.from_named(create_equilibrium(0.5)).unwrap();
        let info = eqm.get_info();

        let util = info.player_utility(PlayerNum::One);
        assert!(
            (util + 1.0 / 18.0).abs() < 1e-3,
            "utility not close to -1/18: {}",
            util
        );

        let eqm_reg = info.regret();
        assert!(eqm_reg < 0.01, "equilibrium regret too large: {}", eqm_reg);

        let eqm = game.from_named(create_equilibrium(1.0)).unwrap();
        let info = eqm.get_info();

        let util = info.player_utility(PlayerNum::One);
        assert!(
            (util + 1.0 / 18.0).abs() < 1e-3,
            "utility not close to -1/18: {}",
            util
        );

        let eqm_reg = info.regret();
        assert!(eqm_reg < 0.01, "equilibrium regret too large: {}", eqm_reg);
    }

    #[test]
    fn test_regret() {
        let game = super::create_kuhn(3);

        let [one, _] = create_equilibrium(0.5);
        let bad = vec![
            ((0, false), vec![(Action::Raise, 1.0)]),
            ((1, false), vec![(Action::Raise, 1.0)]),
            ((2, false), vec![(Action::Call, 1.0)]),
            ((0, true), vec![(Action::Call, 1.0)]),
            ((1, true), vec![(Action::Call, 1.0)]),
            ((2, true), vec![(Action::Fold, 1.0)]),
        ];
        let eqm = game.from_named([one, bad]).unwrap();
        let info = eqm.get_info();

        let util = info.player_utility(PlayerNum::One);
        assert!(util > 0.0, "utility not positive: {}", util);
    }

    #[test]
    fn test_solve_three() {
        let owned_game = super::create_kuhn(3);
        let game = &owned_game;
        // test all methods with multi threading
        [
            SolveMethod::Full,
            SolveMethod::Sampled,
            SolveMethod::External,
        ]
        .into_par_iter()
        .flat_map(|method| [1, 2].into_par_iter().map(move |threads| (method, threads)))
        .for_each(|(method, threads)| {
            let (mut strategies, bounds) = game
                .solve(
                    method,
                    10_000_000,
                    0.005,
                    threads,
                    SolveParams {
                        regret: RegretParams::vanilla(),
                        ..Default::default()
                    },
                )
                .unwrap();
            strategies.truncate(1e-3);

            let alpha = infer_alpha(&strategies);
            let eqm = game.from_named(create_equilibrium(alpha)).unwrap();
            let [dist_one, dist_two] = strategies.distance(&eqm, 1.0);
            assert!(
                dist_one < 0.05,
                "first player strategy not close enough to alpha equilibrium: {} [{:?} {}]",
                dist_one,
                method,
                threads
            );
            assert!(
                dist_two < 0.05,
                "second player strategy not close enough to alpha equilibrium: {} [{:?} {}]",
                dist_two,
                method,
                threads
            );

            let info = strategies.get_info();
            let util = info.player_utility(PlayerNum::One);
            assert!(
                (util + 1.0 / 18.0).abs() < 1e-3,
                "utility not close to -1/18: {} [{:?} {}]",
                util,
                method,
                threads
            );

            let bound = bounds.regret_bound();
            assert!(
                bound < 0.005,
                "regret bound not small enough: {} [{:?} {}]",
                bound,
                method,
                threads
            );

            let regret = info.regret();

            // NOTE with the sampled versions, the bound can be a bit higher
            let eff_bound = bound * 2.0;
            assert!(
                regret <= eff_bound,
                "regret not less than effective bound: {} > {} [{:?} {}]",
                regret,
                eff_bound,
                method,
                threads,
            );
        });
    }

    // NOTE discounts don't have accurate regret thresholds
    #[test]
    fn test_solve_three_discounts() {
        let owned_game = super::create_kuhn(3);
        let game = &owned_game;
        [
            ("vanilla", RegretParams::vanilla()),
            ("lcfr", RegretParams::lcfr()),
            ("cfr+", RegretParams::cfr_plus()),
            ("dcfr", RegretParams::dcfr()),
        ]
        .into_par_iter()
        .for_each(|(name, params)| {
            let (mut strategies, _) = game
                .solve(
                    SolveMethod::Full,
                    10_000,
                    0.0,
                    1,
                    SolveParams {
                        regret: params,
                        ..Default::default()
                    },
                )
                .unwrap();
            strategies.truncate(1e-3);
            let info = strategies.get_info();
            let util = info.player_utility(PlayerNum::One);
            assert!(
                (util + 1.0 / 18.0).abs() < 1e-3,
                "utility not close to -1/18: {} [{}]",
                util,
                name
            );
        });
    }
}
