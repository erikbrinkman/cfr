//! An example implementation of IntoGameNode for Kuhn Poker
use cfr::{Game, GameNode, IntoGameNode, PlayerNum, SolveMethod};
use clap::Parser;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Impossible {}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
enum Action {
    Fold,
    Call,
    Raise,
}

enum Kuhn {
    Terminal(f64),
    Deal(Vec<(f64, Kuhn)>),
    Gambler(PlayerNum, (usize, bool), Vec<(Action, Kuhn)>),
}

impl IntoGameNode for Kuhn {
    type PlayerInfo = (usize, bool);
    type Action = Action;
    type ChanceInfo = Impossible;
    type Outcomes = Vec<(f64, Kuhn)>;
    type Actions = Vec<(Action, Kuhn)>;

    fn into_game_node(self) -> GameNode<Self> {
        match self {
            Kuhn::Terminal(payoff) => GameNode::Terminal(payoff),
            Kuhn::Deal(outcomes) => GameNode::Chance(None, outcomes),
            Kuhn::Gambler(num, info, actions) => GameNode::Player(num, info, actions),
        }
    }
}

fn create_kuhn_one(num_cards: usize) -> Kuhn {
    assert!(num_cards > 1);
    let frac = 1.0 / num_cards as f64;
    Kuhn::Deal(
        (0..num_cards)
            .map(|card| {
                let next: Vec<_> = [
                    (Action::Call, create_kuhn_two(num_cards, card, false)),
                    (Action::Raise, create_kuhn_two(num_cards, card, true)),
                ]
                .into();
                (frac, Kuhn::Gambler(PlayerNum::One, (card, false), next))
            })
            .collect(),
    )
}

fn create_kuhn_two(num_cards: usize, first: usize, one_raised: bool) -> Kuhn {
    let frac = 1.0 / (num_cards - 1) as f64;
    let lose_cards = 0..first;
    let win_cards = first + 1..num_cards;
    Kuhn::Deal(if one_raised {
        let lose_acts = lose_cards.map(|card| {
            let next = [
                (Action::Call, Kuhn::Terminal(2.0)),
                (Action::Fold, Kuhn::Terminal(1.0)),
            ];
            (
                frac,
                Kuhn::Gambler(PlayerNum::Two, (card, true), next.into()),
            )
        });
        let win_acts = win_cards.map(|card| {
            let next = [
                (Action::Call, Kuhn::Terminal(-2.0)),
                (Action::Fold, Kuhn::Terminal(1.0)),
            ];
            (
                frac,
                Kuhn::Gambler(PlayerNum::Two, (card, true), next.into()),
            )
        });
        lose_acts.chain(win_acts).collect()
    } else {
        let lose_acts = lose_cards.map(|card| {
            let raise = [
                (Action::Call, Kuhn::Terminal(2.0)),
                (Action::Fold, Kuhn::Terminal(-1.0)),
            ];
            let next = [
                (Action::Call, Kuhn::Terminal(1.0)),
                (
                    Action::Raise,
                    Kuhn::Gambler(PlayerNum::One, (first, true), raise.into()),
                ),
            ];
            (
                frac,
                Kuhn::Gambler(PlayerNum::Two, (card, false), next.into()),
            )
        });
        let win_acts = win_cards.map(|card| {
            let raise = [
                (Action::Call, Kuhn::Terminal(-2.0)),
                (Action::Fold, Kuhn::Terminal(-1.0)),
            ];
            let next = [
                (Action::Call, Kuhn::Terminal(-1.0)),
                (
                    Action::Raise,
                    Kuhn::Gambler(PlayerNum::One, (first, true), raise.into()),
                ),
            ];
            (
                frac,
                Kuhn::Gambler(PlayerNum::Two, (card, false), next.into()),
            )
        });
        lose_acts.chain(win_acts).collect()
    })
}

fn create_kuhn(num_cards: usize) -> Game<(usize, bool), Action> {
    Game::from_root(create_kuhn_one(num_cards)).unwrap()
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
            None,
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
    use cfr::{PlayerNum, RegretParams, SolveMethod, Strategies};
    use rand::{thread_rng, Rng};
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
            thread_rng().fill(&mut [0; 8]);
            let (mut strategies, bounds) = game
                .solve(
                    method,
                    10_000_000,
                    0.005,
                    threads,
                    Some(RegretParams::vanilla()),
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
            thread_rng().fill(&mut [0; 8]);
            let (mut strategies, _) = game
                .solve(SolveMethod::Full, 10_000, 0.0, 1, Some(params))
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
