//! Tests based on kuhn poker
use cfr::{Game, GameNode, IntoGameNode, PlayerNum, RegretParams, SolveMethod, Strategies};
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
#[cfg(not(tarpaulin))]
fn test_equilibrium() {
    let game = create_kuhn(3);

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
#[cfg(not(tarpaulin))]
fn test_regret() {
    let game = create_kuhn(3);

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
#[cfg(not(tarpaulin))]
fn test_solve_three() {
    let owned_game = create_kuhn(3);
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
#[cfg(not(tarpaulin))]
fn test_solve_three_discounts() {
    let owned_game = create_kuhn(3);
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
