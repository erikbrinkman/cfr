use cfr::{Game, GameNode, IntoGameNode, PlayerNum, Strategies};
use rand::{thread_rng, Rng};

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

fn create_equilibrium(
    game: &Game<(usize, bool), Action>,
    alpha: f64,
) -> Strategies<(usize, bool), Action> {
    assert!(
        (0.0..=1.0).contains(&alpha),
        "alpha not in proper range: {}",
        alpha
    );
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
    game.from_named([one, two]).unwrap()
}

#[test]
#[cfg(not(tarpaulin))]
fn equilibrium_test() {
    let game = create_kuhn(3);

    let eqm = create_equilibrium(&game, 0.5);
    let info = eqm.get_info();

    let util = info.player_utility(PlayerNum::One);
    assert!(
        (util + 1.0 / 18.0).abs() < 1e-3,
        "utility not close to -1/18: {}",
        util
    );

    let eqm_reg = info.regret();
    assert!(eqm_reg < 0.01, "equilibrium regret too large: {}", eqm_reg);

    let eqm = create_equilibrium(&game, 1.0);
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
fn solve_full_three() {
    let game = create_kuhn(3);
    let (mut strategies, [one_bound, two_bound]) = game.solve_full(10000, 0.005);
    strategies.truncate(1e-3);

    let alpha = infer_alpha(&strategies);
    let eqm = create_equilibrium(&game, alpha);
    let [dist_one, dist_two] = strategies.distance(&eqm);
    assert!(
        dist_one < 0.01,
        "first player strategy not close enough to alpha equilibrium: {}",
        dist_one
    );
    assert!(
        dist_two < 0.01,
        "second player strategy not close enough to alpha equilibrium: {}",
        dist_two
    );

    let info = strategies.get_info();
    let util = info.player_utility(PlayerNum::One);
    assert!(
        (util + 1.0 / 18.0).abs() < 1e-3,
        "utility not close to -1/18: {}",
        util
    );

    let bound = f64::max(one_bound, two_bound);
    assert!(bound < 0.005, "regret bound not small enough: {}", bound);

    let regret = info.regret();
    assert!(regret <= bound, "regret not less than bound: {}", regret);
}

#[test]
#[cfg(not(tarpaulin))]
fn solve_sampled_three() {
    thread_rng().fill(&mut [0; 8]);
    let game = create_kuhn(3);
    let (mut strategies, [one_bound, two_bound]) = game.solve_sampled(100000, 0.005);
    strategies.truncate(1e-3);

    let alpha = infer_alpha(&strategies);
    let eqm = create_equilibrium(&game, alpha);
    let [dist_one, dist_two] = strategies.distance(&eqm);
    assert!(
        dist_one < 0.02,
        "first player strategy not close enough to alpha equilibrium: {}",
        dist_one
    );
    assert!(
        dist_two < 0.02,
        "second player strategy not close enough to alpha equilibrium: {}",
        dist_two
    );

    let info = strategies.get_info();
    let util = info.player_utility(PlayerNum::One);
    assert!(
        (util + 1.0 / 18.0).abs() < 1e-3,
        "utility not close to -1/18: {}",
        util
    );

    let bound = f64::max(one_bound, two_bound);
    assert!(bound < 0.005, "regret bound not small enough: {}", bound);

    let regret = info.regret();
    assert!(
        regret <= bound * 2.0,
        "regret not less than bound: {} > {}",
        regret,
        bound * 2.0
    );
}

#[test]
#[cfg(not(tarpaulin))]
fn solve_external_three() {
    thread_rng().fill(&mut [0; 8]);
    let game = create_kuhn(3);
    let (mut strategies, [one_bound, two_bound]) = game.solve_external(100000, 0.005);
    strategies.truncate(1e-3);

    let alpha = infer_alpha(&strategies);
    let eqm = create_equilibrium(&game, alpha);
    let [dist_one, dist_two] = strategies.distance(&eqm);
    assert!(
        dist_one < 0.02,
        "first player strategy not close enough to alpha equilibrium: {}",
        dist_one
    );
    assert!(
        dist_two < 0.02,
        "second player strategy not close enough to alpha equilibrium: {}",
        dist_two
    );

    let info = strategies.get_info();
    let util = info.player_utility(PlayerNum::One);
    assert!(
        (util + 1.0 / 18.0).abs() < 1e-3,
        "utility not close to -1/18: {}",
        util
    );

    let bound = f64::max(one_bound, two_bound);
    assert!(bound < 0.005, "regret bound not small enough: {}", bound);

    let regret = info.regret();
    assert!(
        regret <= bound * 2.0,
        "regret not less than bound: {} > {}",
        regret,
        bound * 2.0
    );
}
