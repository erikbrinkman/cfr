use cfr::{Game, Moves, NodeType, Outcomes, PlayerNum};
use indexmap::IndexMap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum State {
    Terminal(f64),
    Chance {
        infoset: Option<String>,
        outcomes: IndexMap<String, Outcome>,
    },
    Player {
        player_one: bool,
        infoset: String,
        actions: IndexMap<String, State>,
    },
}

#[derive(Debug, Deserialize)]
pub(crate) struct Outcome {
    prob: f64,
    state: State,
}

// A `&State` is the game directly -- it's a cheap handle into the parsed tree, so no wrapper type is
// needed; child states are just borrows.
impl<'a> Game for &'a State {
    type Action = &'a str;
    type Infoset = &'a str;
    type ChanceInfoset = &'a str;
    type Chance = &'a IndexMap<String, Outcome>;
    type Player = &'a IndexMap<String, State>;

    fn into_node(self) -> NodeType<Self> {
        match self {
            State::Terminal(payoff) => NodeType::Terminal(*payoff),
            // a named infoset correlates sampling across chance nodes; `None` is a unique node
            State::Chance { infoset, outcomes } => NodeType::Chance(infoset.as_deref(), outcomes),
            State::Player {
                player_one,
                infoset,
                actions,
            } => NodeType::Player(
                if *player_one {
                    PlayerNum::One
                } else {
                    PlayerNum::Two
                },
                infoset.as_str(),
                actions,
            ),
        }
    }
}

impl<'a> Outcomes<&'a State> for &'a IndexMap<String, Outcome> {
    fn len(&self) -> usize {
        IndexMap::len(self)
    }

    fn get(&self, index: usize) -> (f64, &'a State) {
        let (_, outcome) = self.get_index(index).unwrap();
        (outcome.prob, &outcome.state)
    }

    fn iter(&self) -> impl Iterator<Item = (f64, &'a State)> + '_ {
        self.values().map(|outcome| (outcome.prob, &outcome.state))
    }
}

impl<'a> Moves<&'a State> for &'a IndexMap<String, State> {
    fn len(&self) -> usize {
        IndexMap::len(self)
    }

    fn action(&self, index: usize) -> &'a str {
        self.get_index(index).unwrap().0.as_str()
    }

    fn apply(&self, index: usize) -> &'a State {
        self.get_index(index).unwrap().1
    }

    fn iter(&self) -> impl Iterator<Item = (&'a str, &'a State)> + '_ {
        self.keys()
            .zip(self.values())
            .map(|(action, state)| (action.as_str(), state))
    }
}

#[cfg(test)]
mod tests {
    use cfr::{GameTree, LazySolver, PlayerNum, SolveMethod, SolveParams};

    #[test]
    fn lazy_solver_walks_the_game() {
        // the tree-free solver drives `&State` on demand, exercising the Moves/Outcomes
        // len/get/apply that GameTree materialization (which goes through `iter`) never calls
        let state: super::State = serde_json::from_str(
            r#"{ "chance": { "outcomes": {
                "deal": { "prob": 1.0, "state": { "player": { "player_one": true, "infoset": "p1",
                    "actions": { "h": { "terminal": 1.0 }, "t": { "terminal": -1.0 } } } } } } } }"#,
        )
        .unwrap();
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&&state, 500);
        // player one always prefers heads (payoff 1 vs -1)
        let strat = solver.average(PlayerNum::One, &"p1").unwrap();
        assert!(strat[0] > 0.9, "should favor heads: {strat:?}");
    }

    #[test]
    fn parses_terminal() {
        let state: super::State = serde_json::from_str(r#"{ "terminal": 0.0 }"#).unwrap();
        GameTree::from_game(&state).unwrap();
    }

    #[test]
    fn solves_matching_pennies() {
        // player one picks a side, player two responds from a single shared infoset (so it can't see
        // the choice); matching pays player one. The equilibrium is 50/50 for both, value 0. This
        // checks the rewritten Game pipeline builds a correct game, not just that errors error.
        let state: super::State = serde_json::from_str(
            r#"{ "player": { "player_one": true, "infoset": "p1", "actions": {
                "heads": { "player": { "player_one": false, "infoset": "p2", "actions": {
                    "heads": { "terminal": 1.0 }, "tails": { "terminal": -1.0 } } } },
                "tails": { "player": { "player_one": false, "infoset": "p2", "actions": {
                    "heads": { "terminal": -1.0 }, "tails": { "terminal": 1.0 } } } } } } }"#,
        )
        .unwrap();
        let game = GameTree::from_game(&state).unwrap();
        let (strats, bound) = game
            .solve(SolveMethod::Full, 50_000, 0.0, 1, SolveParams::default())
            .unwrap();
        for player in [PlayerNum::One, PlayerNum::Two] {
            assert!(
                bound.player_regret_bound(player) < 0.02,
                "player {player:?} not converged"
            );
        }
        for named in strats.as_named() {
            for (_infoset, actions) in named {
                for (_action, prob) in actions {
                    assert!((prob - 0.5).abs() < 0.05, "not ~50/50: {prob}");
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_json() {
        let parsed: Result<super::State, _> =
            serde_json::from_str(r#"{ "chance": { "outcomes": { "a": { "terminal": 0.0 } } } }"#);
        assert!(parsed.is_err());
    }

    #[test]
    fn rejects_invalid_game() {
        let state: super::State = serde_json::from_str(
            r#"{ "chance": { "outcomes": { "a": { "prob": 0.0, "state": { "terminal": 0.0 } } } } }"#,
        )
        .unwrap();
        assert!(GameTree::from_game(&state).is_err());
    }
}
