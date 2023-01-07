use cfr::{Game, GameNode, IntoGameNode, PlayerNum};
use serde::Deserialize;
use serde_json::Error;
use std::collections::{btree_map, BTreeMap};
use std::io::Read;
use std::iter::FusedIterator;
// NOTE we use BTree map to easily gain consistent ordering which is necessary for cfr

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum State {
    Terminal(f64),
    Chance {
        infoset: Option<String>,
        outcomes: BTreeMap<String, Outcome>,
    },
    Player {
        player_one: bool,
        infoset: String,
        actions: BTreeMap<String, State>,
    },
}

#[derive(Deserialize)]
struct Outcome {
    prob: f64,
    state: State,
}

type ActionIter = BTreeMap<String, State>;

struct OutcomeIter(btree_map::IntoIter<String, Outcome>);

impl OutcomeIter {
    fn new(map: BTreeMap<String, Outcome>) -> Self {
        OutcomeIter(map.into_iter())
    }
}

impl Iterator for OutcomeIter {
    type Item = (f64, State);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, out)| (out.prob, out.state))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.len();
        (len, Some(len))
    }
}

impl FusedIterator for OutcomeIter {}

impl ExactSizeIterator for OutcomeIter {}

impl IntoGameNode for State {
    type PlayerInfo = String;
    type Action = String;
    type ChanceInfo = String;
    type Outcomes = OutcomeIter;
    type Actions = ActionIter;

    fn into_game_node(self) -> GameNode<Self> {
        match self {
            State::Terminal(player_one_payoff) => GameNode::Terminal(player_one_payoff),
            State::Chance { infoset, outcomes } => {
                GameNode::Chance(infoset, OutcomeIter::new(outcomes))
            }
            State::Player {
                player_one,
                infoset,
                actions,
            } => GameNode::Player(
                if player_one {
                    PlayerNum::One
                } else {
                    PlayerNum::Two
                },
                infoset,
                actions,
            ),
        }
    }
}

pub fn from_str(raw: &str) -> Result<(Game<String, String>, f64), Error> {
    let definition: State = serde_json::from_str(raw)?;
    Ok(from_state(definition))
}

pub fn from_reader(reader: &mut impl Read) -> (Game<String, String>, f64) {
    let definition = serde_json::from_reader(reader).expect(
        "couldn't parse json game definition : https://github.com/erikbrinkman/cfr#json-error",
    );
    from_state(definition)
}

fn from_state(definition: State) -> (Game<String, String>, f64) {
    (Game::from_root(definition).expect("couldn't extract a compact game representation due to problems with the structure : https://github.com/erikbrinkman/cfr#game-error"), 0.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_success() {
        super::from_reader(&mut r#"{ "terminal": 0.0 }"#.as_bytes());
    }

    #[test]
    #[should_panic(expected = "couldn't parse json game definition")]
    fn test_json_error() {
        super::from_reader(
            &mut r#"{ "chance": { "outcomes": { "a": { "terminal": 0.0 } } } }"#.as_bytes(),
        );
    }

    #[test]
    #[should_panic(
        expected = "couldn't extract a compact game representation due to problems with the structure"
    )]
    fn test_game_error() {
        super::from_reader(
            &mut r#"{ "chance": { "outcomes": { "a": { "prob": 0.0, "state": { "terminal": 0.0 } } } } }"#.as_bytes(),
        );
    }
}
