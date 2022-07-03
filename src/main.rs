// FIXME clippy
use cfr::{ChanceNode, Game, Node, Player, PlayerNode, Solution};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::{btree_map, BTreeMap, HashMap};
use std::io;
use std::iter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Try pruning strategies played less than this fraction
    ///
    /// After finding a solution subject to max-regret and max-iters criteria, this will try
    /// pruning any strategy played less than this fraction of the time. If pruning all strategies
    /// below this threshold produces less regret then the pruned strategy is output.
    #[clap(short = 'p', long, value_parser, default_value_t = 0.0)]
    prune_threshold: f64,

    #[clap(subcommand)]
    method: Option<Method>,
}

#[derive(Debug, Subcommand)]
enum Method {
    /// FIXME vanilla
    Vanilla {
        /// The maximum regret of an acceptable solution
        ///
        /// If a strategy with lower regret than this is found, this will stop early and return the
        /// solution. If you want to always run max-iters then keep this at 0.0.
        #[clap(short = 'r', long, value_parser, default_value_t = 0.0)]
        max_regret: f64,

        /// The maximum number of iterations to run
        ///
        /// If the solver reaches max-iters iterations it will stop and return the current strategy
        /// independent of its regret. Set this to 0 to only terminate once the regret threshold has
        /// been reached (note that this will never find a 0 regret solution so keeping max-regret at 0
        /// will this to run indefinitely).
        #[clap(short = 'i', long, value_parser, default_value_t = 1000)]
        max_iters: u64,
    },
    /// FIXME external
    External {
        /// FIXME
        #[clap(short = 'i', long, value_parser, default_value_t = 100)]
        max_steps: u64,

        /// FIXME
        #[clap(short = 's', long, value_parser, default_value_t = 10_000)]
        step_size: u64,

        /// The maximum regret of an acceptable solution
        ///
        /// If a strategy with lower regret than this is found, this will stop early and return the
        /// solution. If you want to always run max-iters then keep this at 0.0.
        #[clap(short = 'r', long, value_parser, default_value_t = 0.0)]
        max_regret: f64,
    },
}

// NOTE we use BTree map to easily gain consistent ordering which is necessary for cfr

type StateNode = Node<Chance, Actor>;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum State {
    Terminal(Terminal),
    Chance(Chance),
    Player(Actor),
}

impl From<State> for StateNode {
    fn from(state: State) -> Self {
        match state {
            State::Terminal(Terminal(player_one_payoff)) => {
                StateNode::Terminal { player_one_payoff }
            }
            State::Chance(chance) => StateNode::Chance(chance),
            State::Player(player) => StateNode::Player(player),
        }
    }
}

#[derive(Deserialize)]
struct Terminal(f64);

#[derive(Deserialize)]
struct Chance(BTreeMap<String, Outcome>);

fn to_chance_node(outcome: Outcome) -> (f64, StateNode) {
    let Outcome { prob, state } = outcome;
    (prob, state.into())
}

impl ChanceNode for Chance {
    type PlayerNode = Actor;
    type Outcomes = iter::Map<btree_map::IntoValues<String, Outcome>, fn(Outcome) -> (f64, StateNode)>;

    fn into_outcomes(self) -> Self::Outcomes {
        self.0.into_values().map(to_chance_node)
    }
}

#[derive(Deserialize)]
struct Outcome {
    prob: f64,
    state: State,
}

#[derive(Deserialize)]
struct Actor {
    player_one: bool,
    infoset: String,
    actions: BTreeMap<String, State>,
}

fn to_actor_node(entry: (String, State)) -> (String, StateNode) {
    let (key, state) = entry;
    (key, state.into())
}

impl PlayerNode for Actor {
    type Infoset = String;
    type Action = String;
    type ChanceNode = Chance;
    type Actions =
        iter::Map<btree_map::IntoIter<String, State>, fn((String, State)) -> (String, StateNode)>;

    fn get_player(&self) -> Player {
        if self.player_one {
            Player::One
        } else {
            Player::Two
        }
    }

    fn get_infoset(&self) -> String {
        self.infoset.clone()
    }

    fn into_actions(self) -> Self::Actions {
        self.actions.into_iter().map(to_actor_node)
    }
}

#[derive(Serialize)]
struct Strategy(HashMap<String, HashMap<String, f64>>);

impl<'a, I, A> From<I> for Strategy
where
    I: IntoIterator<Item = (&'a String, A)>,
    A: IntoIterator<Item = (&'a String, &'a f64)>,
{
    fn from(named_strategies: I) -> Self {
        let mut map = HashMap::new();
        for (info, actions) in named_strategies {
            assert!(
                map.insert(
                    info.clone(),
                    actions
                        .into_iter()
                        .filter(|(_, p)| p > &&0.0)
                        .map(|(a, p)| (a.clone(), *p))
                        .collect()
                )
                .is_none(),
                "internal error: found duplicate infosets"
            );
        }
        Strategy(map)
    }
}

#[derive(Serialize)]
struct Output {
    expected_utility: f64,
    player_one_regret: f64,
    player_two_regret: f64,
    regret: f64,
    player_one_strategy: Strategy,
    player_two_strategy: Strategy,
}

fn main() {
    let args = Args::parse();
    let definition: State = serde_json::from_reader(io::stdin())
        .expect("couln't parse game definition, try looking at the readme: https://github.com/erikbrinkman/cfr#json-error");
    let game = Game::from_node(definition.into()).expect("couldn't extract a compact game representation due to problems with the structure, try looking at the readme: https://github.com/erikbrinkman/cfr#json-error");
    let Solution { strategy, .. } = match args.method {
        // FIXME change solve_full to solve_vanilla
        Some(Method::Vanilla {
            max_regret,
            max_iters,
        }) => game.solve_full(
            if max_iters == 0 { u64::MAX } else { max_iters },
            max_regret,
        ),
        Some(Method::External {
            max_regret,
            max_steps,
            step_size,
        }) => game.solve_external(
            if max_steps == 0 { u64::MAX } else { max_steps },
            step_size,
            max_regret,
        ),
        None => game.solve_external(100, 10_000, 0.0),
    };

    let regret = game
        .regret(&strategy)
        .expect("internal error: got invalid strategy, please report at: https://github.com/erikbrinkman/cfr/issues");

    let mut pruned_strat = strategy.clone();
    pruned_strat.truncate(args.prune_threshold);
    let pruned_regret = game
        .regret(&pruned_strat)
        .expect("internal error: pruned strategy incorrectly, please report at: https://github.com/erikbrinkman/cfr/issues");

    let (strat, info) = if regret.regret() < pruned_regret.regret() {
        (strategy, regret)
    } else {
        (pruned_strat, pruned_regret)
    };
    let (one, two) = game.name_strategy(&strat).expect("internal error: couldn't attach strategy names, please report at: https://github.com/erikbrinkman/cfr/issues");
    let out = Output {
        expected_utility: info.utility,
        player_one_regret: info.player_one_regret,
        player_two_regret: info.player_two_regret,
        regret: info.regret(),
        player_one_strategy: one.into(),
        player_two_strategy: two.into(),
    };

    serde_json::to_writer(io::stdout(), &out).expect("error writing strategies to stdout");
}
