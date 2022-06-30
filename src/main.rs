use cfr::{ChanceNode, Game, Node, Player, PlayerNode, Solution, TerminalNode};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{btree_map, BTreeMap, HashMap};
use std::io;
use std::iter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
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

    /// Try pruning strategies played less than this fraction
    ///
    /// After finding a solution subject to max-regret and max-iters criteria, this will try
    /// pruning any strategy played less than this fraction of the time. If pruning all strategies
    /// below this threshold produces less regret then the pruned strategy is output.
    #[clap(short = 'p', long, value_parser, default_value_t = 0.0)]
    prune_threshold: f64,
}

// NOTE we use BTree map to easily gain consistent ordering which is necessary for cfr

type StateNode = Node<Terminal, Chance, Actor>;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum State {
    Terminal(Terminal),
    Chance(Chance),
    Player(Actor),
}

impl Into<StateNode> for State {
    fn into(self) -> StateNode {
        match self {
            State::Terminal(term) => StateNode::Terminal(term),
            State::Chance(chance) => StateNode::Chance(chance),
            State::Player(player) => StateNode::Player(player),
        }
    }
}

#[derive(Deserialize)]
struct Terminal(f64);

impl TerminalNode for Terminal {
    fn get_one_payoff(&self) -> f64 {
        let Terminal(payoff) = self;
        *payoff
    }
}

#[derive(Deserialize)]
struct Chance(BTreeMap<String, Outcome>);

fn to_chance_node(outcome: Outcome) -> (f64, StateNode) {
    let Outcome { prob, state } = outcome;
    (prob, state.into())
}

impl IntoIterator for Chance {
    type Item = (f64, StateNode);
    type IntoIter = iter::Map<btree_map::IntoValues<String, Outcome>, fn(Outcome) -> Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let Chance(tree) = self;
        tree.into_values().map(to_chance_node)
    }
}

impl ChanceNode<Terminal, Actor> for Chance {}

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

impl IntoIterator for Actor {
    type Item = (String, StateNode);
    type IntoIter =
        iter::Map<btree_map::IntoIter<String, State>, fn((String, State)) -> Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.actions.into_iter().map(to_actor_node)
    }
}

impl PlayerNode<String, String, Terminal, Chance> for Actor {
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
                        .map(|(a, p)| (a.clone(), p.clone()))
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
    let Solution { strategy, .. } = game.solve(args.max_iters, args.max_regret);
    let regret = game
        .regret(&strategy)
        .expect("internal error: got invalid strategy, please report at: https://github.com/erikbrinkman/cfr/issues");

    let mut pruned_strat = Vec::with_capacity(strategy.len());
    for strat in strategy.iter() {
        let total: f64 = strat
            .iter()
            .map(|v| if *v < args.prune_threshold { 0.0 } else { *v })
            .sum();
        let pruned: Vec<_> = strat
            .iter()
            .map(|v| {
                if *v < args.prune_threshold {
                    0.0
                } else {
                    *v / total
                }
            })
            .collect();
        pruned_strat.push(pruned);
    }
    let pruned_regret = game
        .regret(&pruned_strat)
        .expect("internal error: pruned strategy incorrectly, please report at: https://github.com/erikbrinkman/cfr/issues");

    let (info, one, two): (_, Strategy, Strategy) = if regret.regret() < pruned_regret.regret() {
        let (one, two) = game.name_strategy(&strategy);
        (regret, one.into(), two.into())
    } else {
        let (one, two) = game.name_strategy(&pruned_strat);
        (pruned_regret, one.into(), two.into())
    };
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
