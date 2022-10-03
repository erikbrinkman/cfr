mod auto;
mod gambit;
mod json;

use cfr::{PlayerNum, SolveMethod};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fs::File;
use std::io;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Method {
    Full,
    Sampled,
    External,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum InputFormat {
    Auto,
    Gambit,
    Json,
}

/// Counterfactual regret minimization solver for two-player zero-sum incomplete-information games
///
/// This program reads a two-player zero-sum perfect-recall extensive-form game in several formats
/// and then uses veriants of counterfactual regret minimization to find approximate nash
/// equilibria. The result will be a json object like:
///
/// `{ "expected_one_utility": <number> , "player_one_regret": <number>, "player_two_regret":
/// <number>, "regret": <number>, "player_one_strategy": <strat>, "player_two_strategy": <strat>
/// }`
///
/// where strats will be mapping from infosets, to actions to probabilities. Zero probability
/// actions will be omitted.
///
/// For more information see: https://github.com/erikbrinkman/cfr
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Try pruning strategies played less than this fraction
    ///
    /// After finding a solution subject to max-regret and max-iters criteria, this will try
    /// pruning any strategy played less than this fraction of the time. If pruning all strategies
    /// below this threshold produces less regret then the pruned strategy is output.
    #[clap(short, long, value_parser, default_value_t = 0.0)]
    clip_threshold: f64,

    /// Terminate solving early if regret is below `max_regret`
    #[clap(short = 'r', long, value_parser, default_value_t = 0.0)]
    max_regret: f64,

    /// Stop after `max_iters`
    #[clap(short = 't', long, value_parser, default_value_t = 1000)]
    max_iters: u64,

    /// Amount of parallelism to use for solving
    ///
    /// If set to zero (default), this will use rusts `std::thread::available_parallelism` falling
    /// back to single threaded if this can't determine an amount.
    #[clap(short, long, value_parser, default_value_t = 0)]
    parallel: usize,

    /// Method to use for game solving
    ///
    /// External : alternates between players, doing regret updates for one while sampling the
    /// other. This samples poor performing branches of the game tree less often, resulting in
    /// faster convergence overall.
    ///
    /// Sampled : only chance nodes are sampled.
    ///
    /// Full : does no sampling, this tends to be very slow on complex games, but produces fully
    /// acurate regret estimates.
    #[clap(short, long, value_enum, default_value_t = Method::External)]
    method: Method,

    /// Format of the input game file
    ///
    /// Auto : If `input` was specified and the filename ends with ".json" or ".efg" use that
    /// parser. Otherwise attempt all parsers. If all of these fail, no diagnostic information will
    /// be given, instead retry with the desired parser.
    ///
    /// Gambit : parses Gambit style `.efg` files. Because this uses string representations of
    /// information sets, if an information set doesn't have a name it's name will become its
    /// number in string format. If this conflicts with a given infoset name, the parsing will fail
    /// even though the file is valid. This also only works for constant sum games, so if the game
    /// is not constant sum, this will also error. Finally, gambit uses rational payoffs, but this
    /// only computes approximate equilibria and so only uses floating point numbers.
    ///
    /// Json : uses a custom json extensive form game format. The game tree is specified as a tree
    /// of json objects. Terminal nodes have the structure `{ terminal: <number> }`. Chance nodes
    /// have the structure `{ chance: { infoset?: <string>, outcomes: { [<string>]: { prob:
    /// <number>, state: <node> } } } }`. Chance probabilities will be renormalized, so they don't
    /// need to explicitely sum to one. Player nodes have the structure `{ player: { player_one:
    /// <bool>, infoset: <string>, actions: { [<string>]: <node> } } }`.
    #[clap(long, value_enum, default_value_t = InputFormat::Auto)]
    input_format: InputFormat,

    /// Read game from a file instead of from stdin
    #[clap(short, long, value_parser, default_value = "-")]
    input: String,

    /// Write results to a file instead of stdout
    #[clap(short, long, value_parser, default_value = "-")]
    output: String,
}

#[derive(Serialize)]
struct Strategy(HashMap<String, HashMap<String, f64>>);

impl<I, A, S, T, N> From<I> for Strategy
where
    I: IntoIterator<Item = (S, A)>,
    A: IntoIterator<Item = (T, N)>,
    S: AsRef<str>,
    T: AsRef<str>,
    N: Borrow<f64>,
{
    fn from(named_strategies: I) -> Self {
        let mut map = HashMap::new();
        for (info, actions) in named_strategies {
            assert!(
                map.insert(
                    info.as_ref().to_owned(),
                    actions
                        .into_iter()
                        .filter(|(_, p)| p.borrow() > &0.0)
                        .map(|(a, p)| (a.as_ref().to_owned(), *p.borrow()))
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
    expected_one_utility: f64,
    player_one_regret: f64,
    player_two_regret: f64,
    regret: f64,
    player_one_strategy: Strategy,
    player_two_strategy: Strategy,
}

fn main() {
    let args = Args::parse();
    let (game, sum) = if args.input == "-" {
        match args.input_format {
            InputFormat::Json => json::from_reader(io::stdin()),
            InputFormat::Gambit => gambit::from_reader(io::stdin()),
            InputFormat::Auto => auto::from_reader(io::stdin()),
        }
    } else {
        match args.input_format {
            InputFormat::Json => json::from_reader(File::open(args.input).unwrap()),
            InputFormat::Auto if args.input.ends_with(".json") => {
                json::from_reader(File::open(args.input).unwrap())
            }
            InputFormat::Gambit => gambit::from_reader(File::open(args.input).unwrap()),
            InputFormat::Auto if args.input.ends_with(".efg") => {
                gambit::from_reader(File::open(args.input).unwrap())
            }
            InputFormat::Auto => auto::from_reader(File::open(args.input).unwrap()),
        }
    };
    let max_iters = if args.max_iters == 0 {
        u64::MAX
    } else {
        args.max_iters
    };
    let method = match args.method {
        Method::Full => SolveMethod::Full,
        Method::Sampled => SolveMethod::Sampled,
        Method::External => SolveMethod::External,
    };
    let (mut strategies, _) = game
        .solve(method, max_iters, args.max_regret, 0.0, args.parallel)
        .unwrap();
    let mut info = strategies.get_info();
    let mut pruned_strats = strategies.clone();
    pruned_strats.truncate(args.clip_threshold);
    let pruned_info = pruned_strats.get_info();
    if pruned_info.regret() < info.regret() {
        strategies = pruned_strats;
        info = pruned_info;
    }
    let [one, two] = strategies.as_named();
    let out = Output {
        expected_one_utility: info.player_utility(PlayerNum::One) + sum,
        player_one_regret: info.player_regret(PlayerNum::One),
        player_two_regret: info.player_regret(PlayerNum::Two),
        regret: info.regret(),
        player_one_strategy: one.into(),
        player_two_strategy: two.into(),
    };
    if args.output == "-" {
        serde_json::to_writer(io::stdout(), &out).unwrap();
    } else {
        serde_json::to_writer(File::create(args.output).unwrap(), &out).unwrap();
    };
}

#[cfg(test)]
mod tests {
    use super::Args;
    use clap::CommandFactory;

    #[test]
    fn test_cli() {
        Args::command().debug_assert()
    }
}
