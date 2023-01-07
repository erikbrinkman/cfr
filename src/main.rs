mod auto;
mod gambit;
mod json;

use cfr::{PlayerNum, RegretParams, SolveMethod};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufReader;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Method {
    /// No sampling
    Full,
    /// Sample chance nodes
    Sampled,
    /// Sample chance and the other player
    External,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum InputFormat {
    /// Auto-detect format from file extension and contents
    Auto,
    /// Gambit style `.efg` format
    Gambit,
    /// Json game dsl: https://github.com/erikbrinkman/cfr#json-format
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Discount {
    /// No discounting
    Vanilla,
    /// Linear discounting of regret and strategies
    Lcfr,
    /// Forget negative regrets and quadratic strategy discounting (CFR+)
    CfrPlus,
    /// Rough average of LCFR and CFR+
    Dcfr,
    /// DCFR modified to prune poor actions
    DcfrPrune,
}

impl Discount {
    fn into_params(self) -> RegretParams {
        match self {
            Discount::Vanilla => RegretParams::vanilla(),
            Discount::Lcfr => RegretParams::lcfr(),
            Discount::CfrPlus => RegretParams::cfr_plus(),
            Discount::Dcfr => RegretParams::dcfr(),
            Discount::DcfrPrune => RegretParams::dcfr_prune(),
        }
    }
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
    #[clap(short, long, value_enum, default_value_t = Method::External)]
    method: Method,

    /// Format of the input game file
    #[clap(long, value_enum, default_value_t = InputFormat::Auto)]
    input_format: InputFormat,

    /// Discounted CFR parameters
    #[clap(short, long, value_enum, default_value_t = Discount::Dcfr)]
    discount: Discount,

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
    regret: f64,
    player_one_utility: f64,
    player_two_utility: f64,
    player_one_regret: f64,
    player_two_regret: f64,
    player_one_strategy: Strategy,
    player_two_strategy: Strategy,
}

fn main() {
    let args = Args::parse();
    let (game, sum) = if args.input == "-" {
        let mut inp = io::stdin().lock();
        match args.input_format {
            InputFormat::Json => json::from_reader(&mut inp),
            InputFormat::Gambit => gambit::from_reader(&mut inp),
            InputFormat::Auto => auto::from_reader(&mut inp),
        }
    } else {
        let mut inp = BufReader::new(File::open(&args.input).unwrap());
        match args.input_format {
            InputFormat::Json => json::from_reader(&mut inp),
            InputFormat::Auto if args.input.ends_with(".json") => json::from_reader(&mut inp),
            InputFormat::Gambit => gambit::from_reader(&mut inp),
            InputFormat::Auto if args.input.ends_with(".efg") => gambit::from_reader(&mut inp),
            InputFormat::Auto => auto::from_reader(&mut inp),
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
        .solve(
            method,
            max_iters,
            args.max_regret,
            args.parallel,
            Some(args.discount.into_params()),
        )
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
        regret: info.regret(),
        player_one_utility: info.player_utility(PlayerNum::One) + sum,
        player_two_utility: info.player_utility(PlayerNum::Two) - sum,
        player_one_regret: info.player_regret(PlayerNum::One),
        player_two_regret: info.player_regret(PlayerNum::Two),
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
