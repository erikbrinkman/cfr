mod gambit;
mod json;

use cfr::{Game, GameTree, PlayerNum, RegretParams, SolveMethod, SolveParams};
use gambit::{GambitError, GambitNode};
use gambit_parser::ExtensiveFormGame;
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::io;
use std::io::{BufReader, Read};

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

    /// Seed for deterministic outcome sampling
    ///
    /// Sampling is fully deterministic given this seed, so repeated runs with the same seed produce
    /// identical strategies. Vary it to get independent samples.
    #[clap(short, long, value_parser, default_value_t = 0)]
    seed: u64,

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

#[derive(Debug, Serialize)]
struct Strategy(HashMap<String, HashMap<String, f64>>);

impl<I, A, S, T, N> From<I> for Strategy
where
    I: IntoIterator<Item = (S, A)>,
    A: IntoIterator<Item = (T, N)>,
    S: Display,
    T: Display,
    N: Borrow<f64>,
{
    fn from(named_strategies: I) -> Self {
        let mut map = HashMap::new();
        for (info, actions) in named_strategies {
            assert!(
                map.insert(
                    info.to_string(),
                    actions
                        .into_iter()
                        .filter(|(_, p)| p.borrow() > &0.0)
                        .map(|(a, p)| (a.to_string(), *p.borrow()))
                        .collect()
                )
                .is_none(),
                "internal error: found duplicate infosets"
            );
        }
        Strategy(map)
    }
}

#[derive(Debug, Serialize)]
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
    let mut buff = String::new();
    if args.input == "-" {
        io::stdin().lock().read_to_string(&mut buff).unwrap();
    } else {
        BufReader::new(File::open(&args.input).unwrap())
            .read_to_string(&mut buff)
            .unwrap();
    }
    let out = match args.input_format {
        InputFormat::Json => solve_json(&buff, &args),
        InputFormat::Gambit => solve_gambit(&buff, &args),
        InputFormat::Auto if args.input.ends_with(".json") => solve_json(&buff, &args),
        InputFormat::Auto if args.input.ends_with(".efg") => solve_gambit(&buff, &args),
        InputFormat::Auto => solve_auto(&buff, &args),
    }
    .unwrap();
    if args.output == "-" {
        serde_json::to_writer(io::stdout(), &out).unwrap();
    } else {
        serde_json::to_writer(File::create(args.output).unwrap(), &out).unwrap();
    };
}

/// A user-facing failure from reading or solving the input game.
enum CliError {
    /// The input wasn't valid json.
    ParseJson,
    /// The input wasn't a valid gambit `.efg` file.
    ParseGambit,
    /// The gambit game couldn't be read as a two-player zero-sum game.
    Gambit(GambitError),
    /// The game violated a structural invariant the solver requires.
    Materialize,
    /// Auto-detection matched no known format.
    UnknownFormat,
}

impl Display for CliError {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            CliError::ParseJson => write!(
                out,
                "couldn't parse json game definition : https://github.com/erikbrinkman/cfr#json-error"
            ),
            CliError::ParseGambit => write!(
                out,
                "couldn't parse gambit game definition : https://github.com/erikbrinkman/cfr#gambit-error"
            ),
            CliError::Gambit(err) => Display::fmt(err, out),
            CliError::Materialize => write!(
                out,
                "couldn't extract a compact game representation due to problems with the structure : https://github.com/erikbrinkman/cfr#game-error"
            ),
            CliError::UnknownFormat => write!(
                out,
                "couldn't parse any known format; try specifying your format with `--input-format` : https://github.com/erikbrinkman/cfr#auto-error"
            ),
        }
    }
}

// surface the Display message, not the variant, when `main` unwraps a failure
impl fmt::Debug for CliError {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, out)
    }
}

/// Parse a json game (the `&State` is the game) and solve it.
fn solve_json(buff: &str, args: &Args) -> Result<Output, CliError> {
    let state: json::State = serde_json::from_str(buff).map_err(|_| CliError::ParseJson)?;
    solve_game(&state, 0.0, args)
}

/// Parse a gambit game (the [`GambitNode`] cursor keeps borrowing the parsed tree) and solve it.
fn solve_gambit(buff: &str, args: &Args) -> Result<Output, CliError> {
    let parsed = ExtensiveFormGame::try_from(buff).map_err(|_| CliError::ParseGambit)?;
    let game = GambitNode::try_from(&parsed).map_err(CliError::Gambit)?;
    let sum = game.sum();
    solve_game(game, sum, args)
}

/// Try each known input format in turn.
fn solve_auto(buff: &str, args: &Args) -> Result<Output, CliError> {
    if let Ok(state) = serde_json::from_str::<json::State>(buff) {
        solve_game(&state, 0.0, args)
    } else if let Ok(parsed) = ExtensiveFormGame::try_from(buff) {
        let game = GambitNode::try_from(&parsed).map_err(CliError::Gambit)?;
        let sum = game.sum();
        solve_game(game, sum, args)
    } else {
        Err(CliError::UnknownFormat)
    }
}

/// Materialize a [`Game`] into a tree, solve it, and format the result. `sum` offsets the reported
/// utilities by the game's constant-sum value.
fn solve_game<G>(game: G, sum: f64, args: &Args) -> Result<Output, CliError>
where
    G: Game,
    G::Infoset: Eq + Hash + Display,
    G::Action: Eq + Hash + Display,
    G::ChanceInfoset: Eq + Hash,
{
    let game = GameTree::from_game(game).map_err(|_| CliError::Materialize)?;
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
            SolveParams {
                regret: args.discount.into_params(),
                seed: args.seed,
                ..Default::default()
            },
        )
        .unwrap();
    // build the unpruned strategy (converting labels to owned strings) before truncating in place,
    // so we never have to clone `Strategies`; only rebuild from the pruned tree if it's better
    let info = strategies.get_info();
    let [one, two] = strategies.as_named();
    let unpruned = [Strategy::from(one), Strategy::from(two)];
    strategies.truncate(args.clip_threshold);
    let pruned_info = strategies.get_info();
    let (info, [player_one_strategy, player_two_strategy]) =
        if pruned_info.regret() < info.regret() {
            let [one, two] = strategies.as_named();
            (pruned_info, [Strategy::from(one), Strategy::from(two)])
        } else {
            (info, unpruned)
        };
    Ok(Output {
        regret: info.regret(),
        player_one_utility: info.player_utility(PlayerNum::One) + sum,
        player_two_utility: info.player_utility(PlayerNum::Two) - sum,
        player_one_regret: info.player_regret(PlayerNum::One),
        player_two_regret: info.player_regret(PlayerNum::Two),
        player_one_strategy,
        player_two_strategy,
    })
}

#[cfg(test)]
mod tests {
    use super::{Args, CliError, GambitError, solve_auto, solve_gambit, solve_json};
    use clap::{CommandFactory, Parser};

    // matching pennies as a json game: a converging two-player game to drive the solve path
    const PENNIES: &str = r#"{ "player": { "player_one": true, "infoset": "p1", "actions": {
        "h": { "player": { "player_one": false, "infoset": "p2", "actions": {
            "h": { "terminal": 1.0 }, "t": { "terminal": -1.0 } } } },
        "t": { "player": { "player_one": false, "infoset": "p2", "actions": {
            "h": { "terminal": -1.0 }, "t": { "terminal": 1.0 } } } } } } }"#;

    fn default_args() -> Args {
        Args::parse_from(["cfr"])
    }

    #[test]
    fn test_cli() {
        Args::command().debug_assert();
    }

    #[test]
    fn auto_detects_json() {
        solve_auto(r#"{ "terminal": 0.0 }"#, &default_args()).unwrap();
    }

    #[test]
    fn auto_detects_gambit() {
        solve_auto(r#"EFG 2 R "" { "" "" } t "" 1 { 0 0 }"#, &default_args()).unwrap();
    }

    #[test]
    fn auto_rejects_unknown() {
        let err = solve_auto("random", &default_args()).unwrap_err();
        assert!(matches!(err, CliError::UnknownFormat));
    }

    #[test]
    fn error_messages_render() {
        let errors = [
            CliError::ParseJson,
            CliError::ParseGambit,
            CliError::Gambit(GambitError::NotConstantSum),
            CliError::Materialize,
            CliError::UnknownFormat,
        ];
        for err in errors {
            assert!(!err.to_string().is_empty());
            // Debug delegates to Display, so the message surfaces when `main` unwraps
            assert_eq!(format!("{err:?}"), err.to_string());
        }
    }

    #[test]
    fn reports_input_errors() {
        let args = default_args();
        assert!(matches!(solve_json("not json", &args), Err(CliError::ParseJson)));
        assert!(matches!(
            solve_gambit("not gambit", &args),
            Err(CliError::ParseGambit)
        ));
        // parses as gambit but has three players
        assert!(matches!(
            solve_gambit(r#"EFG 2 R "" { "" "" "" } t "" 1 { 0 0 0 }"#, &args),
            Err(CliError::Gambit(_))
        ));
        // parses as json but isn't a valid game (a zero-probability lone chance outcome)
        let bad = r#"{ "chance": { "outcomes": { "a": { "prob": 0.0, "state": { "terminal": 0.0 } } } } }"#;
        assert!(matches!(solve_json(bad, &args), Err(CliError::Materialize)));
    }

    #[test]
    fn solves_across_methods_and_discounts() {
        for method in ["full", "sampled", "external"] {
            for discount in ["vanilla", "lcfr", "cfr-plus", "dcfr", "dcfr-prune"] {
                let args = Args::parse_from(["cfr", "-m", method, "-d", discount, "-t", "50"]);
                let out = solve_json(PENNIES, &args).unwrap();
                assert!(out.regret.is_finite());
            }
        }
    }

    #[test]
    fn clip_threshold_runs_the_pruning_path() {
        let args = Args::parse_from(["cfr", "-c", "0.1", "-t", "200"]);
        let out = solve_json(PENNIES, &args).unwrap();
        assert!(out.regret.is_finite());
    }
}
