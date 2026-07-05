mod gambit;
mod json;

use cfr::{Game, GameError, GameTree, PlayerNum, RegretParams, SolveMethod, SolveParams};
use gambit::{GambitError, GambitInfoset, GambitNode};
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

    /// Key gambit infosets by their always-unique number instead of name
    ///
    /// The default keys by name, which errors when distinct infosets share one.
    #[clap(long, value_parser)]
    infoset_numbers: bool,
}

#[derive(Debug, Serialize)]
struct Strategy(HashMap<String, HashMap<String, f64>>);

impl Strategy {
    /// Collect a named strategy into a JSON-serializable map, keying each infoset with `key`.
    ///
    /// Distinct infosets must key to distinct strings -- otherwise one would silently overwrite the
    /// other in the output object -- so a collision is surfaced as [`CliError::DuplicateInfoset`]
    /// rather than dropped or panicked on.
    fn from_named<S, A, T, N>(
        named_strategies: impl IntoIterator<Item = (S, A)>,
        key: impl Fn(S) -> String,
    ) -> Result<Self, CliError>
    where
        A: IntoIterator<Item = (T, N)>,
        T: Display,
        N: Borrow<f64>,
    {
        let mut map = HashMap::new();
        for (info, actions) in named_strategies {
            let name = key(info);
            let probs = actions
                .into_iter()
                .filter(|(_, p)| p.borrow() > &0.0)
                .map(|(a, p)| (a.to_string(), *p.borrow()))
                .collect();
            if map.insert(name.clone(), probs).is_some() {
                return Err(CliError::DuplicateInfoset(name));
            }
        }
        Ok(Strategy(map))
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
    /// The input wasn't valid json; carries serde_json's line/column detail.
    ParseJson(serde_json::Error),
    /// The input wasn't a valid gambit `.efg` file; carries the parser's rendered message (which
    /// borrows the input, so it's captured as an owned string).
    ParseGambit(String),
    /// The gambit game couldn't be read as a two-player zero-sum game.
    Gambit(GambitError),
    /// The game violated a structural invariant the solver requires.
    Materialize(GameError),
    /// Two distinct infosets render to the same output name (carries the shared name).
    DuplicateInfoset(String),
    /// Auto-detection matched no known format.
    UnknownFormat,
}

impl Display for CliError {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            CliError::ParseJson(err) => write!(
                out,
                "couldn't parse json game definition: {err} : https://github.com/erikbrinkman/cfr#json-error"
            ),
            CliError::ParseGambit(err) => write!(
                out,
                "couldn't parse gambit game definition: {err} : https://github.com/erikbrinkman/cfr#gambit-error"
            ),
            CliError::Gambit(err) => Display::fmt(err, out),
            CliError::Materialize(err) => write!(
                out,
                "couldn't extract a compact game representation due to problems with the structure ({err}) : https://github.com/erikbrinkman/cfr#game-error"
            ),
            CliError::DuplicateInfoset(name) => write!(
                out,
                "two distinct infosets share the name {name:?}; pass `--infoset-numbers` to key gambit infosets by number instead : https://github.com/erikbrinkman/cfr#duplicate-infosets"
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

/// Parse a json game (the `&State` is the game) and solve it. Json infosets are uniquely named, so
/// they're always keyed by name.
fn solve_json(buff: &str, args: &Args) -> Result<Output, CliError> {
    let state: json::State = serde_json::from_str(buff).map_err(CliError::ParseJson)?;
    solve_game(&state, 0.0, args, |info| info.to_string())
}

/// Solve a parsed gambit game, keying infosets by number or name per `--infoset-numbers`.
fn solve_gambit_game(game: GambitNode<'_, '_>, sum: f64, args: &Args) -> Result<Output, CliError> {
    if args.infoset_numbers {
        solve_game(game, sum, args, |info: &GambitInfoset<'_>| {
            info.number().to_string()
        })
    } else {
        solve_game(game, sum, args, |info: &GambitInfoset<'_>| info.to_string())
    }
}

/// Parse a gambit game (the [`GambitNode`] cursor keeps borrowing the parsed tree) and solve it.
fn solve_gambit(buff: &str, args: &Args) -> Result<Output, CliError> {
    let parsed =
        ExtensiveFormGame::try_from(buff).map_err(|err| CliError::ParseGambit(err.to_string()))?;
    let game = GambitNode::try_from(&parsed).map_err(CliError::Gambit)?;
    let sum = game.sum();
    solve_gambit_game(game, sum, args)
}

/// Try each known input format in turn.
fn solve_auto(buff: &str, args: &Args) -> Result<Output, CliError> {
    if let Ok(state) = serde_json::from_str::<json::State>(buff) {
        solve_game(&state, 0.0, args, |info| info.to_string())
    } else if let Ok(parsed) = ExtensiveFormGame::try_from(buff) {
        let game = GambitNode::try_from(&parsed).map_err(CliError::Gambit)?;
        let sum = game.sum();
        solve_gambit_game(game, sum, args)
    } else {
        Err(CliError::UnknownFormat)
    }
}

/// Materialize a [`Game`] into a tree, solve it, and format the result. `sum` offsets the reported
/// utilities by the game's constant-sum value.
fn solve_game<G>(
    game: G,
    sum: f64,
    args: &Args,
    infoset_key: impl Fn(&G::Infoset) -> String,
) -> Result<Output, CliError>
where
    G: Game,
    G::Infoset: Eq + Hash,
    G::Action: Eq + Hash + Display,
    G::ChanceInfoset: Eq + Hash,
{
    let game = GameTree::from_game(game).map_err(CliError::Materialize)?;
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
    let unpruned = [
        Strategy::from_named(one, &infoset_key)?,
        Strategy::from_named(two, &infoset_key)?,
    ];
    strategies.truncate(args.clip_threshold);
    let pruned_info = strategies.get_info();
    let (info, [player_one_strategy, player_two_strategy]) =
        if pruned_info.regret() < info.regret() {
            let [one, two] = strategies.as_named();
            (
                pruned_info,
                [
                    Strategy::from_named(one, &infoset_key)?,
                    Strategy::from_named(two, &infoset_key)?,
                ],
            )
        } else {
            (info, unpruned)
        };
    Ok(Output {
        regret: info.regret(),
        player_one_utility: info.player_utility(PlayerNum::One) + sum,
        player_two_utility: info.player_utility(PlayerNum::Two) + sum,
        player_one_regret: info.player_regret(PlayerNum::One),
        player_two_regret: info.player_regret(PlayerNum::Two),
        player_one_strategy,
        player_two_strategy,
    })
}

#[cfg(test)]
mod tests {
    use super::{Args, CliError, GameError, GambitError, solve_auto, solve_gambit, solve_json};
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
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let errors = [
            CliError::ParseJson(json_err),
            CliError::ParseGambit("error parsing game at: 'oops'".to_owned()),
            CliError::Gambit(GambitError::NotConstantSum),
            CliError::Materialize(GameError::ImperfectRecall),
            CliError::DuplicateInfoset("dup".to_owned()),
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
        assert!(matches!(
            solve_json("not json", &args),
            Err(CliError::ParseJson(_))
        ));
        assert!(matches!(
            solve_gambit("not gambit", &args),
            Err(CliError::ParseGambit(_))
        ));
        // parses as gambit but has three players
        assert!(matches!(
            solve_gambit(r#"EFG 2 R "" { "" "" "" } t "" 1 { 0 0 0 }"#, &args),
            Err(CliError::Gambit(_))
        ));
        // parses as json but isn't a valid game (a zero-probability lone chance outcome)
        let bad = r#"{ "chance": { "outcomes": { "a": { "prob": 0.0, "state": { "terminal": 0.0 } } } } }"#;
        assert!(matches!(
            solve_json(bad, &args),
            Err(CliError::Materialize(_))
        ));
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

    // two distinct player-two infosets that share a name (here both unnamed, so both empty): keyed by
    // name they collide, keyed by number they don't
    const SHARED_NAME_GAME: &str = r#"EFG 2 R "" { "" "" } p "" 1 1 "" { "a" "b" } 0 p "" 2 1 "" { "x" "y" } 0 t "" 1 { 1 -1 } t "" 2 { -1 1 } p "" 2 2 "" { "x" "y" } 0 t "" 3 { -1 1 } t "" 4 { 1 -1 }"#;

    #[test]
    fn duplicate_infoset_names_error_by_default() {
        // two distinct infosets sharing a name collide in the output map, so the CLI errors rather than
        // dropping one (this also covers unnamed infosets, whose name is the empty string)
        assert!(matches!(
            solve_gambit(SHARED_NAME_GAME, &default_args()),
            Err(CliError::DuplicateInfoset(_))
        ));
    }

    #[test]
    fn infoset_numbers_flag_resolves_collisions() {
        // keying by the (always unique) infoset number lets the same game solve
        let numbered = Args::parse_from(["cfr", "--infoset-numbers"]);
        let out = solve_gambit(SHARED_NAME_GAME, &numbered).unwrap();
        assert_eq!(out.player_two_strategy.0.len(), 2, "both p2 infosets present");
    }

    #[test]
    fn reports_both_constant_sum_utilities() {
        // a constant-sum-4 game (no decisions): chance pays player one 3 or 1 with equal odds, so
        // player one expects 2.0 and -- since payoffs sum to 4 -- player two also expects 2.0. The
        // constant-sum offset must be added back to *both* reported utilities.
        let game = r#"EFG 2 R "" { "" "" } c "" 1 "c" { "x" 1/2 "y" 1/2 } 0 t "" 1 { 3 1 } t "" 2 { 1 3 }"#;
        let out = solve_gambit(game, &default_args()).unwrap();
        assert!(
            (out.player_one_utility - 2.0).abs() < 1e-9,
            "player one: {}",
            out.player_one_utility
        );
        assert!(
            (out.player_two_utility - 2.0).abs() < 1e-9,
            "player two: {}",
            out.player_two_utility
        );
    }

    #[test]
    fn clip_threshold_runs_the_pruning_path() {
        let args = Args::parse_from(["cfr", "-c", "0.1", "-t", "200"]);
        let out = solve_json(PENNIES, &args).unwrap();
        assert!(out.regret.is_finite());
    }
}
