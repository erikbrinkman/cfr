#[cfg(doc)]
use crate::GameTree;
use rayon::ThreadPoolBuildError;
use std::error::Error;
use std::fmt::{Display, Error as FmtError, Formatter};

/// Errors that result from game definition errors
///
/// If a game (via [`GameTree::from_game`][crate::GameTree::from_game]) doesn't conform to necessary
/// invariants, one of these will be returned.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[non_exhaustive]
pub enum GameError {
    /// Returned when a chance node has no outcomes
    EmptyChance,
    /// Returned when a chance node has an outcome with a non-positive probability of happening
    NonPositiveChance,
    /// Returned when a chance node has different probabilities than another node in its infoset
    ProbabilitiesNotEqual,
    /// Returned when a game's infosets don't exhibit perfect recall
    ///
    /// For perfect recall a player's infosets must form a tree: every node sharing an infoset must
    /// agree on both the player's previous infoset and the action they took out of it. Single-action
    /// infosets are exempt since they reflect no decision.
    ImperfectRecall,
    /// Returned when a player node has no actions
    EmptyPlayer,
    /// Returned when the actions of a player node didn't match the order and values of an earlier
    /// information set.
    ///
    /// Make sure that all nodes with the same player and information set also have the same
    /// actions in the same order.
    ActionsNotEqual,
    /// Returned when the actions of a player node weren't unique.
    ///
    /// Make sure that all actions of a player node are unique.
    ActionsNotUnique,
    /// Returned when a game has more chance or player infosets than fit in a `u32`
    ///
    /// The solver keys its sampler on `u32` infoset ids, so each player's infoset count and the
    /// chance infoset count must not exceed [`u32::MAX`].
    TooManyInfosets,
}

impl Display for GameError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(fmt, "{self:?}")
    }
}

impl Error for GameError {}

/// An explicit game tree implementing [`Game`][crate::Game], used by the tests below to feed the
/// builder hand-crafted (often malformed) games. `&'static str` labels every infoset/action.
///
/// Children are shared via [`Rc`], so a node hands a child back to the builder by cloning a pointer
/// (O(1)) instead of deep-copying the subtree -- the whole build stays linear. Use the [`terminal`],
/// [`chance`], and [`player`] constructors so the trees read without `Rc::new` everywhere.
#[cfg(test)]
mod tree {
    use crate::{Game, Moves, NodeType, Outcomes, PlayerNum};
    use std::rc::Rc;

    pub(super) type Tree = Rc<Node>;

    #[derive(Debug)]
    pub(super) enum Node {
        Terminal(f64),
        Chance(Option<&'static str>, Vec<(f64, Tree)>),
        Player(PlayerNum, &'static str, Vec<(&'static str, Tree)>),
    }

    pub(super) fn terminal(payoff: f64) -> Tree {
        Rc::new(Node::Terminal(payoff))
    }
    pub(super) fn chance(infoset: Option<&'static str>, outcomes: Vec<(f64, Tree)>) -> Tree {
        Rc::new(Node::Chance(infoset, outcomes))
    }
    pub(super) fn player(
        num: PlayerNum,
        infoset: &'static str,
        actions: Vec<(&'static str, Tree)>,
    ) -> Tree {
        Rc::new(Node::Player(num, infoset, actions))
    }

    pub(super) struct TreeOutcomes(Vec<(f64, Tree)>);
    pub(super) struct TreeMoves(Vec<(&'static str, Tree)>);

    impl Game for Tree {
        type Action = &'static str;
        type Infoset = &'static str;
        type ChanceInfoset = &'static str;
        type Chance = TreeOutcomes;
        type Player = TreeMoves;

        fn into_node(self) -> NodeType<Self> {
            // cloning a child `Vec` is shallow -- it copies pointers (and bumps refcounts), never the
            // subtrees they point at
            match &*self {
                Node::Terminal(payoff) => NodeType::Terminal(*payoff),
                Node::Chance(infoset, outcomes) => {
                    NodeType::Chance(*infoset, TreeOutcomes(outcomes.clone()))
                }
                Node::Player(num, infoset, actions) => {
                    NodeType::Player(*num, *infoset, TreeMoves(actions.clone()))
                }
            }
        }
    }

    impl Outcomes<Tree> for TreeOutcomes {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn get(&self, index: usize) -> (f64, Tree) {
            self.0[index].clone()
        }
    }

    impl Moves<Tree> for TreeMoves {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn action(&self, index: usize) -> &'static str {
            self.0[index].0
        }
        fn apply(&self, index: usize) -> Tree {
            self.0[index].1.clone()
        }
    }
}

#[cfg(test)]
mod game_errors {
    use super::tree::{Tree, chance, player, terminal};
    use crate::{GameError, GameTree, PlayerNum};

    #[test]
    fn empty_chance() {
        let err_game = chance(None, vec![]);
        let err = GameTree::from_game(err_game).unwrap_err();
        assert_eq!(err, GameError::EmptyChance);
    }

    #[test]
    fn non_positive_chance() {
        let err_game = chance(None, vec![(0.0, terminal(0.0))]);
        let err = GameTree::from_game(err_game).unwrap_err();
        assert_eq!(err, GameError::NonPositiveChance);
    }

    #[test]
    fn probabilities_not_equal() {
        let err_game = chance(
            None,
            vec![
                (
                    0.5,
                    chance(
                        Some("x"),
                        vec![(1.0, terminal(0.0)), (1.0, terminal(0.0))],
                    ),
                ),
                (
                    0.5,
                    chance(
                        Some("x"),
                        vec![(1.0, terminal(0.0)), (2.0, terminal(0.0))],
                    ),
                ),
            ],
        );
        let err = GameTree::from_game(err_game).unwrap_err();
        assert_eq!(err, GameError::ProbabilitiesNotEqual);
    }

    #[test]
    fn imperfect_recall() {
        let err_game = chance(
            None,
            vec![
                (
                    0.5,
                    player(
                        PlayerNum::One,
                        "x",
                        vec![("a", terminal(0.0)), ("b", terminal(0.0))],
                    ),
                ),
                (
                    0.5,
                    player(
                        PlayerNum::One,
                        "y",
                        vec![
                            (
                                "a",
                                player(
                                    PlayerNum::One,
                                    // forgot that we played "y"
                                    "x",
                                    vec![
                                        ("a", terminal(0.0)),
                                        ("b", terminal(0.0)),
                                    ],
                                ),
                            ),
                            ("b", terminal(0.0)),
                        ],
                    ),
                ),
            ],
        );
        let err = GameTree::from_game(err_game).unwrap_err();
        assert_eq!(err, GameError::ImperfectRecall);
    }

    #[test]
    fn imperfect_recall_forgotten_action() {
        // both of player one's actions at "p" reach the same infoset "x", so at "x" the player can't
        // tell whether they played "a" or "b". Comparing only the previous infoset misses this (both
        // reach "x" from "p"); the action taken out of "p" is what distinguishes them.
        fn branch() -> Tree {
            player(
                PlayerNum::One,
                "x",
                vec![("c", terminal(0.0)), ("d", terminal(0.0))],
            )
        }
        let err_game = player(PlayerNum::One, "p", vec![("a", branch()), ("b", branch())]);
        let err = GameTree::from_game(err_game).unwrap_err();
        assert_eq!(err, GameError::ImperfectRecall);
    }

    #[test]
    fn empty_player() {
        let err_game = player(PlayerNum::One, "", vec![]);
        let err = GameTree::from_game(err_game).unwrap_err();
        assert_eq!(err, GameError::EmptyPlayer);
    }

    #[test]
    fn actions_not_equal() {
        let ord_game = chance(
            None,
            vec![
                (
                    0.5,
                    player(
                        PlayerNum::One,
                        "x",
                        vec![("a", terminal(0.0)), ("b", terminal(0.0))],
                    ),
                ),
                (
                    0.5,
                    player(
                        PlayerNum::One,
                        "x",
                        vec![("b", terminal(0.0)), ("a", terminal(0.0))],
                    ),
                ),
            ],
        );
        let err = GameTree::from_game(ord_game).unwrap_err();
        assert_eq!(err, GameError::ActionsNotEqual);
    }

    #[test]
    fn actions_not_unique() {
        let dup_game = player(
            PlayerNum::One,
            "x",
            vec![("a", terminal(0.0)), ("a", terminal(0.0))],
        );
        let err = GameTree::from_game(dup_game).unwrap_err();
        assert_eq!(err, GameError::ActionsNotUnique);
    }
}

/// Errors that result from incompatible strategy representation
///
/// If a strategy object passed to [`GameTree::from_named`] doesn't match the games information and
/// action structure, one of these errors will be returned.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[non_exhaustive]
pub enum StratError {
    /// Returned when the game doesn't have a specific infoset
    InvalidInfoset,
    /// Returned when the game doesn't have an action for an infoset
    InvalidAction,
    /// Returned when a probability for an action is negative, nan, or infinite
    InvalidProbability,
    /// Returned when no action in an infoset was assigned positive probability
    UninitializedInfoset,
}

impl Display for StratError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(fmt, "{self:?}")
    }
}

impl Error for StratError {}

#[cfg(test)]
mod strat_errors {
    use super::tree::{player, terminal};
    use crate::{GameTree, PlayerNum, StratError};

    fn create_game() -> GameTree<&'static str, &'static str> {
        let node = player(
            PlayerNum::One,
            "x",
            vec![(
                "a",
                player(
                    PlayerNum::Two,
                    "z",
                    vec![
                        (
                            "b",
                            player(
                                PlayerNum::One,
                                "y",
                                vec![
                                    ("c", terminal(0.0)),
                                    ("d", terminal(0.0)),
                                ],
                            ),
                        ),
                        ("c", terminal(0.0)),
                    ],
                ),
            )],
        );
        GameTree::from_game(node).unwrap()
    }

    #[test]
    fn invalid_infoset() {
        let game = create_game();
        let err = game
            .from_named([vec![("a", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfoset);

        let err = game
            .from_named_eq([vec![("a", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfoset);
    }

    #[test]
    fn invalid_infoset_action() {
        let game = create_game();
        let err = game
            .from_named([vec![("x", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidAction);

        let err = game
            .from_named_eq([vec![("x", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidAction);
    }

    #[test]
    fn invalid_probability() {
        let game = create_game();
        let err = game
            .from_named([vec![("x", vec![("a", -1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidProbability);

        let err = game
            .from_named_eq([vec![("x", vec![("a", -1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidProbability);
    }

    #[test]
    fn invalid_infoset_probability_normal() {
        let game = create_game();
        let err = game
            .from_named([vec![("x", vec![("a", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::UninitializedInfoset);

        let err = game
            .from_named_eq([vec![("x", vec![("a", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::UninitializedInfoset);
    }

    #[test]
    fn invalid_infoset_probability_single() {
        let game = create_game();
        let err = game
            .from_named([vec![("y", vec![("d", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::UninitializedInfoset);

        let err = game
            .from_named_eq([vec![("y", vec![("d", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::UninitializedInfoset);
    }
}

/// Errors that result from problems solving
///
/// Most of of these are either caused by invalid arguments or problems creating threads when using
/// multi-threaded solving. If explicitely using single threaded, or the input parameters are
/// validated in advance, these can be safely unwrapped.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[non_exhaustive]
pub enum SolveError {
    /// Returned when the requested number of threads was too large
    ThreadOverflow,
    /// Returned when a multi-threaded solver couldn't create a thread pool
    ThreadSpawnError,
}
impl Display for SolveError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(fmt, "{self:?}")
    }
}

impl Error for SolveError {}

impl From<ThreadPoolBuildError> for SolveError {
    fn from(_: ThreadPoolBuildError) -> Self {
        SolveError::ThreadSpawnError
    }
}
