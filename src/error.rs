#[cfg(doc)]
use crate::Game;

/// Errors that result from game definition errors
///
/// If the object passed into [Game::from_root] doesn't conform to necessary
/// invariants, one of these will be returned.
#[derive(Debug, PartialEq, Eq)]
pub enum GameError {
    /// Returned when a chance node has no outcomes.
    EmptyChance,
    /// Returned when a chance node has a non-positive probability of happening
    NonPositiveChance,
    /// Returned when a chance node has different probabilities than another node in its infoset
    ProbabilitiesNotEqual,
    /// Returned when a games infosets don't exhibit perfect recall
    ///
    /// If a game does have perfect recall, then a player's infosets must form a tree, that is for
    /// all game nodes with a given infoset the players previous action node must also share the
    /// same infoset. We ignore this criterion for single action infosets since they don't actually
    /// reflect a decision.
    ImperfectRecall,
    /// Returned when a player node has no actions.
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
}

#[cfg(test)]
mod game_errors {
    use crate::{Game, GameError, GameNode, IntoGameNode, PlayerNum};

    struct Node(GameNode<Node>);

    impl IntoGameNode for Node {
        type PlayerInfo = &'static str;
        type Action = &'static str;
        type ChanceInfo = &'static str;
        type Outcomes = Vec<(f64, Node)>;
        type Actions = Vec<(&'static str, Node)>;

        fn into_game_node(self) -> GameNode<Self> {
            self.0
        }
    }

    #[test]
    fn empty_chance() {
        let err_game = Node(GameNode::Chance(None, [].into()));
        let err = Game::from_root(err_game).unwrap_err();
        assert_eq!(err, GameError::EmptyChance);
    }

    #[test]
    fn non_positive_chance() {
        let err_game = Node(GameNode::Chance(
            None,
            [(0.0, Node(GameNode::Terminal(0.0)))].into(),
        ));
        let err = Game::from_root(err_game).unwrap_err();
        assert_eq!(err, GameError::NonPositiveChance);
    }

    #[test]
    fn probabilities_not_equal() {
        let err_game = Node(GameNode::Chance(
            None,
            vec![
                (
                    0.5,
                    Node(GameNode::Chance(
                        Some("x"),
                        vec![
                            (1.0, Node(GameNode::Terminal(0.0))),
                            (1.0, Node(GameNode::Terminal(0.0))),
                        ],
                    )),
                ),
                (
                    0.5,
                    Node(GameNode::Chance(
                        Some("x"),
                        vec![
                            (1.0, Node(GameNode::Terminal(0.0))),
                            (2.0, Node(GameNode::Terminal(0.0))),
                        ],
                    )),
                ),
            ],
        ));
        let err = Game::from_root(err_game).unwrap_err();
        assert_eq!(err, GameError::ProbabilitiesNotEqual);
    }

    #[test]
    fn imperfect_recall() {
        let err_game = Node(GameNode::Chance(
            None,
            vec![
                (
                    0.5,
                    Node(GameNode::Player(
                        PlayerNum::One,
                        "x",
                        vec![
                            ("a", Node(GameNode::Terminal(0.0))),
                            ("b", Node(GameNode::Terminal(0.0))),
                        ],
                    )),
                ),
                (
                    0.5,
                    Node(GameNode::Player(
                        PlayerNum::One,
                        "y",
                        vec![
                            (
                                "a",
                                Node(GameNode::Player(
                                    PlayerNum::One,
                                    // NOTE forgot "y"
                                    "x",
                                    vec![
                                        ("a", Node(GameNode::Terminal(0.0))),
                                        ("b", Node(GameNode::Terminal(0.0))),
                                    ],
                                )),
                            ),
                            ("b", Node(GameNode::Terminal(0.0))),
                        ],
                    )),
                ),
            ],
        ));
        let err = Game::from_root(err_game).unwrap_err();
        assert_eq!(err, GameError::ImperfectRecall);
    }

    #[test]
    fn empty_player() {
        let err_game = Node(GameNode::Player(PlayerNum::One, "", [].into()));
        let err = Game::from_root(err_game).unwrap_err();
        assert_eq!(err, GameError::EmptyPlayer);
    }

    #[test]
    fn actions_not_equal() {
        let ord_game = Node(GameNode::Chance(
            None,
            vec![
                (
                    0.5,
                    Node(GameNode::Player(
                        PlayerNum::One,
                        "x",
                        vec![
                            ("a", Node(GameNode::Terminal(0.0))),
                            ("b", Node(GameNode::Terminal(0.0))),
                        ],
                    )),
                ),
                (
                    0.5,
                    Node(GameNode::Player(
                        PlayerNum::One,
                        "x",
                        vec![
                            ("b", Node(GameNode::Terminal(0.0))),
                            ("a", Node(GameNode::Terminal(0.0))),
                        ],
                    )),
                ),
            ],
        ));
        let err = Game::from_root(ord_game).unwrap_err();
        assert_eq!(err, GameError::ActionsNotEqual);
    }

    #[test]
    fn actions_not_unique() {
        let dup_game = Node(GameNode::Player(
            PlayerNum::One,
            "x",
            vec![
                ("a", Node(GameNode::Terminal(0.0))),
                ("a", Node(GameNode::Terminal(0.0))),
            ],
        ));
        let err = Game::from_root(dup_game).unwrap_err();
        assert_eq!(err, GameError::ActionsNotUnique);
    }
}

/// Errors that result from incompatible strategy representation
///
/// If a strategy object passed to [Game::from_named] doesn't match the games information and
/// action structure, one of these errors will be returned.
#[derive(Debug, PartialEq, Eq)]
pub enum StratError {
    /// Returned when the game doesn't have a specific infoset
    InvalidInfoset,
    /// Returned when the game doesn't have an action for an infoset
    InvalidInfosetAction,
    /// Returned when a probability for an action is negative, nan, or infinite
    InvalidProbability,
    /// Returned when every action in an infoset has zero probability
    InvalidInfosetProbability,
}

#[cfg(test)]
mod strat_errors {
    use crate::{Game, GameNode, IntoGameNode, PlayerNum, StratError};

    struct Node(GameNode<Node>);

    impl IntoGameNode for Node {
        type PlayerInfo = &'static str;
        type Action = &'static str;
        type ChanceInfo = &'static str;
        type Outcomes = Vec<(f64, Node)>;
        type Actions = Vec<(&'static str, Node)>;

        fn into_game_node(self) -> GameNode<Self> {
            self.0
        }
    }

    fn create_game() -> Game<&'static str, &'static str> {
        let node = Node(GameNode::Player(
            PlayerNum::One,
            "x",
            vec![(
                "a",
                Node(GameNode::Player(
                    PlayerNum::Two,
                    "z",
                    vec![
                        (
                            "b",
                            Node(GameNode::Player(
                                PlayerNum::One,
                                "y",
                                vec![
                                    ("c", Node(GameNode::Terminal(0.0))),
                                    ("d", Node(GameNode::Terminal(0.0))),
                                ],
                            )),
                        ),
                        ("c", Node(GameNode::Terminal(0.0))),
                    ],
                )),
            )],
        ));
        Game::from_root(node).unwrap()
    }

    #[test]
    fn invalid_infoset() {
        let game = create_game();
        let err = game
            .from_named([vec![("a", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfoset);

        let err = game
            .from_named_slow([vec![("a", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfoset);
    }

    #[test]
    fn invalid_infoset_action() {
        let game = create_game();
        let err = game
            .from_named([vec![("x", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfosetAction);

        let err = game
            .from_named_slow([vec![("x", vec![("b", 1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfosetAction);
    }

    #[test]
    fn invalid_probability() {
        let game = create_game();
        let err = game
            .from_named([vec![("x", vec![("a", -1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidProbability);

        let err = game
            .from_named_slow([vec![("x", vec![("a", -1.0)])], vec![]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidProbability);
    }

    #[test]
    fn invalid_infoset_probability_normal() {
        let game = create_game();
        let err = game
            .from_named([vec![("x", vec![("a", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfosetProbability);

        let err = game
            .from_named_slow([vec![("x", vec![("a", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfosetProbability);
    }

    #[test]
    fn invalid_infoset_probability_single() {
        let game = create_game();
        let err = game
            .from_named([vec![("y", vec![("d", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfosetProbability);

        let err = game
            .from_named_slow([vec![("y", vec![("d", 1.0)])], vec![("z", vec![("c", 1.0)])]])
            .unwrap_err();
        assert_eq!(err, StratError::InvalidInfosetProbability);
    }
}
