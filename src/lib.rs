//! Counterfactual Regret (CFR) is a library for finding an approximate nash equilibrium in
//! two-player zero-sum games of incomplete information, such as poker etc., using counterfactual
//! regret minimization[^cfr] and variants[^mccfr].
//!
//! # Usage
//!
//! To use this library first implement the traits [PlayerNode], [ChanceNode], and [TerminalNode]
//! for the data of your choice to specify a game tree in a standard [extensive form
//! game](https://en.wikipedia.org/wiki/Extensive-form_game). From there it can be turned into an
//! efficient game representation with [Game::from_node]. That game can then be
//! [solved](Game::solve), before adding the naming information back with [Game::name_strategy].
//!
//! If you want to tweak or create your own strategies, you can measure the regret of an arbitrary
//! compact strategy with [Game::regret].
//!
//! # Examples
//!
//! To compute a nash equilibrium, simply define a game by implementing the relevant traits on your
//! input data, then execute the following functions:
//! ```
//! use cfr::{Game, Solution};
//! # use cfr::{Node, PlayerNode, ChanceNode, Player, GameError};
//! # use std::iter::{Empty, empty};
//! # struct Chance;
//! # impl ChanceNode for Chance {
//! #     type PlayerNode = Play;
//! #     type Outcomes = Empty<(f64, ExNode)>;
//! #     fn into_outcomes(self) -> Self::Outcomes { empty() }
//! # }
//! # struct Play;
//! # impl PlayerNode for Play {
//! #     type Infoset = usize;
//! #     type Action = usize;
//! #     type ChanceNode = Chance;
//! #     type Actions = Empty<(usize, ExNode)>;
//! #     fn get_player(&self) -> Player { Player::One }
//! #     fn get_infoset(&self) -> usize { 0 }
//! #     fn into_actions(self) -> Self::Actions { empty() }
//! # }
//! # type ExNode = Node<Chance, Play>;
//! # let my_game = ExNode::Terminal{ player_one_payoff: 0.0 };
//! // my_game = ...
//! let game = Game::from_node(my_game).unwrap();
//! let Solution { regret, strategy } = game.solve_full(1000, 0.1);
//! let tighter_regret = game.regret(&strategy).unwrap();
//! let named_strategy = game.name_strategy(&strategy);
//! ```
//!
//! [^cfr]: [Zinkevich, Martin, et al. "Regret minimization in games with incomplete information."
//!   Advances in neural information processing systems 20
//!   (2007)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).
//! [^mccfr]: [Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive
//!   games." Advances in neural information processing systems 22
//!   (2009)](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).
#![warn(missing_docs)]
use core::slice;
use rand::thread_rng;
use rand_distr::weighted_alias::WeightedAliasIndex;
use rand_distr::{Distribution, WeightedError};
use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::Zip;
use std::ops::Deref;
use std::vec;

// ------ //
// Public //
// ------ //

// FIXME generally rethink and reorganize these errors

/// Errors that result from game definition errors
///
/// If the object passed into [Game::from_node] doesn't conform to necessary
/// invariants, one of these will be returned.
#[derive(Debug, PartialEq)]
pub enum GameError<I, A> {
    /// Returned when a chance node has no outcomes.
    EmptyChance,
    /// Returned when a chance node has a non-positive probability of happening
    NonPositiveChance {
        /// The degenerate probability
        prob: f64,
    },
    /// Returned when a chance node has a single outcome.
    DegenerateChance,
    /// There are too many outcomes in a chance node
    TooManyOutcomes,
    /// Returned when a games infosets don't exhibit perfect recall
    ///
    /// If a game does have perfect recall, then a player's infosets must form a tree, that is for
    /// all game nodes with a given infoset the players previous action node must also share the
    /// same infoset.
    ImperfectRecall {
        /// The player with imperfect recall
        player: Player,
        /// The infoset with imperfect recall
        infoset: I,
    },
    /// Returned when a player node has no actions.
    EmptyPlayer,
    /// Returned when a player node has a single action
    DegeneratePlayer,
    /// Returned when the actions of a player node didn't match the order and values of an earlier
    /// information set.
    ///
    /// Make sure that all nodes with the same player and information set also have the same
    /// actions in the same order.
    ActionsNotEqual {
        /// The player of the node in question.
        player: Player,
        /// The information set in question.
        infoset: I,
        /// The actions of the node that were different than expected.
        actions: Vec<A>,
    },
    /// Returned when the actions of a player node weren't unique.
    ///
    /// Make sure that all actions of a player node are unique.
    ActionsNotUnique {
        /// The player of the node in question.
        player: Player,
        /// The information set in question.
        infoset: I,
        /// The actions of the node that contained at least one duplicate action.
        actions: Vec<A>,
    },
}

impl<I, A> From<WeightedError> for GameError<I, A> {
    fn from(err: WeightedError) -> Self {
        match err {
            WeightedError::NoItem => Self::EmptyChance,
            WeightedError::InvalidWeight => panic!("unexpected invalid weight"),
            WeightedError::AllWeightsZero => Self::DegenerateChance,
            WeightedError::TooMany => Self::TooManyOutcomes,
        }
    }
}

/// Errors that result from incompatible compact strategy representations
///
/// If a strategy object passed to a [Game] doesn't match the games information and action
/// structure (notably [Game::regret]), one of these errors will be returned.
#[derive(Debug, PartialEq)]
pub enum CompactError {
    /// The number of information sets in the the strategy didn't match the number in the game
    ///
    /// Make sure that the length of strategy is the same as the original game.
    InfosetNum {
        /// The number of information sets in the underlying game.
        game_num: usize,
        /// The number of information sets of the passed in strategy.
        strategy_num: usize,
    },
    /// The number of actions in a strategy didn't match the number of actions in the associated
    /// information set
    ///
    /// Make sure that the passed in strategy has the same number of actions as each information
    /// set in the original game.
    ActionNum {
        /// The index where they differed
        ind: usize,
        /// The number of actions in the game's information set
        infoset_actions: usize,
        /// The number of actions in the passed in strategy
        strategy_actions: usize,
    },
    /// The total probability of all actions in an information set didn't sum one
    ///
    /// Try making sure that all games information sets sum to one.
    StratNotProbability {
        /// The information set index that's strategies didn't sum to one.
        ind: usize,
        /// The total probability the information set's actions did sum to.
        total_prob: f64,
    },
}

/// Errors that result from incompatible strategy representation
///
/// If a strategy object passed to [Game::compact_strategy] doesn't match the games information and action
/// structure, one of these errors will be returned.
#[derive(Debug, PartialEq)]
pub enum StratError<I, A> {
    /// Returned when the game doesn't have an action for the infoset of the specified player
    ///
    /// The player may not ever see the infoset, or the action may be an invalid action for the
    /// specified infoset.
    InvalidInfosetAction {
        /// The player corresponding to the invalid action
        player: Player,
        /// The invalid infoset
        infoset: I,
        /// The invalid action
        action: A,
    },
    /// Returned when a probability for an action isn't in [0, 1]
    InvalidProbability {
        /// The player corresponding to the invalid probability
        player: Player,
        /// The valid infoset with the invalid probability
        infoset: I,
        /// The valid action with the invalid probability
        action: A,
        /// The invalid probability
        prob: f64,
    },
    /// Returned when the total probability of each action in an infoset doesn't sum to one
    InvalidInfosetProbability {
        /// The player corresponding to the invalid infoset
        player: Player,
        /// The infoset where the probability doesn't sum to one
        infoset: I,
        /// The total probability of the infoset
        total_prob: f64,
    },
}

/// An enum indicating which player a state in the tree belongs to
#[derive(Debug, Copy, Eq, Clone, PartialEq, Hash)]
pub enum Player {
    /// The first player
    One,
    /// The second player
    Two,
}

impl Player {
    fn ind(&self) -> usize {
        match self {
            Player::One => 0,
            Player::Two => 1,
        }
    }
}

/// An enum for distinguishing between game node types
///
/// A game node can be one of a terminal node with the payoff for player one, a chance node with a
/// fixed probability of advancing, or a player node indicating a place where a player can make a
/// decision.
#[derive(Debug)]
pub enum Node<C, P> {
    /// A terminal node, the game is over
    Terminal {
        /// The payoff to player one at this conclusion
        ///
        /// The payoff to player two is inherently the negative of this.
        player_one_payoff: f64,
    },
    /// A chance node, the game advances independent of player action
    Chance(C),
    /// a node in the tree where the player can choose between different actions
    Player(P),
}

/// A node with several outcomes, each which has a different probability of outcome
///
/// This node represents randomness in the game independent of player action. Each outcome should
/// have a positive weight proportional to its probability of happening.
pub trait ChanceNode: Sized {
    type PlayerNode;
    type Outcomes: IntoIterator<Item=(f64, Node<Self, Self::PlayerNode>)>;

    fn into_outcomes(self) -> Self::Outcomes;
}

/// A node indicating a player decision point in the game
///
/// Nodes are iterables over an action the player can choose and future states. In addition, player
/// nodes must have an assigned player who's acting, and a marker for the information set
/// indicating what the player knows at this stage of the game.
pub trait PlayerNode: Sized {
    type Infoset;
    type Action;
    type ChanceNode;
    type Actions: IntoIterator<Item = (Self::Action, Node<Self::ChanceNode, Self>)>;

    /// Get the player who's acting at this node in the game tree
    fn get_player(&self) -> Player;

    /// Get the information set of this game node
    ///
    /// An information set serves as a comapct representation of the information that a player has
    /// available when making a decision. A payer can not distinguish (and thus must play the same
    /// strategy) at each node that has the same information set.
    fn get_infoset(&self) -> Self::Infoset;

    fn into_actions(self) -> Self::Actions;
}

// --------- //
// Internals //
// --------- //

#[derive(Debug)]
struct Random {
    outcomes: Box<[(f64, Vertex)]>,
    // NOTE that alias sampling will increase creation time and memory
    sampler: WeightedAliasIndex<f64>,
}

impl Random {
    fn new(data: impl Into<Box<[(f64, Vertex)]>>) -> Result<Random, WeightedError> {
        let outcomes = data.into();
        let sampler = WeightedAliasIndex::new(outcomes.iter().map(|(w, _)| *w).collect())?;
        Ok(Random { outcomes, sampler })
    }

    fn sample_vertex(&self) -> &Vertex {
        let ind = self.sampler.sample(&mut thread_rng());
        let (_, vert) = &self.outcomes[ind];
        vert
    }
}

#[derive(Debug)]
struct Agent {
    player: Player,
    infoset: usize,
    actions: Box<[Vertex]>,
}

type Vertex = Node<Random, Agent>;

#[derive(Debug)]
struct Infoset<I, A> {
    player: Player,
    infoset: I,
    actions: Box<[A]>,
    prev_infoset: usize,
}

impl<I, A> Infoset<I, A> {
    fn new(
        player: Player,
        infoset: I,
        actions: impl Into<Box<[A]>>,
        prev_infoset: Option<usize>,
    ) -> Infoset<I, A> {
        assert_ne!(
            prev_infoset,
            Some(usize::MAX),
            "can't process more than a machine word of infosets"
        );
        Infoset {
            player,
            infoset,
            actions: actions.into(),
            prev_infoset: prev_infoset.unwrap_or(usize::MAX),
        }
    }
}

trait AgnosticInfoset {
    fn player(&self) -> Player;
    fn num_actions(&self) -> usize;
    fn prev_infoset(&self) -> Option<usize>;
}

impl<I, A> AgnosticInfoset for Infoset<I, A> {
    fn player(&self) -> Player {
        self.player
    }

    fn num_actions(&self) -> usize {
        self.actions.len()
    }

    fn prev_infoset(&self) -> Option<usize> {
        if self.prev_infoset == usize::MAX {
            None
        } else {
            Some(self.prev_infoset)
        }
    }
}

/// A compact game representation that includes strategies for finding approximate nash equilibria
/// and computing regret of strategies.
#[derive(Debug)]
pub struct Game<I, A> {
    infosets: Box<[Infoset<I, A>]>,
    start: Vertex,
}

impl<I, A> Game<I, A> {
    /// Create a game from the root node of an arbitrary game tree
    ///
    /// # Examples
    ///
    /// FIXME
    pub fn from_node<C, P>(start: Node<C, P>) -> Result<Game<I, A>, GameError<I, A>>
    where
        I: Hash + Eq + Clone,
        A: Hash + Eq,
        C: ChanceNode<PlayerNode=P>,
        P: PlayerNode<Infoset=I, Action=A, ChanceNode=C>,
    {
        let mut infosets = Vec::new();
        let mut inds = HashMap::<(Player, I), usize>::new();
        let start = Game::recurse_vertex_from_node(&mut infosets, &mut inds, start, [None; 2])?;

        Ok(Game {
            infosets: infosets.into_boxed_slice(),
            start,
        })
    }

    fn recurse_vertex_from_node<C, P>(
        infosets: &mut Vec<Infoset<I, A>>,
        inds: &mut HashMap<(Player, I), usize>,
        node: Node<C, P>,
        mut prev_infosets: [Option<usize>; 2],
    ) -> Result<Vertex, GameError<I, A>>
    where
        I: Hash + Eq + Clone,
        A: Hash + Eq,
        C: ChanceNode<PlayerNode=P>,
        P: PlayerNode<Infoset=I, Action=A, ChanceNode=C>,
    {
        match node {
            Node::Terminal { player_one_payoff } => Ok(Vertex::Terminal { player_one_payoff }),
            Node::Chance(chance) => {
                let mut total = 0.0;
                let mut outcomes = Vec::new();
                for (prob, next) in chance.into_outcomes() {
                    if prob <= 0.0 || !prob.is_finite() {
                        Err(GameError::NonPositiveChance { prob })
                    } else {
                        let next_vert =
                            Game::recurse_vertex_from_node(infosets, inds, next, prev_infosets)?;
                        outcomes.push((prob, next_vert));
                        total += prob;
                        Ok(())
                    }?;
                }
                if outcomes.is_empty() {
                    Err(GameError::EmptyChance)
                } else if outcomes.len() == 1 {
                    Err(GameError::DegenerateChance)
                } else {
                    // renormalize to make sure consistency
                    for (prob, _) in &mut outcomes {
                        *prob /= total;
                    }
                    Ok(Vertex::Chance(Random::new(outcomes)?))
                }
            }
            Node::Player(play) => {
                let player = play.get_player();
                let infoset = play.get_infoset();
                let mut actions = Vec::new();
                let mut nexts = Vec::new();
                for (action, next) in play.into_actions() {
                    actions.push(action);
                    nexts.push(next);
                }
                match actions.len() {
                    0 => Err(GameError::EmptyPlayer),
                    1 => Err(GameError::DegeneratePlayer),
                    _ => Ok(()),
                }?;
                let info_ind = match inds.entry((player, infoset.clone())) {
                    Entry::Occupied(ent) => {
                        let ind = ent.get();
                        let info = &mut infosets[*ind];
                        if *info.actions != *actions {
                            Err(GameError::ActionsNotEqual {
                                player,
                                infoset,
                                actions,
                            })
                        } else if info.prev_infoset() != prev_infosets[player.ind()] {
                            Err(GameError::ImperfectRecall { player, infoset })
                        } else {
                            Ok(*ind)
                        }
                    }
                    Entry::Vacant(ent) => {
                        let hash_names: HashSet<&A> = actions.iter().collect();
                        if hash_names.len() == actions.len() {
                            let info = infosets.len();
                            infosets.push(Infoset::new(
                                player,
                                infoset,
                                actions.into_boxed_slice(),
                                prev_infosets[player.ind()],
                            ));
                            ent.insert(info);
                            Ok(info)
                        } else {
                            Err(GameError::ActionsNotUnique {
                                player,
                                infoset,
                                actions,
                            })
                        }
                    }
                }?;
                prev_infosets[player.ind()] = Some(info_ind);
                let next_verts = nexts
                    .into_iter()
                    .map(|next| Game::recurse_vertex_from_node(infosets, inds, next, prev_infosets))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Vertex::Player(Agent {
                    player,
                    infoset: info_ind,
                    actions: next_verts.into_boxed_slice(),
                }))
            }
        }
    }

    /// Find an approximate Nash equilibrium of the current game
    ///
    /// This will run no more than `max_iter` iterations, and terminate early if it can guarantee
    /// that it's found a solution with regret smaller than `max_reg`.
    ///
    /// # Examples
    ///
    /// FIXME
    pub fn solve_full(&self, max_iter: u64, max_reg: f64) -> Solution {
        full::solve(&self.start, &self.infosets, max_iter, max_reg)
    }

    /// FIXME document, also name this like the actual method it uses
    /// Lanctot 2009
    pub fn solve_external(&self, max_steps: u64, step_size: u64, max_reg: f64) -> Solution {
        external::solve(&self.start, &self.infosets, max_steps, step_size, max_reg)
    }

    /// Compute the regret of a strategy like object
    ///
    /// # Examples
    ///
    /// FIXME
    // FIXME I should change these strategy implementations to the minimum type required to
    // actually do what this specific method needs, e.g. if it needs iterable, then iterable, if it
    // needs index then that, etc. IT should be such that the dense strategy implementation works
    // for all of them.
    pub fn regret(&self, strategy: &[impl AsRef<[f64]>]) -> Result<EquilibriumInfo, CompactError> {
        regret::regret(&self.start, &self.infosets, strategy)
    }

    /// attach player, infoset, and action information to a strategy
    ///
    /// # Examples
    ///
    /// FIXME
    pub fn name_strategy<'a, R>(
        &'a self,
        strategy: &'a [R],
    ) -> Result<NamedStrategies<'a, I, A, R>, CompactError>
    where
        R: AsRef<[f64]>,
    {
        validate_strategy(&self.infosets, strategy)?;
        Ok((
            NamedStrategyIter::new(Player::One, &self.infosets, strategy),
            NamedStrategyIter::new(Player::Two, &self.infosets, strategy),
        ))
    }

    /// convert a named strategy representation into a compact representation that can be used for
    /// regret calculation
    ///
    /// # Examples
    ///
    /// FIXME
    ///
    /// # Remarks
    ///
    /// The clone attribute is only necessary for returning error messages with extra information.
    /// In principle it's possible to avoid cloning, but at the cost of an uglier interface.
    pub fn compact_strategy(
        &self,
        one: impl IntoIterator<Item = (impl Borrow<I>, impl Borrow<A>, impl Borrow<f64>)>,
        two: impl IntoIterator<Item = (impl Borrow<I>, impl Borrow<A>, impl Borrow<f64>)>,
    ) -> Result<DenseStrategy, StratError<I, A>>
    where
        I: Eq + Hash + Clone,
        A: Eq + Hash + Clone,
    {
        let mut one_inds = HashMap::new();
        let mut two_inds = HashMap::new();
        for (ii, infoset) in self.infosets.iter().enumerate() {
            let inds = match infoset.player {
                Player::One => &mut one_inds,
                Player::Two => &mut two_inds,
            };
            for (ai, act) in infoset.actions.iter().enumerate() {
                inds.insert((&infoset.infoset, act), (ii, ai));
            }
        }

        let mut strategy: Vec<_> = self
            .infosets
            .iter()
            .map(|info| vec![0.0; info.actions.len()].into_boxed_slice())
            .collect();
        Self::assign_from_named(Player::One, one, one_inds, &mut strategy)?;
        Self::assign_from_named(Player::Two, two, two_inds, &mut strategy)?;

        for (strat, info) in strategy.iter_mut().zip(self.infosets.iter()) {
            let total: f64 = strat.iter().sum();
            if (total - 1.0).abs() > 1e-6 {
                Err(StratError::InvalidInfosetProbability {
                    player: info.player,
                    infoset: info.infoset.clone(),
                    total_prob: total,
                })
            } else {
                // normalize
                for prob in strat.iter_mut() {
                    *prob /= total
                }
                Ok(())
            }?;
        }

        Ok(DenseStrategy(strategy.into_boxed_slice()))
    }

    fn assign_from_named<'a>(
        player: Player,
        named_strat: impl IntoIterator<Item = (impl Borrow<I>, impl Borrow<A>, impl Borrow<f64>)>,
        inds: HashMap<(&'a I, &'a A), (usize, usize)>,
        strategy: &mut [Box<[f64]>],
    ) -> Result<(), StratError<I, A>>
    where
        I: Eq + Hash + Clone,
        A: Eq + Hash + Clone,
    {
        for (binfoset, baction, bprob) in named_strat {
            let infoset = binfoset.borrow();
            let action = baction.borrow();
            let prob = bprob.borrow();
            match inds.get(&(infoset, action)) {
                Some(&(ii, ai)) if (0.0..=1.0).contains(prob) => {
                    strategy[ii][ai] = *prob;
                    Ok(())
                }
                None => Err(StratError::InvalidInfosetAction {
                    player,
                    infoset: infoset.clone(),
                    action: action.clone(),
                }),
                _ => Err(StratError::InvalidProbability {
                    player,
                    infoset: infoset.clone(),
                    action: action.clone(),
                    prob: *prob,
                }),
            }?;
        }
        Ok(())
    }
}

/// This is a module the implements the full vanilla counter-factual regret minimization algorithm
mod full {
    use super::{AgnosticInfoset, DenseStrategy, Node, Player, Solution, Vertex};

    #[derive(Debug)]
    struct RegretInfoset {
        // TODO this could be updated to have one buffer of heap memory that we chunk since we know the
        // chunks will have the same size.
        cum_regret: Box<[f64]>,
        utils: Box<[f64]>,
        probs: Box<[f64]>,
        cum_probs: Box<[f64]>,
        prob_reach: f64,
        cum_prob_reach: f64,
    }

    impl RegretInfoset {
        fn new(num_actions: usize) -> RegretInfoset {
            RegretInfoset {
                cum_regret: vec![0.0; num_actions].into_boxed_slice(),
                utils: vec![0.0; num_actions].into_boxed_slice(),
                probs: vec![1.0 / num_actions as f64; num_actions].into_boxed_slice(),
                cum_probs: vec![0.0; num_actions].into_boxed_slice(),
                prob_reach: 0.0,
                cum_prob_reach: 0.0,
            }
        }

        fn update(&mut self, cardinal_iter: u64) -> f64 {
            // compute expected utility
            let expected_util: f64 = self
                .utils
                .iter()
                .zip(self.probs.iter())
                .map(|(util, prob)| util * prob)
                .sum();
            // use expected utility to update cumulative average regret
            let mut avg_reg = 0.0;
            for (cum_reg, util) in self.cum_regret.iter_mut().zip(self.utils.iter()) {
                let reg = util - expected_util;
                *cum_reg += (reg - *cum_reg) / cardinal_iter as f64;
                avg_reg = f64::max(avg_reg, *cum_reg);
            }
            // update average probability proportional to reach probability
            self.cum_prob_reach += self.prob_reach;
            for (cum_prob, prob) in self.cum_probs.iter_mut().zip(self.probs.iter()) {
                *cum_prob += (*prob - *cum_prob) * self.prob_reach / self.cum_prob_reach;
            }
            // set probabilities to cumulative regret
            let mut total = 0.0;
            for (prob, cum_reg) in self.probs.iter_mut().zip(self.cum_regret.iter()) {
                *prob = f64::max(*cum_reg, 0.0);
                total += *prob;
            }
            if total == 0.0 {
                // FIXME alternate assignments
                self.probs.fill(1.0 / self.utils.len() as f64);
            } else {
                for prob in self.probs.iter_mut() {
                    *prob /= total;
                }
            }
            // reset reach prob and utilities
            self.utils.fill(0.0);
            self.prob_reach = 0.0;
            avg_reg
        }
    }

    fn recurse(
        infosets: &mut [RegretInfoset],
        node: &Vertex,
        p_chance: f64,
        p_player: [f64; 2],
    ) -> f64 {
        match node {
            Node::Terminal { player_one_payoff } => *player_one_payoff,
            Node::Chance(rand) => {
                let mut expected = 0.0;
                for (prob, next) in rand.outcomes.iter() {
                    expected += prob * recurse(infosets, next, p_chance * prob, p_player);
                }
                expected
            }
            Node::Player(agent) => {
                // NOTE this uses zero-sum nature to compute payoff for player 2
                let mult = match agent.player {
                    Player::One => p_chance * p_player[1],
                    Player::Two => -p_chance * p_player[0],
                };
                infosets[agent.infoset].prob_reach += p_player[agent.player.ind()];
                let mut expected = 0.0;
                for (i, next) in agent.actions.iter().enumerate() {
                    let prob = infosets[agent.infoset].probs[i];
                    let mut p_next = p_player;
                    p_next[agent.player.ind()] *= prob;
                    let util = recurse(infosets, next, p_chance, p_next);
                    infosets[agent.infoset].utils[i] += mult * util;
                    expected += prob * util;
                }
                expected
            }
        }
    }

    pub(super) fn solve(
        start: &Vertex,
        info: &[impl AgnosticInfoset],
        max_iter: u64,
        max_reg: f64,
    ) -> Solution {
        // FIXME make this range and panic on 0 max_steps and test panic
        // TODO This allocates a bunch of memory independentally. We could do it in one allocation,
        // but it makes this much more complicated, and it's not clear that the single allocation
        // and locality is worth the complexity
        let mut infosets: Vec<_> = info
            .iter()
            .map(|info| RegretInfoset::new(info.num_actions()))
            .collect();
        let mut reg = max_reg;
        for it in 0..max_iter {
            // go through self and update utilities and probability of reaching a node
            recurse(&mut infosets, start, 1.0, [1.0; 2]);

            // update infoset information for next iteration
            let mut regs = [0.0; 2];
            for (infoset, inf) in infosets.iter_mut().zip(info.iter()) {
                regs[inf.player().ind()] += infoset.update(it + 1);
            }
            reg = 2.0 * f64::max(regs[0], regs[1]);
            if reg < max_reg {
                break;
            }
        }
        let strat: Vec<_> = infosets.into_iter().map(|info| info.cum_probs).collect();
        Solution {
            regret: reg,
            strategy: DenseStrategy(strat.into_boxed_slice()),
        }
    }
}

/// This is a module the implements the external sampled montecarlo counter-factual regret
/// minimization algorithm FIXME
mod external {
    use super::{regret, AgnosticInfoset, DenseStrategy, Node, Player, Solution, Vertex};
    use rand::random;

    #[derive(Debug)]
    struct RegretInfoset {
        // TODO this could be updated to have one buffer of heap memory that we chunk since we know the
        // chunks will have the same size.
        cum_regret: Box<[f64]>,
        probs: Box<[f64]>,
        cum_probs: Box<[f64]>,
        updates: u64,
        cum_prob_reach: f64,
        sampled_ind: usize,
        last_sample: u64,
    }

    impl RegretInfoset {
        fn new(num_actions: usize) -> RegretInfoset {
            RegretInfoset {
                cum_regret: vec![0.0; num_actions].into_boxed_slice(),
                probs: vec![1.0 / num_actions as f64; num_actions].into_boxed_slice(),
                cum_probs: vec![0.0; num_actions].into_boxed_slice(),
                updates: 0,
                cum_prob_reach: 0.0,
                sampled_ind: 0,
                last_sample: 0,
            }
        }

        fn sample_ind(&mut self, seq: u64) -> usize {
            if self.last_sample == seq {
                self.sampled_ind
            } else {
                self.last_sample = seq;
                self.sampled_ind = self.probs.len() - 1;
                let mut sample: f64 = random();
                for (ind, prob) in self.probs.iter().enumerate() {
                    if sample < *prob {
                        self.sampled_ind = ind;
                        break;
                    } else {
                        sample -= prob;
                    }
                }
                self.sampled_ind
            }
        }

        fn update_probs(&mut self, prob_reach: f64) {
            if prob_reach > 0.0 {
                self.cum_prob_reach += prob_reach;
                for (cum_prob, prob) in self.cum_probs.iter_mut().zip(self.probs.iter()) {
                    *cum_prob += (*prob - *cum_prob) * prob_reach / self.cum_prob_reach;
                }
            }
        }

        fn update(&mut self, expected: f64, seq: u64) {
            // update regret
            let diff = (seq - self.updates) as f64;
            self.updates = seq;
            for (total_reg, util) in self.cum_regret.iter_mut().zip(self.probs.iter()) {
                let reg = *util - expected;
                *total_reg += (reg - *total_reg * diff) / self.updates as f64;
            }
            // update probability
            let total: f64 = self.cum_regret.iter().map(|r| f64::max(0.0, *r)).sum();
            if total == 0.0 {
                // FIXME alternate ways to assign this
                self.probs.fill(1.0 / self.probs.len() as f64);
            } else {
                for (total_reg, prob) in self.cum_regret.iter().zip(self.probs.iter_mut()) {
                    *prob = f64::max(*total_reg, 0.0) / total;
                }
            }
        }
    }

    fn recurse(
        infosets: &mut [RegretInfoset],
        node: &Vertex,
        target: Player,
        seq: u64,
        p_reach: f64,
    ) -> f64 {
        match node {
            Node::Terminal { player_one_payoff } => match target {
                Player::One => *player_one_payoff,
                Player::Two => -player_one_payoff,
            },
            Node::Chance(rand) => recurse(infosets, rand.sample_vertex(), target, seq, p_reach),
            Node::Player(agent) if agent.player != target => {
                let ind = infosets[agent.infoset].sample_ind(seq);
                recurse(infosets, &agent.actions[ind], target, seq, p_reach)
            }
            Node::Player(agent) => {
                infosets[agent.infoset].update_probs(p_reach);
                let mut expected = 0.0;
                for (i, next) in agent.actions.iter().enumerate() {
                    let prob = infosets[agent.infoset].probs[i];
                    let util = recurse(infosets, next, target, seq, p_reach * prob);
                    infosets[agent.infoset].probs[i] = util;
                    expected += util * prob;
                }
                infosets[agent.infoset].update(expected, seq);
                expected
            }
        }
    }

    pub(super) fn solve(
        start: &Vertex,
        info: &[impl AgnosticInfoset],
        max_steps: u64,
        step_size: u64,
        max_reg: f64,
    ) -> Solution {
        // FIXME test assertion
        assert_ne!(step_size, 0, "can't set zero step_size");
        assert_ne!(max_steps, 0, "can't set zero max_steps");
        let mut infosets: Vec<_> = info
            .iter()
            .map(|info| RegretInfoset::new(info.num_actions()))
            .collect();
        let mut reg = f64::NAN;
        for step in 0..max_steps {
            let init = step * step_size + 1;
            for it in init..(init + step_size) {
                recurse(&mut infosets, start, Player::One, it, 1.0);
                recurse(&mut infosets, start, Player::Two, it, 1.0);
            }
            // FIXME avoid copy with appropriate trait implementations
            let strat: Vec<_> = infosets
                .iter()
                .map(|info| {
                    if info.cum_prob_reach == 0.0 {
                        vec![1.0 / info.cum_probs.len() as f64; info.cum_probs.len()]
                            .into_boxed_slice()
                    } else {
                        info.cum_probs.clone()
                    }
                })
                .collect();
            let info = regret::regret(start, info, &strat).unwrap();
            reg = info.regret();
            if reg < max_reg {
                break;
            }
        }
        // FIXME if cumprobs is 0 / steps isn't updated than do uniform
        let strat: Vec<_> = infosets
            .into_iter()
            .map(|info| {
                if info.cum_prob_reach == 0.0 {
                    vec![1.0 / info.cum_probs.len() as f64; info.cum_probs.len()].into_boxed_slice()
                } else {
                    info.cum_probs
                }
            })
            .collect();
        Solution {
            regret: reg,
            strategy: DenseStrategy(strat.into_boxed_slice()),
        }
    }
}

/// private module for computing regret
mod regret {
    use super::{
        validate_strategy, Agent, AgnosticInfoset, CompactError, EquilibriumInfo, Node, Player,
        Vertex,
    };
    use std::mem::take;

    #[derive(Debug)]
    struct RegretInfoset<'a, 'b> {
        action_probs: &'a [f64],
        util: f64,
        prob_reach: f64,
        states: Vec<(f64, &'b Agent)>,
        num_fut_infosets: usize,
    }

    impl<'a, 'b> RegretInfoset<'a, 'b> {
        fn new(action_probs: &'a [f64]) -> RegretInfoset<'a, 'b> {
            RegretInfoset {
                action_probs,
                util: 0.0,
                prob_reach: 0.0,
                states: Vec::new(),
                num_fut_infosets: 0,
            }
        }
    }

    /// walk through the tree and update the probability of reaching a player's node and infoset given
    /// the actions of others
    fn recurse_reach<'a, 'b>(
        infosets: &mut [RegretInfoset<'a, 'b>],
        node: &'b Vertex,
        p_chance: f64,
        p_player: [f64; 2],
    ) {
        match node {
            Node::Terminal { .. } => {}
            Node::Chance(rand) => {
                for (prob, next) in rand.outcomes.iter() {
                    recurse_reach(infosets, next, p_chance * prob, p_player);
                }
            }
            Node::Player(agent) => {
                let info = &mut infosets[agent.infoset];
                let mut p_reach = p_chance;
                for (i, prob) in p_player.iter().enumerate() {
                    if i != agent.player.ind() {
                        p_reach *= prob;
                    }
                }
                info.prob_reach += p_reach;
                info.states.push((p_reach, agent));
                for (next, prob) in agent
                    .actions
                    .iter()
                    .zip(infosets[agent.infoset].action_probs)
                {
                    let mut next_probs = p_player;
                    next_probs[agent.player.ind()] *= prob;
                    recurse_reach(infosets, next, p_chance, next_probs);
                }
            }
        }
    }

    fn recurse_regret<'a, 'b>(
        infosets: &[RegretInfoset<'a, 'b>],
        node: &'b Vertex,
        target: Player,
    ) -> f64 {
        match node {
            Node::Terminal { player_one_payoff } => *player_one_payoff,
            Node::Chance(rand) => {
                let mut expected = 0.0;
                for (prob, next) in rand.outcomes.iter() {
                    expected += prob * recurse_regret(infosets, next, target);
                }
                expected
            }
            Node::Player(agent) if agent.player != target => {
                let mut expected = 0.0;
                let probs = &infosets[agent.infoset].action_probs;
                for (prob, next) in probs.iter().zip(agent.actions.iter()) {
                    expected += prob * recurse_regret(infosets, next, target);
                }
                expected
            }
            Node::Player(agent) => infosets[agent.infoset].util,
        }
    }

    fn recurse_expected<'a, 'b>(infosets: &[RegretInfoset<'a, 'b>], node: &'b Vertex) -> f64 {
        match node {
            Node::Terminal { player_one_payoff } => *player_one_payoff,
            Node::Chance(rand) => rand
                .outcomes
                .iter()
                .map(|(prob, next)| prob * recurse_expected(infosets, next))
                .sum(),
            Node::Player(agent) => {
                let probs = &infosets[agent.infoset].action_probs;
                probs
                    .iter()
                    .zip(agent.actions.iter())
                    .map(|(prob, next)| prob * recurse_expected(infosets, next))
                    .sum()
            }
        }
    }

    pub(super) fn regret(
        start: &Vertex,
        info: &[impl AgnosticInfoset],
        strategy: &[impl AsRef<[f64]>],
    ) -> Result<EquilibriumInfo, CompactError> {
        // NOTE in order to compute regret, we must compute regret for tail infosets before
        // computing regret for earlier ones. We do this by creating metadata for each infoset that
        // contains the states in it, and the number of future infosets. As we process infosets we
        // decrement that number making sure we only process an infoset after computing the maximum
        // utility for all future infosets.
        validate_strategy(info, strategy)?;

        let mut infosets = Vec::with_capacity(strategy.len());
        for (probs_trait, inf) in strategy.iter().zip(info.iter()) {
            let probs = probs_trait.as_ref();
            infosets.push(RegretInfoset::new(probs));
            // NOTE this works because previous infosets are guaranteed to have a lower index
            // than future ones
            if let Some(prev) = inf.prev_infoset() {
                infosets[prev].num_fut_infosets += 1;
            }
        }

        // add reach probability, and states to each infoset
        recurse_reach(&mut infosets, start, 1.0, [1.0; 2]);

        // iterate through in reverse breadth-first order by infoset for each player
        let mut queue: Vec<_> = infosets
            .iter()
            .enumerate()
            .filter(|(_, info)| info.num_fut_infosets == 0)
            .map(|(ind, _)| ind)
            .collect();
        while let Some(ind) = queue.pop() {
            let infoset_info = &info[ind];
            let info = &mut infosets[ind];
            let states = take(&mut info.states);
            let prob_reach = info.prob_reach;

            infosets[ind].util = if prob_reach == 0.0 {
                // NOTE if prob_reach is 0 here it means that due to the other agent's play we can
                // never reach this infoset.
                0.0
            } else {
                let mut utils = vec![0.0; infoset_info.num_actions()];
                for (prob, next) in states {
                    assert_eq!(next.actions.len(), infoset_info.num_actions());
                    for (i, exp) in next.actions.iter().enumerate() {
                        utils[i] += prob * recurse_regret(&infosets, exp, infoset_info.player())
                            / prob_reach;
                    }
                }
                let reduce = match infoset_info.player() {
                    Player::One => f64::max,
                    Player::Two => f64::min,
                };
                utils.iter().copied().reduce(reduce).unwrap()
            };

            if let Some(prev_info) = infoset_info.prev_infoset() {
                let prev = &mut infosets[prev_info];
                prev.num_fut_infosets -= 1;
                if prev.num_fut_infosets == 0 {
                    queue.push(prev_info);
                }
            }
        }

        let util_one = recurse_regret(&infosets, start, Player::One);
        let util_two = recurse_regret(&infosets, start, Player::Two);
        let utility = recurse_expected(&infosets, start);

        Ok(EquilibriumInfo {
            utility,
            player_one_regret: f64::max(0.0, util_one - utility),
            player_two_regret: f64::max(0.0, utility - util_two),
        })
    }
}

fn validate_strategy(
    info: &[impl AgnosticInfoset],
    strategy: &[impl AsRef<[f64]>],
) -> Result<(), CompactError> {
    if strategy.len() != info.len() {
        return Err(CompactError::InfosetNum {
            game_num: info.len(),
            strategy_num: strategy.len(),
        });
    }
    for (ind, (probs_trait, inf)) in strategy.iter().zip(info.iter()).enumerate() {
        let probs = probs_trait.as_ref();
        if probs.len() != inf.num_actions() {
            return Err(CompactError::ActionNum {
                ind,
                infoset_actions: inf.num_actions(),
                strategy_actions: probs.len(),
            });
        } else if (1.0 - probs.iter().sum::<f64>()).abs() > 1e-6 {
            return Err(CompactError::StratNotProbability {
                ind,
                total_prob: probs.iter().sum(),
            });
        }
    }
    Ok(())
}

/// A compact strategy for both players
///
/// Returned from [Game::solve].
#[derive(Debug, Clone)]
pub struct DenseStrategy(Box<[Box<[f64]>]>);

impl DenseStrategy {
    /// truncate a strategy where any probability below thresh is made zero
    pub fn truncate(&mut self, thresh: f64) {
        for strat in self.0.iter_mut() {
            let mut total = 0.0;
            for val in strat.iter_mut() {
                if *val < thresh {
                    *val = 0.0;
                } else {
                    total += *val;
                }
            }
            for val in strat.iter_mut() {
                *val /= total;
            }
        }
    }
}

// FIXME remove this?
impl Deref for DenseStrategy {
    type Target = [Box<[f64]>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// An approximate Nash equilibrium of a game
#[derive(Debug)]
pub struct Solution {
    /// An upper bound on the regret of the returned strategy
    pub regret: f64,
    /// A compact representation of a set of player strategies
    ///
    /// Use [Game::name_strategy] to convert it to a named version referencing information sets and
    /// actions.
    pub strategy: DenseStrategy,
}

/// Information about an approximate equilibrium
#[derive(Debug, PartialEq)]
pub struct EquilibriumInfo {
    /// The expected utility of player one under the current strategies
    pub utility: f64,
    /// The regret of player one
    pub player_one_regret: f64,
    /// The regret of player two
    pub player_two_regret: f64,
}

impl EquilibriumInfo {
    /// The regret of this equilibrium independent of player
    pub fn regret(&self) -> f64 {
        f64::max(self.player_one_regret, self.player_two_regret)
    }
}

/// FIXME
type NamedStrategies<'a, I, A, R> = (
    NamedStrategyIter<'a, I, A, R>,
    NamedStrategyIter<'a, I, A, R>,
);

/// An iterator over named information sets of a strategy.
///
/// This is returned when converting a [DenseStrategy] to a named strategy with
/// [Game::name_strategy].
#[derive(Debug)]
pub struct NamedStrategyIter<'a, I, A, R> {
    player: Player,
    info_strats: Zip<slice::Iter<'a, Infoset<I, A>>, slice::Iter<'a, R>>,
}

impl<'a, I, A, R> NamedStrategyIter<'a, I, A, R> {
    fn new(
        player: Player,
        info: &'a [Infoset<I, A>],
        strategy: &'a [R],
    ) -> NamedStrategyIter<'a, I, A, R> {
        NamedStrategyIter {
            player,
            info_strats: info.iter().zip(strategy.iter()),
        }
    }
}

impl<'a, I, A, R> Iterator for NamedStrategyIter<'a, I, A, R>
where
    R: AsRef<[f64]>,
{
    type Item = (&'a I, NamedStrategyActionIter<'a, A>);

    fn next(&mut self) -> Option<Self::Item> {
        self.info_strats
            .find(|(info, _)| info.player == self.player)
            .map(|(info, strat)| {
                (
                    &info.infoset,
                    NamedStrategyActionIter::new(&info.actions, strat.as_ref()),
                )
            })
    }
}

/// An iterator over name actions assiciated probabilities
///
/// This is returned when converting a [DenseStrategy] to a named strategy with
/// [Game::name_strategy].
#[derive(Debug)]
pub struct NamedStrategyActionIter<'a, A>(Zip<slice::Iter<'a, A>, slice::Iter<'a, f64>>);

impl<'a, A> NamedStrategyActionIter<'a, A> {
    fn new(actions: &'a [A], strategy: &'a [f64]) -> NamedStrategyActionIter<'a, A> {
        NamedStrategyActionIter(actions.iter().zip(strategy.iter()))
    }
}

impl<'a, A> Iterator for NamedStrategyActionIter<'a, A> {
    type Item = (&'a A, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        let NamedStrategyActionIter(iter) = self;
        iter.next()
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    struct Term;

    impl Term {
        fn new_node() -> SimpNode {
            SimpNode::Terminal {
                player_one_payoff: 0.0,
            }
        }
    }

    struct Rand(Vec<(f64, SimpNode)>);

    impl Rand {
        fn new_node(iter: impl IntoIterator<Item = (f64, SimpNode)>) -> SimpNode {
            SimpNode::Chance(Rand(iter.into_iter().collect()))
        }
    }

    impl ChanceNode for Rand {
        type PlayerNode = SimpPlayer;
        type Outcomes = Vec<(f64, SimpNode)>;

        fn into_outcomes(self) -> Self::Outcomes {
            self.0
        }
    }

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    enum Info {
        X,
        Y,
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
    enum Action {
        A,
        B,
        C,
    }

    struct SimpPlayer {
        player: Player,
        info: Info,
        actions: Vec<(Action, SimpNode)>,
    }

    impl SimpPlayer {
        fn new_node(
            player: Player,
            iter: impl IntoIterator<Item = (Action, SimpNode)>,
        ) -> SimpNode {
            SimpNode::Player(SimpPlayer {
                player,
                info: Info::X,
                actions: iter.into_iter().collect(),
            })
        }

        fn new_node_with_info(
            player: Player,
            info: Info,
            iter: impl IntoIterator<Item = (Action, SimpNode)>,
        ) -> SimpNode {
            SimpNode::Player(SimpPlayer {
                player,
                info,
                actions: iter.into_iter().collect(),
            })
        }
    }

    impl PlayerNode for SimpPlayer {
        type Infoset = Info;
        type Action = Action;
        type ChanceNode = Rand;
        type Actions = Vec<(Action, SimpNode)>;

        fn get_player(&self) -> Player {
            self.player
        }

        fn get_infoset(&self) -> Info {
            self.info
        }

        fn into_actions(self) -> Self::Actions {
            self.actions
        }
    }

    type SimpNode = Node<Rand, SimpPlayer>;

    #[test]
    fn empty_chance() {
        let err_game = Rand::new_node([]);
        let err = Game::from_node(err_game).unwrap_err();
        assert_eq!(err, GameError::EmptyChance);
    }

    #[test]
    fn non_positive_chance() {
        let err_game = Rand::new_node([(0.0, Term::new_node())]);
        let err = Game::from_node(err_game).unwrap_err();
        assert_eq!(err, GameError::NonPositiveChance { prob: 0.0 });
    }

    #[test]
    fn degenerate_chance() {
        let err_game = Rand::new_node([(1.0, Term::new_node())]);
        let err = Game::from_node(err_game).unwrap_err();
        assert_eq!(err, GameError::DegenerateChance);
    }

    #[test]
    fn empty_player() {
        let err_game = SimpPlayer::new_node(Player::One, []);
        let err = Game::from_node(err_game).unwrap_err();
        assert_eq!(err, GameError::EmptyPlayer);
    }

    #[test]
    fn degenerate_player() {
        let err_game = SimpPlayer::new_node(Player::One, [(Action::A, Term::new_node())]);
        let err = Game::from_node(err_game).unwrap_err();
        assert_eq!(err, GameError::DegeneratePlayer);
    }

    #[test]
    fn imperfect_recall() {
        let err_game = Rand::new_node([
            (
                0.5,
                SimpPlayer::new_node_with_info(
                    Player::One,
                    Info::X,
                    [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
                ),
            ),
            (
                0.5,
                SimpPlayer::new_node_with_info(
                    Player::One,
                    Info::Y,
                    [
                        (
                            Action::A,
                            SimpPlayer::new_node_with_info(
                                Player::One,
                                Info::X,
                                [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
                            ),
                        ),
                        (Action::B, Term::new_node()),
                    ],
                ),
            ),
        ]);
        let err = Game::from_node(err_game).unwrap_err();
        assert_eq!(
            err,
            GameError::ImperfectRecall {
                player: Player::One,
                infoset: Info::X,
            }
        );
    }

    #[test]
    fn actions_not_equal() {
        let ord_game = Rand::new_node([
            (
                0.5,
                SimpPlayer::new_node(
                    Player::One,
                    [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
                ),
            ),
            (
                0.5,
                SimpPlayer::new_node(
                    Player::One,
                    [(Action::B, Term::new_node()), (Action::A, Term::new_node())],
                ),
            ),
        ]);
        let err = Game::from_node(ord_game).unwrap_err();
        assert_eq!(
            err,
            GameError::ActionsNotEqual {
                player: Player::One,
                infoset: Info::X,
                actions: vec![Action::B, Action::A]
            }
        );
    }

    #[test]
    fn actions_not_unique() {
        let dup_game = SimpPlayer::new_node(
            Player::One,
            [(Action::A, Term::new_node()), (Action::A, Term::new_node())],
        );
        let err = Game::from_node(dup_game).unwrap_err();
        assert_eq!(
            err,
            GameError::ActionsNotUnique {
                player: Player::One,
                infoset: Info::X,
                actions: vec![Action::A, Action::A]
            }
        );
    }

    #[test]
    fn incorrect_infosets() {
        let base_game = SimpPlayer::new_node(
            Player::One,
            [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
        );
        let game = Game::from_node(base_game).unwrap();

        let info_err = game.regret(&[] as &[&[f64]]).unwrap_err();
        assert_eq!(
            info_err,
            CompactError::InfosetNum {
                game_num: 1,
                strategy_num: 0
            }
        );

        let action_err = game.regret(&[[1.0]]).unwrap_err();
        assert_eq!(
            action_err,
            CompactError::ActionNum {
                ind: 0,
                infoset_actions: 2,
                strategy_actions: 1
            }
        );

        let strat_err = game.regret(&[[1.0, 1.0]]).unwrap_err();
        assert_eq!(
            strat_err,
            CompactError::StratNotProbability {
                ind: 0,
                total_prob: 2.0
            }
        );

        assert_eq!(
            game.regret(&[[0.5, 0.5]]).unwrap(),
            EquilibriumInfo {
                utility: 0.0,
                player_one_regret: 0.0,
                player_two_regret: 0.0
            }
        );
    }

    #[test]
    fn invalid_info_action() {
        let base_game = SimpPlayer::new_node(
            Player::One,
            [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
        );
        let game = Game::from_node(base_game).unwrap();

        let empty: [(Info, Action, f64); 0] = [];

        let info_err = game
            .compact_strategy([(Info::Y, Action::A, 0.5)], empty)
            .unwrap_err();
        assert_eq!(
            info_err,
            StratError::InvalidInfosetAction {
                player: Player::One,
                infoset: Info::Y,
                action: Action::A,
            }
        );

        let action_err = game
            .compact_strategy([(Info::X, Action::C, 0.5)], empty)
            .unwrap_err();
        assert_eq!(
            action_err,
            StratError::InvalidInfosetAction {
                player: Player::One,
                infoset: Info::X,
                action: Action::C,
            }
        );
    }

    #[test]
    fn invalid_prob() {
        let base_game = SimpPlayer::new_node(
            Player::One,
            [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
        );
        let game = Game::from_node(base_game).unwrap();

        let empty: [(Info, Action, f64); 0] = [];

        let prob_err = game
            .compact_strategy([(Info::X, Action::A, 1.5)], empty)
            .unwrap_err();
        assert_eq!(
            prob_err,
            StratError::InvalidProbability {
                player: Player::One,
                infoset: Info::X,
                action: Action::A,
                prob: 1.5,
            }
        );
    }

    #[test]
    fn incorrect_strategies() {
        let base_game = SimpPlayer::new_node(
            Player::One,
            [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
        );
        let game = Game::from_node(base_game).unwrap();

        let empty: [(Info, Action, f64); 0] = [];

        let total_err = game.compact_strategy(empty, empty).unwrap_err();
        assert_eq!(
            total_err,
            StratError::InvalidInfosetProbability {
                player: Player::One,
                infoset: Info::X,
                total_prob: 0.0
            }
        );
    }

    #[test]
    fn zero_reach_regret() {
        let base_game = SimpPlayer::new_node(
            Player::One,
            [
                (
                    Action::A,
                    SimpPlayer::new_node_with_info(
                        Player::Two,
                        Info::X,
                        [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
                    ),
                ),
                (
                    Action::B,
                    SimpPlayer::new_node_with_info(
                        Player::Two,
                        Info::Y,
                        [(Action::A, Term::new_node()), (Action::B, Term::new_node())],
                    ),
                ),
            ],
        );
        let game = Game::from_node(base_game).unwrap();
        let one = [(Info::X, Action::A, 1.0)];
        let two = [
            (Info::X, Action::A, 0.5),
            (Info::X, Action::B, 0.5),
            (Info::Y, Action::A, 0.5),
            (Info::Y, Action::B, 0.5),
        ];
        let strat = game.compact_strategy(one, two).unwrap();

        assert_eq!(
            game.regret(&strat).unwrap(),
            EquilibriumInfo {
                utility: 0.0,
                player_one_regret: 0.0,
                player_two_regret: 0.0
            }
        );
    }
}

#[cfg(test)]
mod akq_tests {
    use super::*;

    enum Pot {
        WonAnte,
        LostAnte,
        WonRaise,
        LostRaise,
    }

    impl Pot {
        fn node(self: Pot) -> AkqNode {
            AkqNode::Terminal {
                player_one_payoff: match self {
                    Pot::WonAnte => 1.0,
                    Pot::LostAnte => -1.0,
                    Pot::WonRaise => 2.0,
                    Pot::LostRaise => -2.0,
                },
            }
        }
    }

    struct Deal(Vec<AkqNode>);

    impl Deal {
        fn new_node(iter: impl IntoIterator<Item = AkqNode>) -> AkqNode {
            AkqNode::Chance(Deal(iter.into_iter().collect()))
        }
    }

    struct DealIter(f64, vec::IntoIter<AkqNode>);

    impl Iterator for DealIter {
        type Item = (f64, AkqNode);

        fn next(&mut self) -> Option<Self::Item> {
            let DealIter(frac, iter) = self;
            iter.next().map(|node| (*frac, node))
        }
    }

    impl ChanceNode for Deal {
        type PlayerNode = AkqPlayer;
        type Outcomes = DealIter;

        fn into_outcomes(self) -> Self::Outcomes {
            let Deal(vec) = self;
            DealIter(1.0 / vec.len() as f64, vec.into_iter())
        }
    }

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    enum Card {
        A,
        K,
        Q,
    }

    #[derive(Debug, Hash, PartialEq, Eq)]
    enum Action {
        Fold,
        Call,
        Raise,
    }

    struct AkqPlayer {
        player: Player,
        infoset: Card,
        actions: Vec<(Action, AkqNode)>,
    }

    impl AkqPlayer {
        fn new_node(
            player: Player,
            infoset: Card,
            iter: impl IntoIterator<Item = (Action, AkqNode)>,
        ) -> AkqNode {
            AkqNode::Player(AkqPlayer {
                player,
                infoset,
                actions: iter.into_iter().collect(),
            })
        }
    }

    impl PlayerNode for AkqPlayer {
        type Infoset = Card;
        type Action = Action;
        type ChanceNode = Deal;
        type Actions = Vec<(Action, AkqNode)>;

        fn get_player(&self) -> Player {
            self.player
        }

        fn get_infoset(&self) -> Card {
            self.infoset
        }

        fn into_actions(self) -> Self::Actions {
            self.actions
        }
    }

    type AkqNode = Node<Deal, AkqPlayer>;

    fn create_akq() -> Game<Card, Action> {
        // verify it runs
        let aqk_poker = Deal::new_node([
            AkqPlayer::new_node(
                Player::One,
                Card::Q,
                [
                    (Action::Call, Pot::LostAnte.node()),
                    (
                        Action::Raise,
                        Deal::new_node([
                            AkqPlayer::new_node(
                                Player::Two,
                                Card::K,
                                [
                                    (Action::Fold, Pot::WonAnte.node()),
                                    (Action::Call, Pot::LostRaise.node()),
                                ],
                            ),
                            AkqPlayer::new_node(
                                Player::Two,
                                Card::A,
                                [
                                    (Action::Fold, Pot::WonAnte.node()),
                                    (Action::Call, Pot::LostRaise.node()),
                                ],
                            ),
                        ]),
                    ),
                ],
            ),
            AkqPlayer::new_node(
                Player::One,
                Card::K,
                [
                    (
                        Action::Call,
                        Deal::new_node([Pot::WonAnte.node(), Pot::LostAnte.node()]),
                    ),
                    (
                        Action::Raise,
                        Deal::new_node([
                            AkqPlayer::new_node(
                                Player::Two,
                                Card::Q,
                                [
                                    (Action::Fold, Pot::WonAnte.node()),
                                    (Action::Call, Pot::WonRaise.node()),
                                ],
                            ),
                            AkqPlayer::new_node(
                                Player::Two,
                                Card::A,
                                [
                                    (Action::Fold, Pot::WonAnte.node()),
                                    (Action::Call, Pot::LostRaise.node()),
                                ],
                            ),
                        ]),
                    ),
                ],
            ),
            AkqPlayer::new_node(
                Player::One,
                Card::A,
                [
                    (Action::Call, Pot::WonAnte.node()),
                    (
                        Action::Raise,
                        Deal::new_node([
                            AkqPlayer::new_node(
                                Player::Two,
                                Card::Q,
                                [
                                    (Action::Fold, Pot::WonAnte.node()),
                                    (Action::Call, Pot::WonRaise.node()),
                                ],
                            ),
                            AkqPlayer::new_node(
                                Player::Two,
                                Card::K,
                                [
                                    (Action::Fold, Pot::WonAnte.node()),
                                    (Action::Call, Pot::WonRaise.node()),
                                ],
                            ),
                        ]),
                    ),
                ],
            ),
        ]);
        Game::from_node(aqk_poker).unwrap()
    }

    #[test]
    fn solve_full() {
        let game = create_akq();
        let Solution { regret, strategy } = game.solve_full(100000, 0.001);
        assert!(regret < 0.001);

        // measure regret manually
        let reg = game.regret(&strategy).unwrap().regret();
        assert!(reg <= regret);

        // in this game, pruning should help, so we try pruning
        let mut pruned_strat = strategy.clone();
        pruned_strat.truncate(1e-3);
        let pruned_reg = game.regret(&pruned_strat).unwrap().regret();
        // NOTE given the strategies, there's no formal guarantee that pruning will reduce regret
        // if it exploits the opponents non-pruned strategy more
        assert!(pruned_reg <= regret * 2.0);

        // verify we get the right named strategy
        let (one, two) = game.name_strategy(&strategy).unwrap();

        for (info, strat) in one {
            match info {
                Card::Q => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => panic!(),
                            Action::Call => assert!((2.0 / 3.0 - prob).abs() < 0.01),
                            Action::Raise => assert!((1.0 / 3.0 - prob).abs() < 0.01),
                        }
                    }
                }
                Card::K => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => panic!(),
                            Action::Call => assert!((1.0 - prob).abs() < 1e-3),
                            Action::Raise => assert!((0.0 - prob).abs() < 1e-3),
                        }
                    }
                }
                Card::A => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => panic!(),
                            Action::Call => assert!((0.0 - prob).abs() < 1e-3),
                            Action::Raise => assert!((1.0 - prob).abs() < 1e-3),
                        }
                    }
                }
            }
        }

        for (info, strat) in two {
            match info {
                Card::Q => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => assert!((1.0 - prob).abs() < 1e-3),
                            Action::Call => assert!((0.0 - prob).abs() < 1e-3),
                            Action::Raise => panic!(),
                        }
                    }
                }
                Card::K => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => assert!((2.0 / 3.0 - prob).abs() < 0.01),
                            Action::Call => assert!((1.0 / 3.0 - prob).abs() < 0.01),
                            Action::Raise => panic!(),
                        }
                    }
                }
                Card::A => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => assert!((0.0 - prob).abs() < 1e-3),
                            Action::Call => assert!((1.0 - prob).abs() < 1e-3),
                            Action::Raise => panic!(),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn solve_external() {
        let game = create_akq();
        let Solution { regret, strategy } = game.solve_external(u64::MAX, 1000, 0.001);
        assert!(regret < 0.001);

        // measure regret manually
        let reg = game.regret(&strategy).unwrap().regret();
        assert!(reg <= regret);

        // in this game, pruning should help, so we try pruning
        let mut pruned_strat = strategy.clone();
        pruned_strat.truncate(1e-3);
        let pruned_reg = game.regret(&pruned_strat).unwrap().regret();
        // NOTE given the strategies, there's no formal guarantee that pruning will reduce regret
        // if it exploits the opponents non-pruned strategy more
        assert!(pruned_reg <= regret * 2.0);

        // verify we get the right named strategy
        let (one, two) = game.name_strategy(&strategy).unwrap();

        for (info, strat) in one {
            match info {
                Card::Q => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => panic!(),
                            Action::Call => assert!((2.0 / 3.0 - prob).abs() < 0.01),
                            Action::Raise => assert!((1.0 / 3.0 - prob).abs() < 0.01),
                        }
                    }
                }
                Card::K => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => panic!(),
                            Action::Call => assert!((1.0 - prob).abs() < 0.01),
                            Action::Raise => assert!((0.0 - prob).abs() < 0.01),
                        }
                    }
                }
                Card::A => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => panic!(),
                            Action::Call => assert!((0.0 - prob).abs() < 0.01),
                            Action::Raise => assert!((1.0 - prob).abs() < 0.01),
                        }
                    }
                }
            }
        }

        for (info, strat) in two {
            match info {
                Card::Q => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => assert!((1.0 - prob).abs() < 0.01),
                            Action::Call => assert!((0.0 - prob).abs() < 0.01),
                            Action::Raise => panic!(),
                        }
                    }
                }
                Card::K => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => assert!((2.0 / 3.0 - prob).abs() < 0.01),
                            Action::Call => assert!((1.0 / 3.0 - prob).abs() < 0.01),
                            Action::Raise => panic!(),
                        }
                    }
                }
                Card::A => {
                    for (act, prob) in strat {
                        match act {
                            Action::Fold => assert!((0.0 - prob).abs() < 0.01),
                            Action::Call => assert!((1.0 - prob).abs() < 0.01),
                            Action::Raise => panic!(),
                        }
                    }
                }
            }
        }
    }
}
