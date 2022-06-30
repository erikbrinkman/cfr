//! Counterfactual Regret (CFR) is a library for finding an approximate nash equilibrium in
//! two-player zero-sum games of incomplete information, such as poker etc., using counterfactual
//! regret minimization[^cfr].
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
//! # use cfr::{TerminalNode, Node, PlayerNode, ChanceNode, Player, GameError};
//! # use std::iter::{Empty, empty};
//! # struct Term;
//! # impl TerminalNode for Term {
//! #     fn get_one_payoff(&self) -> f64 { 0.0 }
//! # }
//! # struct Chance;
//! # impl IntoIterator for Chance {
//! #     type Item = (f64, ExNode);
//! #     type IntoIter = Empty<Self::Item>;
//! #     fn into_iter(self) -> Self::IntoIter { empty() }
//! # }
//! # impl ChanceNode<Term, Play> for Chance {}
//! # struct Play;
//! # impl IntoIterator for Play {
//! #     type Item = (usize, ExNode);
//! #     type IntoIter = Empty<Self::Item>;
//! #     fn into_iter(self) -> Self::IntoIter { empty() }
//! # }
//! # impl PlayerNode<usize, usize, Term, Chance> for Play {
//! #     fn get_player(&self) -> Player { Player::One }
//! #     fn get_infoset(&self) -> usize { 0 }
//! # }
//! # type ExNode = Node<Term, Chance, Play>;
//! # let my_game = ExNode::Terminal(Term);
//! // my_game = ...
//! let game = Game::from_node(my_game).unwrap();
//! let Solution { regret, strategy } = game.solve(1000, 0.1);
//! let tighter_regret = game.regret(&strategy).unwrap();
//! let named_strategy = game.name_strategy(&strategy);
//! ```
//!
//! [^cfr]: [Zinkevich, Martin, et al. "Regret minimization in games with incomplete information."
//!   Advances in neural information processing systems 20
//!   (2007)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).
#![warn(missing_docs)]
use core::slice;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::Zip;
use std::ops::Deref;
use std::vec;

// TODO think about better interface for jagged array

// ------ //
// Public //
// ------ //

/// Errors that result from game definition errors
///
/// If the object passed into [Game::from_node] doesn't conform to necessary
/// invariants, one of these will be returned.
#[derive(Debug, PartialEq)]
pub enum GameError<I, A> {
    /// Returned when a chance node's probability did not sum to one.
    ///
    /// Try making sure that all chance node probabilities sum to one.
    ChanceNotProbability {
        /// The probability that the node actually summed to.
        total_prob: f64,
    },
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

/// Errors that result from incompatible compact strategy representations
///
/// If a strategy object passed to a [Game] doesn't match the games information and action
/// structure (notably [Game::regret]), one of these errors will be returned.
#[derive(Debug, PartialEq)]
pub enum StratError {
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

/// An enum indicating which player a state in the tree belongs to
#[derive(Debug, Copy, Eq, Clone, PartialEq, Hash)]
pub enum Player {
    /// The first player
    One,
    /// The second player
    Two,
}

/// An enum for distinguishing between game node types
///
/// A game node can be one of a terminal node with the payoff for player one, a chance node with a
/// fixed probability of advancing, or a player node indicating a place where a player can make a
/// decision.
#[derive(Debug)]
pub enum Node<T, C, P> {
    /// A terminal node, the game is over
    Terminal(T),
    /// A chance node, the game advances independent of player action
    Chance(C),
    /// a node in the tree where the player can choose between different actions
    Player(P),
}

/// An arbitrary terminal node with a payoff for player one
pub trait TerminalNode {
    /// Get the payoff to player one
    ///
    /// Since this library only models zero-sum games, player two inherently has the negative of
    /// player one's payoff.
    fn get_one_payoff(&self) -> f64;
}

/// A node with several outcomes, each which has a different probability of outcome
///
/// This node represents randomness in the game independent of player action. The total of all
/// probabilities must sum to one.
pub trait ChanceNode<T, P>: IntoIterator<Item = (f64, Node<T, Self, P>)> + Sized {}

/// A node indicating a player decision point in the game
///
/// Nodes are iterables over an action the player can choose and future states. In addition, player
/// nodes must have an assigned player who's acting, and a marker for the information set
/// indicating what the player knows at this stage of the game.
pub trait PlayerNode<I, A, T, C>: IntoIterator<Item = (A, Node<T, C, Self>)> + Sized {
    /// Get the player who's acting at this node in the game tree
    fn get_player(&self) -> Player;

    /// Get the information set of this game node
    ///
    /// An information set serves as a comapct representation of the information that a player has
    /// available when making a decision. A payer can not distinguish (and thus must play the same
    /// strategy) at each node that has the same information set.
    fn get_infoset(&self) -> I;
}

// --------- //
// Internals //
// --------- //

#[derive(Debug)]
struct Terminal(f64);

#[derive(Debug)]
struct Random(Box<[(f64, Vertex)]>);

#[derive(Debug)]
struct Agent {
    player: Player,
    infoset: usize,
    actions: Box<[Vertex]>,
}

type Vertex = Node<Terminal, Random, Agent>;

#[derive(Debug)]
struct CumInfoset {
    // TODO this could be updated to have one buffer of heap memory that we chunk since we know the
    // chunks will have the same size.
    cum_regret: Box<[f64]>,
    utils: Box<[f64]>,
    probs: Box<[f64]>,
    cum_probs: Box<[f64]>,
    prob_reach: f64,
    cum_prob_reach: f64,
}

impl CumInfoset {
    fn new(num_actions: usize) -> CumInfoset {
        CumInfoset {
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
        let mut expected_util = 0.0;
        for (util, prob) in self.utils.iter().zip(self.probs.iter()) {
            expected_util += util * prob
        }
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

impl Infoset for CumInfoset {
    fn prob_reach_mut(&mut self) -> &mut f64 {
        &mut self.prob_reach
    }

    fn action_probs(&self) -> &[f64] {
        &self.probs
    }

    fn utils_mut(&mut self) -> &mut [f64] {
        &mut self.utils
    }
}

struct StratInfoset<'a> {
    probs: &'a [f64],
    utils: Box<[f64]>,
    // NOTE this is need for the trait, but is unused here. The current thinking is that the
    // removal of duplicate code is worth this slight memory and time regression
    prob_reach: f64,
}

impl<'a> StratInfoset<'a> {
    fn new(probs: &'a [f64]) -> StratInfoset<'a> {
        StratInfoset {
            probs,
            utils: vec![0.0; probs.len()].into_boxed_slice(),
            prob_reach: 0.0,
        }
    }

    fn regret(&self) -> f64 {
        // compute expected utility
        let mut expected_util = 0.0;
        for (util, prob) in self.utils.iter().zip(self.probs.iter()) {
            expected_util += util * prob
        }
        // use expected utility to update cumulative average regret
        let mut reg = 0.0;
        for util in self.utils.iter() {
            reg = f64::max(reg, util - expected_util);
        }
        reg
    }
}

impl<'a> Infoset for StratInfoset<'a> {
    fn prob_reach_mut(&mut self) -> &mut f64 {
        &mut self.prob_reach
    }

    fn action_probs(&self) -> &[f64] {
        &self.probs
    }

    fn utils_mut(&mut self) -> &mut [f64] {
        &mut self.utils
    }
}

#[derive(Debug)]
struct InfosetInfo<I, A> {
    player: Player,
    infoset: I,
    actions: Box<[A]>,
}

/// A compact game representation that includes strategies for finding approximate nash equilibria
/// and computing regret of strategies.
#[derive(Debug)]
pub struct Game<I, A> {
    infosets: Box<[InfosetInfo<I, A>]>,
    start: Vertex,
}

impl<I, A> Game<I, A> {
    /// Create a game from the root node of an arbitrary game tree
    pub fn from_node<T, C, P>(start: Node<T, C, P>) -> Result<Game<I, A>, GameError<I, A>>
    where
        I: Hash + Eq + Clone,
        A: Hash + Eq,
        T: TerminalNode,
        C: ChanceNode<T, P>,
        P: PlayerNode<I, A, T, C>,
    {
        let mut infosets = Vec::new();
        let mut inds = HashMap::<(Player, I), usize>::new();
        let start = recursive_vertex_from_node(&mut infosets, &mut inds, start)?;

        Ok(Game {
            infosets: infosets.into_boxed_slice(),
            start,
        })
    }

    /// Find an approximate Nash equilibrium of the current game
    ///
    /// This will run no more than `max_iter` iterations, and terminate early if it can guarantee
    /// that it's found a solution with regret smaller than `max_reg`.
    pub fn solve(&self, max_iter: u64, max_reg: f64) -> Solution {
        let mut infosets: Vec<_> = self
            .infosets
            .iter()
            .map(|info| CumInfoset::new(info.actions.len()))
            .collect();
        let mut it = 1;
        let mut reg = max_reg;
        while it != max_iter && reg >= max_reg {
            // go through self and update utilities and probability of reaching a node
            recurse_util(&mut infosets, &self.start, 1.0, 1.0, 1.0);

            // update infoset information for next iteration
            let mut one_reg = 0.0;
            let mut two_reg = 0.0;
            for (infoset, info) in infosets.iter_mut().zip(self.infosets.iter()) {
                *(match info.player {
                    Player::One => &mut one_reg,
                    Player::Two => &mut two_reg,
                }) += infoset.update(it);
            }
            reg = 2.0 * f64::max(one_reg, two_reg);
            it += 1;
        }
        let strat: Vec<_> = infosets.into_iter().map(|info| info.cum_probs).collect();
        Solution {
            regret: reg,
            strategy: CompactStrategy(strat.into_boxed_slice()),
        }
    }

    /// Compute the regret of a strategy like object
    pub fn regret(&self, strategy: &[impl AsRef<[f64]>]) -> Result<EquilibriumInfo, StratError> {
        if strategy.len() != self.infosets.len() {
            return Err(StratError::InfosetNum {
                game_num: self.infosets.len(),
                strategy_num: strategy.len(),
            });
        }
        let mut infosets = Vec::with_capacity(strategy.len());
        for (ind, (probs_trait, info)) in strategy.iter().zip(self.infosets.iter()).enumerate() {
            let probs = probs_trait.as_ref();
            if probs.len() != info.actions.len() {
                Err(StratError::ActionNum {
                    ind,
                    infoset_actions: info.actions.len(),
                    strategy_actions: probs.len(),
                })
            } else if (1.0 - probs.iter().sum::<f64>()).abs() > 1e-6 {
                Err(StratError::StratNotProbability {
                    ind,
                    total_prob: probs.iter().sum(),
                })
            } else {
                infosets.push(StratInfoset::new(probs));
                Ok(())
            }?;
        }
        let utility = recurse_util(&mut infosets, &self.start, 1.0, 1.0, 1.0);

        // add up regrets by infoset
        let mut player_one_regret = 0.0;
        let mut player_two_regret = 0.0;
        for (infoset, info) in infosets.iter_mut().zip(self.infosets.iter()) {
            *(match info.player {
                Player::One => &mut player_one_regret,
                Player::Two => &mut player_two_regret,
            }) += infoset.regret();
        }
        Ok(EquilibriumInfo {
            utility,
            player_one_regret,
            player_two_regret,
        })
    }

    /// attach player, infoset, and action information to a strategy
    pub fn name_strategy<'a, R>(
        &'a self,
        strategy: &'a [R],
    ) -> (
        NamedStrategyIter<'a, I, A, R>,
        NamedStrategyIter<'a, I, A, R>,
    )
    where
        R: AsRef<[f64]>,
    {
        (
            NamedStrategyIter::new(Player::One, &self.infosets, strategy),
            NamedStrategyIter::new(Player::Two, &self.infosets, strategy),
        )
    }
}

fn recursive_vertex_from_node<I, A, T, C, P>(
    infosets: &mut Vec<InfosetInfo<I, A>>,
    inds: &mut HashMap<(Player, I), usize>,
    node: Node<T, C, P>,
) -> Result<Vertex, GameError<I, A>>
where
    I: Hash + Eq + Clone,
    A: Hash + Eq,
    T: TerminalNode,
    C: ChanceNode<T, P>,
    P: PlayerNode<I, A, T, C>,
{
    match node {
        Node::Terminal(term) => Ok(Vertex::Terminal(Terminal(term.get_one_payoff()))),
        Node::Chance(chance) => {
            let mut total = 0.0;
            let mut outcomes = Vec::new();
            for (prob, next) in chance {
                let next_vert = recursive_vertex_from_node(infosets, inds, next)?;
                outcomes.push((prob, next_vert));
                total += prob;
            }
            if (total - 1.0).abs() < 1e-6 {
                // renormalize to make sure consistency
                for (prob, _) in &mut outcomes {
                    *prob /= total;
                }

                Ok(Vertex::Chance(Random(outcomes.into_boxed_slice())))
            } else {
                Err(GameError::ChanceNotProbability { total_prob: total })
            }
        }
        Node::Player(player) => {
            let play = player.get_player();
            let infoset = player.get_infoset();
            let mut names = Vec::new();
            let mut actions = Vec::new();
            for (action, next) in player {
                let next_vert = recursive_vertex_from_node(infosets, inds, next)?;
                names.push(action);
                actions.push(next_vert);
            }
            let info_ind = match inds.entry((play, infoset.clone())) {
                Entry::Occupied(ent) => {
                    let info = ent.get();
                    let existing_names = &infosets[*info].actions;
                    if **existing_names == *names {
                        Ok(*info)
                    } else {
                        Err(GameError::ActionsNotEqual {
                            player: play,
                            infoset,
                            actions: names,
                        })
                    }
                }
                Entry::Vacant(ent) => {
                    let hash_names: HashSet<&A> = names.iter().collect();
                    if hash_names.len() == names.len() {
                        let info = infosets.len();
                        infosets.push(InfosetInfo {
                            player: play,
                            infoset,
                            actions: names.into_boxed_slice(),
                        });
                        ent.insert(info);
                        Ok(info)
                    } else {
                        Err(GameError::ActionsNotUnique {
                            player: play,
                            infoset,
                            actions: names,
                        })
                    }
                }
            }?;
            Ok(Vertex::Player(Agent {
                player: play,
                infoset: info_ind,
                actions: actions.into_boxed_slice(),
            }))
        }
    }
}

trait Infoset {
    fn prob_reach_mut(&mut self) -> &mut f64;

    fn action_probs(&self) -> &[f64];

    fn utils_mut(&mut self) -> &mut [f64];
}

fn recurse_util(
    infosets: &mut [impl Infoset],
    node: &Vertex,
    p_chance: f64,
    p_one: f64,
    p_two: f64,
) -> f64 {
    match node {
        Node::Terminal(Terminal(payoff)) => *payoff,
        Node::Chance(Random(outcomes)) => {
            let mut expected = 0.0;
            for (prob, next) in outcomes.iter() {
                expected += prob * recurse_util(infosets, next, p_chance * prob, p_one, p_two);
            }
            expected
        }
        Node::Player(agent) => {
            *infosets[agent.infoset].prob_reach_mut() += p_chance * p_one * p_two;
            let mult = match agent.player {
                Player::One => p_chance * p_two,
                Player::Two => -p_chance * p_one,
            };
            let mut expected = 0.0;
            for (i, next) in agent.actions.iter().enumerate() {
                let prob = infosets[agent.infoset].action_probs()[i];
                let (p_one_next, p_two_next) = match agent.player {
                    Player::One => (prob * p_one, p_two),
                    Player::Two => (p_one, prob * p_two),
                };
                let util = recurse_util(infosets, next, p_chance, p_one_next, p_two_next);
                infosets[agent.infoset].utils_mut()[i] += mult * util;
                expected += prob * util;
            }
            expected
        }
    }
}

/// A compact strategy for both players
///
/// Returned from [Game::solve].
#[derive(Debug)]
pub struct CompactStrategy(Box<[Box<[f64]>]>);

impl Deref for CompactStrategy {
    type Target = [Box<[f64]>];

    fn deref(&self) -> &Self::Target {
        let CompactStrategy(bx) = self;
        &bx
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
    pub strategy: CompactStrategy,
}

/// Information about an approximate equilibrium
#[derive(Debug)]
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

/// An iterator over named information sets of a strategy.
///
/// This is returned when converting a [CompactStrategy] to a named strategy with
/// [Game::name_strategy].
#[derive(Debug)]
pub struct NamedStrategyIter<'a, I, A, R> {
    player: Player,
    info_strats: Zip<slice::Iter<'a, InfosetInfo<I, A>>, slice::Iter<'a, R>>,
}

impl<'a, I, A, R> NamedStrategyIter<'a, I, A, R> {
    fn new(
        player: Player,
        info: &'a [InfosetInfo<I, A>],
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
/// This is returned when converting a [CompactStrategy] to a named strategy with
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
            SimpNode::Terminal(Term)
        }
    }

    impl TerminalNode for Term {
        fn get_one_payoff(&self) -> f64 {
            0.0
        }
    }

    struct Rand(Vec<(f64, SimpNode)>);

    impl Rand {
        fn new_node(iter: impl IntoIterator<Item = (f64, SimpNode)>) -> SimpNode {
            SimpNode::Chance(Rand(iter.into_iter().collect()))
        }
    }

    impl IntoIterator for Rand {
        type Item = (f64, SimpNode);
        type IntoIter = vec::IntoIter<Self::Item>;

        fn into_iter(self) -> Self::IntoIter {
            let Rand(vec) = self;
            vec.into_iter()
        }
    }

    impl ChanceNode<Term, SimpPlayer> for Rand {}

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    struct Info;

    #[derive(Debug, Hash, PartialEq, Eq)]
    enum Action {
        A,
        B,
    }

    struct SimpPlayer {
        player: Player,
        actions: Vec<(Action, SimpNode)>,
    }

    impl SimpPlayer {
        fn new_node(
            player: Player,
            iter: impl IntoIterator<Item = (Action, SimpNode)>,
        ) -> SimpNode {
            SimpNode::Player(SimpPlayer {
                player,
                actions: iter.into_iter().collect(),
            })
        }
    }

    impl IntoIterator for SimpPlayer {
        type Item = (Action, SimpNode);
        type IntoIter = vec::IntoIter<Self::Item>;

        fn into_iter(self) -> Self::IntoIter {
            self.actions.into_iter()
        }
    }

    impl PlayerNode<Info, Action, Term, Rand> for SimpPlayer {
        fn get_player(&self) -> Player {
            self.player
        }

        fn get_infoset(&self) -> Info {
            Info
        }
    }

    type SimpNode = Node<Term, Rand, SimpPlayer>;

    #[test]
    fn chance_not_prob() {
        let err_game = Rand::new_node([(1.0, Term::new_node()), (1.0, Term::new_node())]);
        let err = Game::from_node(err_game).unwrap_err();
        assert_eq!(err, GameError::ChanceNotProbability { total_prob: 2.0 });
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
                infoset: Info,
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
                infoset: Info,
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
            StratError::InfosetNum {
                game_num: 1,
                strategy_num: 0
            }
        );

        let action_err = game.regret(&[[1.0]]).unwrap_err();
        assert_eq!(
            action_err,
            StratError::ActionNum {
                ind: 0,
                infoset_actions: 2,
                strategy_actions: 1
            }
        );

        let strat_err = game.regret(&[[1.0, 1.0]]).unwrap_err();
        assert_eq!(
            strat_err,
            StratError::StratNotProbability {
                ind: 0,
                total_prob: 2.0
            }
        );

        assert!(game.regret(&[[0.5, 0.5]]).is_ok());
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
            AkqNode::Terminal(self)
        }
    }

    impl TerminalNode for Pot {
        fn get_one_payoff(&self) -> f64 {
            match self {
                Pot::WonAnte => 1.0,
                Pot::LostAnte => -1.0,
                Pot::WonRaise => 2.0,
                Pot::LostRaise => -2.0,
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

    impl IntoIterator for Deal {
        type Item = (f64, AkqNode);
        type IntoIter = DealIter;

        fn into_iter(self) -> Self::IntoIter {
            let Deal(vec) = self;
            DealIter(1.0 / vec.len() as f64, vec.into_iter())
        }
    }

    impl ChanceNode<Pot, AkqPlayer> for Deal {}

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

    impl IntoIterator for AkqPlayer {
        type Item = (Action, AkqNode);
        type IntoIter = vec::IntoIter<Self::Item>;

        fn into_iter(self) -> Self::IntoIter {
            self.actions.into_iter()
        }
    }

    impl PlayerNode<Card, Action, Pot, Deal> for AkqPlayer {
        fn get_player(&self) -> Player {
            self.player
        }

        fn get_infoset(&self) -> Card {
            self.infoset
        }
    }

    type AkqNode = Node<Pot, Deal, AkqPlayer>;

    #[test]
    fn solution() {
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
        let game = Game::from_node(aqk_poker).unwrap();
        let Solution { regret, strategy } = game.solve(100000, 0.001);
        assert!(regret < 0.001);

        // measure regret manually
        let reg = game.regret(&strategy).unwrap().regret();
        assert!(reg <= regret);

        // in this game, pruning should help, so we try pruning
        let mut pruned_strat = Vec::with_capacity(strategy.len());
        let thresh = 1e-3;
        for strat in strategy.iter() {
            let total: f64 = strat
                .iter()
                .map(|v| if *v < thresh { 0.0 } else { *v })
                .sum();
            let pruned: Vec<_> = strat
                .iter()
                .map(|v| if *v < thresh { 0.0 } else { *v / total })
                .collect();
            pruned_strat.push(pruned);
        }
        let pruned_reg = game.regret(&pruned_strat).unwrap().regret();
        assert!(pruned_reg <= regret);

        // verify we get the right named strategy
        let (one, two) = game.name_strategy(&strategy);

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
}
