//! Counterfactual Regret (CFR) is a library for finding an approximate nash equilibrium in
//! two-player zero-sum games of incomplete information, such as poker etc., using counterfactual
//! regret minimization[^cfr] and variants[^mccfr].
//!
//! # Usage
//!
//! To use this library, define the [IntoGameNode] trait for the representation of your [extensive
//! form game](https://en.wikipedia.org/wiki/Extensive-form_game). See the trait for details on how
//! to implement this. The use of [IntoIterator] allows this conversion to be zero-copy for
//! expensive types. Once this trait is defined, a [Game] can be created with [Game::from_root].
//! Compute an approximate equilibrium with `solve_*` methods likes [Game::solve_full]. You can
//! then get information like utilities and regret from [Strategies::get_info].
//!
//! # Examples
//!
//! To compute a nash equilibrium, simply define a game by implementing the relevant traits on your
//! input data, then execute the following functions:
//!
//! ```
//! use cfr::{GameNode, Game, IntoGameNode};
//! struct ExData {
//! }
//! impl IntoGameNode for ExData {
//! # type PlayerInfo = ();
//! # type ChanceInfo = ();
//! # type Action = ();
//! # type Actions = Vec<((), ExData)>;
//! # type Outcomes = Vec<(f64, ExData)>;
//! # fn into_game_node(self) -> GameNode<Self> { GameNode::Terminal(0.0) }
//! }
//! let game = Game::from_root(ExData {
//! }).unwrap();
//! let (strats, reg_bounds) = game.solve_full(100, 0.0);
//! let named = strats.as_named();
//! let strat_info = strats.get_info();
//! let regret = strat_info.regret();
//! ```
//!
//! [^cfr]: [Zinkevich, Martin, et al. "Regret minimization in games with incomplete information."
//!   Advances in neural information processing systems 20
//!   (2007)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).
//!
//! [^mccfr]: [Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive
//!   games." Advances in neural information processing systems 22
//!   (2009)](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).
#![warn(missing_docs)]

mod compact;
mod error;
mod regret;
mod solve;
mod split;

use compact::{Builder, OptBuilder};
pub use error::{GameError, StratError};
use solve::{external, vanilla};
use split::{split_by, split_by_mut};
use std::borrow::Borrow;
use std::collections::hash_map;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::{self, FusedIterator, Once, Zip};
use std::ptr;
use std::slice;

/// An enum indicating which player a state in the tree belongs to
#[derive(Debug, Copy, Eq, Clone, PartialEq, Hash)]
pub enum PlayerNum {
    /// The first player
    One,
    /// The second player
    Two,
}

impl PlayerNum {
    fn ind<'a, T>(&self, arr: &'a [T; 2]) -> &'a T {
        match (self, arr) {
            (PlayerNum::One, [first, _]) => first,
            (PlayerNum::Two, [_, second]) => second,
        }
    }

    fn ind_mut<'a, T>(&self, arr: &'a mut [T; 2]) -> &'a mut T {
        match (self, arr) {
            (PlayerNum::One, [first, _]) => first,
            (PlayerNum::Two, [_, second]) => second,
        }
    }
}

/// An intemediary representation of a node in a game tree
///
/// This enum represents a conversion type from custom data to a game node that can be turned
/// into a full game representation. By implementing [IntoGameNode] on a custom tree-like object,
/// you can specify a lazy allocation into the internal representation of a game, and then perform
/// the conversion with [Game::from_root].
#[derive(Debug)]
pub enum GameNode<T: IntoGameNode + ?Sized> {
    /// A terminal node represents the end of a game, and should return the payoff to player one
    Terminal(f64),
    /// A chance node selects randomly between several outcomes
    ///
    /// The first element of the chance node is an optional infoset, if omitted its assumed this
    /// chance node has a unique infoset. Chance nodes with the same infoset must have the same
    /// outcome probabilities in the same order. When random sampling, chance nodes with the same
    /// infoset will be sampled the same way.
    ///
    /// The second element should implement [IntoIterator] with an `Item` that's a tuple of outcome
    /// probabilities, and a type that can be converted into a [GameNode].
    Chance(Option<T::ChanceInfo>, T::Outcomes),
    /// A player node indicate a place where agents make a strategic decision
    ///
    /// The first element is which player number this node corresponds to.
    ///
    /// The second element is the infoset of this node. Nodes with the same infoset must specify
    /// the same actions in the same order.
    ///
    /// The final element should implement [IntoIterator] with an `Item` that's a tuple of an
    /// action and a type that can be converted into a [GameNode].
    Player(PlayerNum, T::PlayerInfo, T::Actions),
}

/// A trait that defines how to convert game-tree-like data into a [Game]
///
/// Define this trait on your custom data type to allow zero-copy conversion into the internal game
/// tree representation to enable game solving. There are a lot of associated types that define how
/// your game is represented.
///
/// The trait ultimately resolves to converting each of your tree nodes into a coresponding
/// [GameNode] that contains all the information necessary for the internal game structure.
///
/// # Examples
///
/// If you're constructing your data from scratch and don't have a custom representation then the
/// easiest way to structure your data is with a custom singleton wrapper. Any data types that fix
/// the required contracts should work in this scenario, but note that information sets are defined
/// by the order of actions, so using a structure without consistent iteration order could cause
/// exceptions when trying to create a full game.
///
/// ```
/// # use cfr::{GameNode, IntoGameNode, Game, PlayerNum};
/// struct Node(GameNode<Node>);
///
/// #[derive(Hash, PartialEq, Eq)]
/// enum Impossible {}
///
/// impl IntoGameNode for Node {
///     type PlayerInfo = u64;
///     type Action = String;
///     type ChanceInfo = Impossible;
///     type Outcomes = Vec<(f64, Node)>;
///     type Actions = Vec<(String, Node)>;
///
///     fn into_game_node(self) -> GameNode<Self> {
///         self.0
///     }
/// }
///
/// let game = Game::from_root(
///     Node(GameNode::Player(PlayerNum::One, 1, vec![
///         ("fixed".into(), Node(GameNode::Terminal(0.0))),
///         ("random".into(), Node(GameNode::Chance(None, vec![
///             (0.5, Node(GameNode::Terminal(1.0))),
///             (0.5, Node(GameNode::Terminal(-1.0))),
///         ]))),
///     ]))
/// );
/// ```
///
/// However, this can also be used to create more advanced games in a lazy manner. This example is
/// only to illustrate how you would do that.
///
/// ```
/// # use cfr::{GameNode, IntoGameNode, Game, PlayerNum};
/// struct Node(u64);
///
/// #[derive(Hash, PartialEq, Eq)]
/// enum Impossible {}
///
/// struct ActionIter(u64);
///
/// impl Iterator for ActionIter {
///     type Item = (u64, Node);
///
///     fn next(&mut self) -> Option<Self::Item> {
///         if self.0 > 2 {
///             self.0 -= 2;
///             Some((self.0, Node(self.0)))
///         } else {
///             None
///         }
///     }
/// }
///
/// impl IntoGameNode for Node {
///     type PlayerInfo = u64;
///     type Action = u64;
///     type ChanceInfo = Impossible;
///     type Outcomes = [(f64, Node); 0];
///     type Actions = ActionIter;
///
///     fn into_game_node(self) -> GameNode<Self> {
///         if self.0 == 0 {
///             GameNode::Terminal(0.0)
///         } else {
///             let num = if self.0 % 2 == 0 {
///                 PlayerNum::One
///             } else {
///                 PlayerNum::Two
///             };
///             GameNode::Player(num, self.0, ActionIter(self.0 + 1))
///         }
///     }
/// }
///
/// let game = Game::from_root(Node(6));
/// ```
pub trait IntoGameNode {
    /// The type for player information sets
    ///
    /// All nodes that have the same player information set are indistinguishable from each other.
    /// That means that they must have the same actions available in the same order. In addition,
    /// this library only works for games with perfect recall, which means that a player can't
    /// forget their own actions. Another way to states this is that all nodes with the same
    /// infoset must all have follow the same infoset for that player.
    type PlayerInfo: Eq;
    /// The type of the player action
    ///
    /// Player nodes have an iterator of actions attached to future states. The actual action
    /// representation isn't that important, but infosets must have the same actions in the same
    /// order. When converting a set of strategies back into their named representations, these
    /// will be used to represent them.
    type Action: Eq;
    /// The information set type for chance nodes
    ///
    /// Chance node information sets have the same identical actions restrictions that player
    /// infosets do, but don't require perfect recall. The benefit of specifying chance infosets is
    /// that sampling based methods can preserve the correlation in sampling which helps
    /// convergence. For example if chance is revealing cards, later draws may be independent of
    /// player actions, and so should be in the same infoset.
    ///
    /// Since these only help convergence, they are optional. If you know these are unspecified,
    /// this should be set to the `!` type, or any empty type.
    type ChanceInfo: Eq;
    /// The type for iterating over the actions in a chance node
    type Outcomes: IntoIterator<Item = (f64, Self)>;
    /// The type for iterating over the actions in a player nodes
    type Actions: IntoIterator<Item = (Self::Action, Self)>;

    /// Convert this type into a `GameNode`
    ///
    /// Note that the GameNode is just an intemediary representation meant to convert custom types
    /// into a [Game].
    fn into_game_node(self) -> GameNode<Self>;
}

#[derive(Debug)]
enum Node {
    /// A terminal node, the game is over the payoff to player one
    Terminal(f64),
    /// A chance node, the game advances independent of player action
    Chance(Chance),
    /// a node in the tree where the player can choose between different actions
    Player(Player),
}

#[derive(Debug)]
struct Chance {
    outcomes: Box<[Node]>,
    infoset: usize,
}

impl Chance {
    fn new(data: impl Into<Box<[Node]>>, infoset: usize) -> Chance {
        let outcomes = data.into();
        Chance { outcomes, infoset }
    }
}

#[derive(Debug)]
struct Player {
    num: PlayerNum,
    infoset: usize,
    actions: Box<[Node]>,
}

#[derive(Debug)]
struct ChanceInfosetData {
    probs: Box<[f64]>,
}

impl ChanceInfosetData {
    fn new(data: impl Into<Box<[f64]>>) -> ChanceInfosetData {
        ChanceInfosetData { probs: data.into() }
    }
}

/// This is a builder for played infosets
///
/// It omits the actual infoset because we need to as a key, when inpacking back into a vec we'll
/// take ownership again.
#[derive(Debug)]
struct PlayerInfosetBuilder<A> {
    actions: Box<[A]>,
    prev_infoset: Option<usize>,
}

impl<A> PlayerInfosetBuilder<A> {
    fn new(actions: impl Into<Box<[A]>>, prev_infoset: Option<usize>) -> Self {
        PlayerInfosetBuilder {
            actions: actions.into(),
            prev_infoset,
        }
    }
}

#[derive(Debug)]
struct PlayerInfosetData<I, A> {
    infoset: I,
    actions: Box<[A]>,
    prev_infoset: Option<usize>,
}

impl<I, A> PlayerInfosetData<I, A> {
    fn new(infoset: I, builder: PlayerInfosetBuilder<A>) -> Self {
        PlayerInfosetData {
            infoset,
            actions: builder.actions,
            prev_infoset: builder.prev_infoset,
        }
    }

    fn num_actions(&self) -> usize {
        self.actions.len()
    }
}

trait PlayerInfoset {
    fn num_actions(&self) -> usize;

    fn prev_infoset(&self) -> Option<usize>;
}

impl<I, A> PlayerInfoset for PlayerInfosetData<I, A> {
    fn num_actions(&self) -> usize {
        self.num_actions()
    }

    fn prev_infoset(&self) -> Option<usize> {
        self.prev_infoset
    }
}

trait ChanceInfoset {
    fn probs(&self) -> &[f64];
}

impl ChanceInfoset for ChanceInfosetData {
    fn probs(&self) -> &[f64] {
        &self.probs
    }
}

/// A compact game representation that includes strategies for finding approximate nash equilibria
/// and computing regret of strategies.
#[derive(Debug)]
pub struct Game<I, A> {
    chance_infosets: Box<[ChanceInfosetData]>,
    player_infosets: [Box<[PlayerInfosetData<I, A>]>; 2],
    single_infosets: [Box<[(I, A)]>; 2],
    root: Node,
}

impl<I, A> PartialEq for Game<I, A> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl<I, A> Eq for Game<I, A> {}

impl<I: Hash + Eq, A: Hash + Eq> Game<I, A> {
    /// Create a game from the root node of an arbitrary game tree
    ///
    /// To create a game from data, the data must implement [Into<GameNode<...>>].
    pub fn from_root<T>(root: T) -> Result<Self, GameError>
    where
        T: IntoGameNode<PlayerInfo = I, Action = A>,
        T::ChanceInfo: Hash + Eq,
    {
        let mut chance_infosets = OptBuilder::new();
        let mut player_infosets = [Builder::new(), Builder::new()];
        let mut single_infosets = [HashMap::new(), HashMap::new()];
        let [first_player, second_player] = &mut player_infosets;
        let [first_single, second_single] = &mut single_infosets;
        let root = Game::init_recurse(
            &mut chance_infosets,
            &mut [first_player, second_player],
            &mut [first_single, second_single],
            root,
            [None; 2],
        )?;
        Ok(Game {
            chance_infosets: chance_infosets.into_iter().map(|(_, v)| v).collect(),
            player_infosets: player_infosets.map(|pinfo| {
                pinfo
                    .into_iter()
                    .map(|(infoset, builder)| PlayerInfosetData::new(infoset, builder))
                    .collect()
            }),
            single_infosets: single_infosets.map(|sinfo| sinfo.into_iter().collect()),
            root,
        })
    }

    fn init_recurse<T>(
        chance_infosets: &mut OptBuilder<T::ChanceInfo, ChanceInfosetData>,
        player_infosets: &mut [&mut Builder<I, PlayerInfosetBuilder<A>>; 2],
        single_infosets: &mut [&mut HashMap<I, A>; 2],
        node: T,
        mut prev_infosets: [Option<usize>; 2],
    ) -> Result<Node, GameError>
    where
        T: IntoGameNode<PlayerInfo = I, Action = A>,
        T::ChanceInfo: Hash + Eq,
    {
        match node.into_game_node() {
            GameNode::Terminal(payoff) => Ok(Node::Terminal(payoff)),
            GameNode::Chance(info, raw_outcomes) => {
                let mut probs = Vec::new();
                let mut outcomes = Vec::new();
                for (prob, next) in raw_outcomes {
                    if prob > 0.0 && prob.is_finite() {
                        probs.push(prob);
                        outcomes.push(Game::init_recurse(
                            chance_infosets,
                            player_infosets,
                            single_infosets,
                            next,
                            prev_infosets,
                        )?);
                    } else {
                        return Err(GameError::NonPositiveChance);
                    };
                }

                match outcomes.len() {
                    0 => Err(GameError::EmptyChance),
                    1 => Ok(outcomes.pop().unwrap()),
                    _ => {
                        // renormalize to make sure consistency
                        let total: f64 = probs.iter().sum();
                        for prob in &mut probs {
                            *prob /= total;
                        }
                        let ind = match chance_infosets.entry(info) {
                            compact::Entry::Vacant(ent) => {
                                ent.insert(ChanceInfosetData::new(probs))
                            }
                            compact::Entry::Occupied(ent) => {
                                let (ind, data) = ent.get();
                                if *data.probs != *probs {
                                    Err(GameError::ProbabilitiesNotEqual)
                                } else {
                                    Ok(ind)
                                }?
                            }
                        };
                        Ok(Node::Chance(Chance::new(outcomes, ind)))
                    }
                }
            }
            GameNode::Player(player_num, infoset, raw_actions) => {
                let mut actions = Vec::new();
                let mut nexts = Vec::new();
                for (action, next) in raw_actions {
                    actions.push(action);
                    nexts.push(next);
                }
                match actions.len() {
                    0 => Err(GameError::EmptyPlayer),
                    1 => {
                        let action = actions.pop().unwrap();
                        match player_num.ind_mut(single_infosets).entry(infoset) {
                            hash_map::Entry::Occupied(ent) => {
                                if ent.get() != &action {
                                    return Err(GameError::ActionsNotEqual);
                                }
                            }
                            hash_map::Entry::Vacant(ent) => {
                                ent.insert(action);
                            }
                        };
                        let next = nexts.pop().unwrap();
                        Game::init_recurse(
                            chance_infosets,
                            player_infosets,
                            single_infosets,
                            next,
                            prev_infosets,
                        )
                    }
                    _ => {
                        let info_ind = match player_num.ind_mut(player_infosets).entry(infoset) {
                            compact::Entry::Occupied(ent) => {
                                let (ind, info) = ent.get();
                                if *info.actions != *actions {
                                    Err(GameError::ActionsNotEqual)
                                } else if &info.prev_infoset != player_num.ind(&prev_infosets) {
                                    Err(GameError::ImperfectRecall)
                                } else {
                                    Ok(ind)
                                }
                            }
                            compact::Entry::Vacant(ent) => {
                                let hash_names: HashSet<&A> = actions.iter().collect();
                                if hash_names.len() == actions.len() {
                                    Ok(ent.insert(PlayerInfosetBuilder::new(
                                        actions,
                                        *player_num.ind(&prev_infosets),
                                    )))
                                } else {
                                    Err(GameError::ActionsNotUnique)
                                }
                            }
                        }?;
                        *player_num.ind_mut(&mut prev_infosets) = Some(info_ind);
                        let next_verts: Result<Box<[_]>, _> = nexts
                            .into_iter()
                            .map(|next| {
                                Game::init_recurse(
                                    chance_infosets,
                                    player_infosets,
                                    single_infosets,
                                    next,
                                    prev_infosets,
                                )
                            })
                            .collect();
                        Ok(Node::Player(Player {
                            num: player_num,
                            infoset: info_ind,
                            actions: next_verts?,
                        }))
                    }
                }
            }
        }
    }
}

impl<I, A> Game<I, A> {
    /// Find an approximate Nash equilibrium of the current game
    ///
    /// This will run no more than `max_iter` iterations, and terminate early if it can guarantee
    /// that it's found a solution with regret smaller than `max_reg`.
    ///
    /// This uses the full vanilla CFR algorithm which is deterministc, but tends to have worse
    /// convergence than [Game::solve_sampled] or [Game::solve_external].
    pub fn solve_full(&self, max_iter: u64, max_reg: f64) -> (Strategies<I, A>, [f64; 2]) {
        let [first_player, second_player] = &self.player_infosets;
        let (regrets, probs) = vanilla::solve_full(
            &self.root,
            &self.chance_infosets,
            [first_player, second_player],
            max_iter,
            max_reg,
        );
        (Strategies { game: self, probs }, regrets)
    }

    /// Find an approximate Nash equilibrium of the current game
    ///
    /// This will run no more than `max_iter` iterations, and terminate early if it can guarantee
    /// that it's found a solution with regret smaller than `max_reg`.
    ///
    /// This uses the sampled CFR algorithm, where chance nodes are sampled, but both player nodes
    /// are expanded each iteration.
    pub fn solve_sampled(&self, max_iter: u64, max_reg: f64) -> (Strategies<I, A>, [f64; 2]) {
        let [first_player, second_player] = &self.player_infosets;
        let (regrets, probs) = vanilla::solve_sampled(
            &self.root,
            &self.chance_infosets,
            [first_player, second_player],
            max_iter,
            max_reg,
        );
        (Strategies { game: self, probs }, regrets)
    }

    /// Find an approximate Nash equilibrium of the current game
    ///
    /// This will run no more than `max_iter` iterations, and terminate early if it can guarantee
    /// that it's found a solution with regret smaller than `max_reg`.
    ///
    /// This uses the external CFR algorithm, where chance nodes and the oposing player's
    /// strategies are sampled at each iteraton.
    pub fn solve_external(&self, max_iter: u64, max_reg: f64) -> (Strategies<I, A>, [f64; 2]) {
        let [first_player, second_player] = &self.player_infosets;
        let (regrets, probs) = external::solve_external(
            &self.root,
            &self.chance_infosets,
            [first_player, second_player],
            max_iter,
            max_reg,
        );
        (Strategies { game: self, probs }, regrets)
    }
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> Game<I, A> {
    /// Convert a named strategy into [Strategies]
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// # use cfr::{GameNode, Game, IntoGameNode, PlayerNum};
    /// # struct Node(GameNode<Node>);
    /// # impl IntoGameNode for Node {
    /// # type PlayerInfo = &'static str;
    /// # type Action = &'static str;
    /// # type ChanceInfo = &'static str;
    /// # type Outcomes = Vec<(f64, Node)>;
    /// # type Actions = Vec<(&'static str, Node)>;
    /// # fn into_game_node(self) -> GameNode<Self> { self.0 }
    /// # }
    /// let game = Game::from_root(
    /// # Node(GameNode::Chance(None, vec![
    /// #     (0.5, Node(GameNode::Player(PlayerNum::One, "info", vec![
    /// #         ("A", Node(GameNode::Terminal(0.0))),
    /// #         ("B", Node(GameNode::Terminal(0.0))),
    /// #     ]))),
    /// #     (0.5, Node(GameNode::Player(PlayerNum::Two, "info", vec![
    /// #         ("1", Node(GameNode::Terminal(0.0))),
    /// #         ("2", Node(GameNode::Terminal(0.0))),
    /// #     ]))),
    /// # ]))
    /// ).unwrap();
    /// let one: HashMap<&'static str, HashMap<&'static str, f64>> = [
    ///     ("info", [("A", 0.2), ("B", 0.8)].into())
    /// ].into();
    /// let two: HashMap<&'static str, HashMap<&'static str, f64>> = [
    ///     ("info", [("1", 0.5), ("2", 0.5)].into())
    /// ].into();
    /// let strat = game.from_named([one, two]).unwrap();
    /// let info = strat.get_info();
    /// info.regret();
    /// ```
    pub fn from_named(
        &self,
        strats: [impl IntoIterator<
            Item = (
                impl Borrow<I>,
                impl IntoIterator<Item = (impl Borrow<A>, impl Borrow<f64>)>,
            ),
        >; 2],
    ) -> Result<Strategies<I, A>, StratError> {
        let [one_strat, two_strat] = strats;
        let [one_info, two_info] = &self.player_infosets;
        let [one_single, two_single] = &self.single_infosets;
        Ok(Strategies {
            game: self,
            probs: [
                Self::strat_into_box(one_strat, one_info, one_single)?,
                Self::strat_into_box(two_strat, two_info, two_single)?,
            ],
        })
    }

    fn strat_into_box(
        strat: impl IntoIterator<
            Item = (
                impl Borrow<I>,
                impl IntoIterator<Item = (impl Borrow<A>, impl Borrow<f64>)>,
            ),
        >,
        infos: &[PlayerInfosetData<I, A>],
        raw_singles: &[(I, A)],
    ) -> Result<Box<[f64]>, StratError> {
        let mut num_inds = 0;
        let mut inds: HashMap<I, HashMap<A, usize>> = HashMap::with_capacity(infos.len());
        for info in infos {
            let mut actions: HashMap<A, usize> = HashMap::with_capacity(info.num_actions());
            for action in info.actions.iter() {
                actions.insert(action.clone(), num_inds);
                num_inds += 1;
            }
            inds.insert(info.infoset.clone(), actions);
        }
        let mut dense = vec![0.0; num_inds].into_boxed_slice();

        let mut singles: HashMap<_, _> = raw_singles
            .iter()
            .map(|(info, act)| (info, (act, false)))
            .collect();

        for (binfoset, actions) in strat {
            let infoset = binfoset.borrow();
            if let Some(action_inds) = inds.get(infoset) {
                for (baction, bprob) in actions {
                    let action = baction.borrow();
                    let prob = bprob.borrow();
                    if prob >= &0.0 && prob.is_finite() {
                        let ind = action_inds
                            .get(action)
                            .ok_or(StratError::InvalidInfosetAction)?;
                        dense[*ind] = *prob;
                    } else {
                        return Err(StratError::InvalidProbability);
                    }
                }
            } else if let Some((act, seen)) = singles.get_mut(infoset) {
                for (baction, bprob) in actions {
                    let action = baction.borrow();
                    let prob = bprob.borrow();
                    if &action != act {
                        return Err(StratError::InvalidInfosetAction);
                    } else if prob >= &0.0 && prob.is_finite() {
                        *seen = true;
                    } else {
                        return Err(StratError::InvalidProbability);
                    }
                }
            } else {
                return Err(StratError::InvalidInfoset);
            }
        }

        // check we wrote to all locations
        for vals in split_by_mut(&mut *dense, infos.iter().map(|info| info.num_actions())) {
            let total: f64 = vals.iter().sum();
            if total == 0.0 {
                return Err(StratError::InvalidInfosetProbability);
            } else {
                for val in vals.iter_mut() {
                    *val /= total;
                }
            }
        }
        if !singles.into_values().all(|(_, seen)| seen) {
            return Err(StratError::InvalidInfosetProbability);
        }

        Ok(dense)
    }
}

impl<I: Hash + Eq, A: Hash + Eq> Game<I, A> {
    /// Convert a named strategy into [Strategies]
    ///
    /// In case cloning is very expensive, this version doesn't require cloning, but otherwise
    /// runs in time quadratic in the number of infosets and actions, which is almost certaintly
    /// going to be worse than the cost of cloning.
    ///
    /// This is otherwise the same as [Game::from_named], so see that method for examples.
    // NOTE this is very similar to from_named, but writing it generically with traits wasn't worth
    // that overhead
    pub fn from_named_slow(
        &self,
        strats: [impl IntoIterator<
            Item = (
                impl Borrow<I>,
                impl IntoIterator<Item = (impl Borrow<A>, impl Borrow<f64>)>,
            ),
        >; 2],
    ) -> Result<Strategies<I, A>, StratError> {
        let [one_strat, two_strat] = strats;
        let [one_info, two_info] = &self.player_infosets;
        let [one_single, two_single] = &self.single_infosets;
        Ok(Strategies {
            game: self,
            probs: [
                Self::strat_into_box_slow(one_strat, one_info, one_single)?,
                Self::strat_into_box_slow(two_strat, two_info, two_single)?,
            ],
        })
    }

    fn strat_into_box_slow(
        strat: impl IntoIterator<
            Item = (
                impl Borrow<I>,
                impl IntoIterator<Item = (impl Borrow<A>, impl Borrow<f64>)>,
            ),
        >,
        infos: &[PlayerInfosetData<I, A>],
        singles: &[(I, A)],
    ) -> Result<Box<[f64]>, StratError> {
        let mut action_inds = Vec::with_capacity(infos.len());
        let mut num_inds = 0;
        for info in infos {
            action_inds.push(num_inds);
            num_inds += info.num_actions();
        }
        let mut dense = vec![0.0; num_inds].into_boxed_slice();
        let mut seen_singles: Box<[_]> = vec![false; singles.len()].into();

        for (binfoset, actions) in strat {
            let infoset = binfoset.borrow();
            if let Some((ind, info)) = infos
                .iter()
                .enumerate()
                .find(|(_, info)| &info.infoset == infoset)
            {
                let info_ind = action_inds[ind];
                for (baction, bprob) in actions {
                    let action = baction.borrow();
                    let prob = bprob.borrow();
                    if prob >= &0.0 && prob.is_finite() {
                        let (act_ind, _) = info
                            .actions
                            .iter()
                            .enumerate()
                            .find(|(_, act)| act == &action)
                            .ok_or(StratError::InvalidInfosetAction)?;
                        dense[info_ind + act_ind] = *prob;
                    } else {
                        return Err(StratError::InvalidProbability);
                    }
                }
            } else if let Some((ind, (_, act))) = singles
                .iter()
                .enumerate()
                .find(|(_, (info, _))| info == infoset)
            {
                for (baction, bprob) in actions {
                    let action = baction.borrow();
                    let prob = bprob.borrow();
                    if action != act {
                        return Err(StratError::InvalidInfosetAction);
                    } else if prob >= &0.0 && prob.is_finite() {
                        seen_singles[ind] = true;
                    } else {
                        return Err(StratError::InvalidProbability);
                    }
                }
            } else {
                return Err(StratError::InvalidInfoset);
            }
        }

        // check that we wrote to every location
        for vals in split_by_mut(&mut *dense, infos.iter().map(|info| info.num_actions())) {
            let total: f64 = vals.iter().sum();
            if total == 0.0 {
                return Err(StratError::InvalidInfosetProbability);
            } else {
                for val in vals.iter_mut() {
                    *val /= total;
                }
            }
        }
        if !Vec::from(seen_singles).into_iter().all(|seen| seen) {
            return Err(StratError::InvalidInfosetProbability);
        }

        Ok(dense)
    }
}

/// A compact strategy for both players
#[derive(Debug, Clone)]
pub struct Strategies<'a, I, A> {
    game: &'a Game<I, A>,
    probs: [Box<[f64]>; 2],
}

impl<'a, I, A> PartialEq for Strategies<'a, I, A> {
    fn eq(&self, other: &Self) -> bool {
        self.game == other.game && self.probs == other.probs
    }
}

impl<'a, I, A> Eq for Strategies<'a, I, A> {}

impl<'a, I, A> Strategies<'a, I, A> {
    /// Attach player, infoset, and action information to a strategy
    pub fn as_named<'b: 'a>(&'b self) -> [NamedStrategyIter<'a, I, A>; 2] {
        let [info_one, info_two] = &self.game.player_infosets;
        let [single_one, single_two] = &self.game.single_infosets;
        let [probs_one, probs_two] = &self.probs;
        [
            NamedStrategyIter::new(info_one, probs_one, single_one),
            NamedStrategyIter::new(info_two, probs_two, single_two),
        ]
    }

    /// Truncate actions with small probability
    pub fn truncate(&mut self, thresh: f64) {
        for (infos, box_probs) in self.game.player_infosets.iter().zip(self.probs.iter_mut()) {
            for strat in split_by_mut(
                box_probs.as_mut(),
                infos.iter().map(|info| info.num_actions()),
            ) {
                let total: f64 = strat.iter().filter(|p| p > &&thresh).sum();
                for p in strat.iter_mut() {
                    *p = if *p > thresh { *p / total } else { 0.0 }
                }
            }
        }
    }

    /// Get the distance between this strategy and another strategy
    ///
    /// This computes the avg of the l1 earth movers distance between the strategies for each
    /// player, thus the value is between 0 and 1 where 0 represents identical strategies, and 1
    /// represents strategies that share no support.
    ///
    /// # Panics
    ///
    /// Panics if `other` isn't from the same game.
    pub fn distance(&self, other: &Self) -> [f64; 2] {
        assert!(
            self.game == other.game,
            "can only compare strategies for the same game"
        );
        let dists: Vec<_> = self
            .probs
            .iter()
            .zip(other.probs.iter())
            .zip(self.game.player_infosets.iter())
            .map(|((left, right), info)| {
                let mut dist = 0.0;
                for (left_val, right_val) in left.iter().zip(right.iter()) {
                    dist += (left_val - right_val).abs();
                }
                dist / info.len() as f64
            })
            .collect();
        dists.try_into().unwrap()
    }

    /// Get regret and utility information for this strategy profile
    pub fn get_info(&self) -> StrategiesInfo {
        let [one_strat, two_strat] = &self.probs;
        let [one_info, two_info] = &self.game.player_infosets;
        let one_split: Box<[&[f64]]> =
            split_by(one_strat, one_info.iter().map(|info| info.num_actions())).collect();
        let two_split: Box<[&[f64]]> =
            split_by(two_strat, two_info.iter().map(|info| info.num_actions())).collect();
        let (util, regrets) = regret::regret(
            &self.game.root,
            &self.game.chance_infosets,
            [one_info, two_info],
            [&*one_split, &*two_split],
        );
        StrategiesInfo { util, regrets }
    }
}

/// Information about the regret and utility of a specific strategy profile
pub struct StrategiesInfo {
    util: f64,
    regrets: [f64; 2],
}

impl StrategiesInfo {
    /// Get the regret of a specific player
    pub fn player_regret(&self, player_num: PlayerNum) -> f64 {
        *player_num.ind(&self.regrets)
    }

    /// Get the total regret
    pub fn regret(&self) -> f64 {
        let [one, two] = self.regrets;
        f64::max(one, two)
    }

    /// Get the utility for a specific player
    pub fn player_utility(&self, player_num: PlayerNum) -> f64 {
        match player_num {
            PlayerNum::One => self.util,
            PlayerNum::Two => -self.util,
        }
    }
}

/// An iterator over named information sets of a strategy.
///
/// This is returned when getting named [Strategies] using
/// [Strategies::as_named].
#[derive(Debug)]
pub struct NamedStrategyIter<'a, I, A> {
    info: &'a [PlayerInfosetData<I, A>],
    probs: &'a [f64],
    singles: slice::Iter<'a, (I, A)>,
}

impl<'a, I, A> NamedStrategyIter<'a, I, A> {
    fn new(info: &'a [PlayerInfosetData<I, A>], probs: &'a [f64], singles: &'a [(I, A)]) -> Self {
        NamedStrategyIter {
            info,
            probs,
            singles: singles.iter(),
        }
    }
}

impl<'a, I, A> Iterator for NamedStrategyIter<'a, I, A> {
    type Item = (&'a I, NamedStrategyActionIter<'a, A>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((info, rest_infos)) = self.info.split_first() {
            let (probs, rest_probs) = self.probs.split_at(info.num_actions());
            self.info = rest_infos;
            self.probs = rest_probs;
            Some((
                &info.infoset,
                NamedStrategyActionIter {
                    iter: ActionType::Data(info.actions.iter().zip(probs.iter())),
                },
            ))
        } else if let Some((info, act)) = self.singles.next() {
            Some((
                info,
                NamedStrategyActionIter {
                    iter: ActionType::Single(iter::once(act)),
                },
            ))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.probs.len() + self.singles.len();
        (len, Some(len))
    }
}

impl<'a, I, A> FusedIterator for NamedStrategyIter<'a, I, A> {}

impl<'a, I, A> ExactSizeIterator for NamedStrategyIter<'a, I, A> {}

/// An iterator over named actions and assiciated probabilities
///
/// This is returned when getting named [Strategies] using
/// [Strategies::as_named].
#[derive(Debug)]
pub struct NamedStrategyActionIter<'a, A> {
    iter: ActionType<'a, A>,
}

#[derive(Debug)]
enum ActionType<'a, A> {
    Data(Zip<slice::Iter<'a, A>, slice::Iter<'a, f64>>),
    Single(Once<&'a A>),
}

impl<'a, A> Iterator for NamedStrategyActionIter<'a, A> {
    type Item = (&'a A, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.iter {
            ActionType::Data(zip) => zip
                .find(|(_, prob)| prob > &&0.0)
                .map(|(act, &prob)| (act, prob)),
            ActionType::Single(once) => once.next().map(|a| (a, 1.0)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match &self.iter {
            ActionType::Data(zip) => zip.len(),
            ActionType::Single(once) => once.len(),
        };
        (len, Some(len))
    }
}

impl<'a, A> FusedIterator for NamedStrategyActionIter<'a, A> {}

impl<'a, A> ExactSizeIterator for NamedStrategyActionIter<'a, A> {}

#[cfg(test)]
mod tests {
    use super::{Game, GameNode, IntoGameNode, PlayerNum};

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
    fn strat_names() {
        let game = create_game();
        let fast = game
            .from_named([
                vec![("x", vec![("a", 1.0)]), ("y", vec![("c", 1.0), ("d", 2.0)])],
                vec![("z", vec![("b", 2.0), ("c", 3.0)])],
            ])
            .unwrap();
        let slow = game
            .from_named_slow([
                vec![("x", vec![("a", 1.0)]), ("y", vec![("c", 1.0), ("d", 2.0)])],
                vec![("z", vec![("b", 2.0), ("c", 3.0)])],
            ])
            .unwrap();
        assert_eq!(fast, slow);
        assert_eq!(fast.distance(&slow), [0.0; 2]);

        let cloned = game.from_named(fast.as_named()).unwrap();
        assert_eq!(fast, cloned);
    }
}
