//! Counterfactual Regret (CFR) is a library for finding an approximate nash equilibrium in
//! two-player zero-sum games of incomplete information with perfect recall, such as poker etc.,
//! using discounted[^dcfr] monte carlo[^mccfr] counterfactual regret minimization[^cfr].
//!
//! # Usage
//!
//! To use the command line tool, see [documentation on
//! github](https://github.com/erikbrinkman/cfr#binary), or use `cfr --help`.
//!
//! To use this as a rust library, implement the [`Game`] trait for your
//! [extensive form game](https://en.wikipedia.org/wiki/Extensive-form_game) -- a state machine that
//! classifies each state as terminal, chance, or player and yields its child states by index. See
//! the trait for the contracts of implementation, and the `kuhn_poker` example for a full game.
//! To solve, build a [`GameTree`] from the game with [`GameTree::from_game`] and call
//! [`GameTree::solve`] (exact, with regret bounds). You can get equilibrium utilities and regret
//! from [`Strategies::get_info`].
//!
//! # Examples
//!
//! Once [`Game`] is implemented for your game, you solve for equilibria like:
//!
//! ```
//! # use cfr::{Game, Moves, NodeType, PlayerNum, SolveMethod, SolveParams, GameTree};
//! # use std::convert::Infallible;
//! # #[derive(Clone, Copy)]
//! # enum MyGame { Start, End(f64) }
//! impl Game for MyGame {
//! #     type Action = bool;
//! #     type Infoset = ();
//! #     type ChanceInfoset = Infallible; // no chance nodes
//! #     type Chance = Infallible;
//! #     type Player = MyGame;
//! #     fn into_node(self) -> NodeType<Self> {
//! #         match self {
//! #             MyGame::Start => NodeType::Player(PlayerNum::One, (), MyGame::Start),
//! #             MyGame::End(payoff) => NodeType::Terminal(payoff),
//! #         }
//! #     }
//!     // ...
//! }
//! # impl Moves<MyGame> for MyGame {
//! #     fn len(&self) -> usize { 2 }
//! #     fn apply(&self, index: usize) -> MyGame { MyGame::End(if index == 0 { 1.0 } else { -1.0 }) }
//! #     fn action(&self, index: usize) -> bool { index == 0 }
//! # }
//! // small game: materialize the tree and solve it exactly
//! let game = GameTree::from_game(MyGame::Start).unwrap();
//! let (strats, reg_bounds) = game.solve(
//!     SolveMethod::External,
//!     100,  // number of iterations
//!     0.0,  // early termination regret
//!     1,    // number of threads
//!     SolveParams::default(), // advanced options
//! ).unwrap();
//! // get named versions, i.e. exportable version
//! let named = strats.as_named();
//! // compute regret and other values
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
//!
//! [^dcfr]: [Brown, Noam, and Tuomas Sandholm. "Solving imperfect-information games via discounted
//!   regret minimization." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33.
//!   No. 01. (2019)](https://ojs.aaai.org/index.php/AAAI/article/view/4007/3885)
#![warn(missing_docs, clippy::pedantic)]

mod compact;
mod error;
mod regret;
mod solve;
mod split;

use compact::{Builder, OptBuilder};
pub use error::{GameError, SolveError, StratError};
pub use solve::RegretParams;
use solve::{external, vanilla};
use split::{split_by, split_by_mut};
use std::borrow::Borrow;
use std::collections::hash_map;
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::hash::Hash;
use std::iter::{self, FusedIterator, Once, Zip};
use std::num::NonZeroUsize;
use std::ptr;
use std::slice;
use std::thread;

/// An enum indicating a player
///
/// This is used to indicate which player is acting at a given player node, as well as get regret
/// or utilities tied to a specific player. Sometimes information for both players is returned as a
/// two element array. In these cases, [One][PlayerNum::One] corresponds to index 0, and one
/// [Two][PlayerNum::Two] corresponds to index 1.
#[derive(Debug, Copy, Eq, Clone, PartialEq, Hash)]
pub enum PlayerNum {
    /// The first player
    One,
    /// The second player
    Two,
}

impl PlayerNum {
    fn ind<T>(self, arr: &[T; 2]) -> &T {
        match (self, arr) {
            (PlayerNum::One, [first, _]) => first,
            (PlayerNum::Two, [_, second]) => second,
        }
    }

    fn ind_mut<T>(self, arr: &mut [T; 2]) -> &mut T {
        match (self, arr) {
            (PlayerNum::One, [first, _]) => first,
            (PlayerNum::Two, [_, second]) => second,
        }
    }

    /// Return `arr` with this player's entry replaced by `value`, leaving the other player's alone.
    fn replace<T>(self, mut arr: [T; 2], value: T) -> [T; 2] {
        *self.ind_mut(&mut arr) = value;
        arr
    }
}

/// Which kind of node a [`Game`] state is, with the data needed to enumerate its children.
///
/// Returned by [`Game::into_node`]. Splitting the per-kind data into variants is what keeps the
/// node-specific operations mutually exclusive: a terminal carries only a payoff, a player node an
/// infoset, and a chance node an optional sampling key (`None` for a unique, uncorrelated node).
pub enum NodeType<G: Game> {
    /// A terminal state, holding the payoff to player one.
    Terminal(f64),
    /// A chance node: an optional sampling key (`None` is a unique, uncorrelated node) and the
    /// weighted outcomes.
    Chance(Option<G::ChanceInfoset>, G::Chance),
    /// A player decision: the acting player, its infoset, and the legal moves.
    Player(PlayerNum, G::Infoset, G::Player),
}

/// A two-player zero-sum game of imperfect information, expressed as a state machine.
///
/// A state classifies itself with [`into_node`][Game::into_node] into a [`NodeType`]; chance and
/// player nodes hand back a lazy, indexable child sequence ([`Outcomes`] / [`Moves`]), so a solver
/// samples a child by `usize` index without ever materializing or naming an action. Action labels
/// ([`Action`][Game::Action]) exist only to import and export named strategies. The solver walks
/// these on demand while sampling rather than building a tree, so advancing a state should be cheap.
///
/// Every state sharing an infoset must enumerate its children in the same order, since the regret
/// table for that infoset is indexed by child position.
pub trait Game: Sized {
    /// An action label, used only to import and export named strategies -- never while sampling.
    type Action;
    /// The acting player's view of the state: the regret-table key and the sampling seed.
    type Infoset;
    /// A chance node's sampling key. Nodes sharing one sample the same outcome; use [`Infallible`]
    /// (and always return `None`) for a game whose chance nodes are never correlated.
    type ChanceInfoset;
    /// The weighted child sequence at a chance node; [`Infallible`] if the game has no chance nodes.
    type Chance: Outcomes<Self>;
    /// The child sequence at a player node. Every game has player nodes, so this is always a real
    /// type.
    type Player: Moves<Self>;

    /// Classify this state as a terminal, chance, or player node.
    fn into_node(self) -> NodeType<Self>;
}

/// A lazy, indexable sequence of chance outcomes, each a `(probability, next state)` pair.
#[allow(clippy::len_without_is_empty)] // a well-formed chance node always has at least one outcome
pub trait Outcomes<G: Game> {
    /// The number of outcomes.
    fn len(&self) -> usize;
    /// The `index`-th outcome as `(probability, next state)`.
    fn get(&self, index: usize) -> (f64, G);
    /// Walk every outcome in order.
    fn iter(&self) -> impl Iterator<Item = (f64, G)> + '_ {
        (0..self.len()).map(|index| self.get(index))
    }
}

/// A lazy, indexable sequence of the legal moves at a player node.
#[allow(clippy::len_without_is_empty)] // a well-formed player node always has at least one action
pub trait Moves<G: Game> {
    /// The number of legal actions.
    fn len(&self) -> usize;
    /// The next state after taking the `index`-th action.
    fn apply(&self, index: usize) -> G;
    /// The label of the `index`-th action, for importing and exporting strategies.
    fn action(&self, index: usize) -> G::Action;
    /// Walk every action paired with its next state.
    fn iter(&self) -> impl Iterator<Item = (G::Action, G)> + '_ {
        (0..self.len()).map(|index| (self.action(index), self.apply(index)))
    }
}

/// [`Infallible`] stands in as the [`Game::Chance`] of a game with no chance nodes: its
/// [`Outcomes`] methods are unreachable because no value can exist. There is deliberately no
/// `Moves` impl -- every game has player nodes, so [`Game::Player`] is always a real type.
impl<G: Game> Outcomes<G> for Infallible {
    fn len(&self) -> usize {
        match *self {}
    }
    fn get(&self, _index: usize) -> (f64, G) {
        match *self {}
    }
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
/// It omits the actual infoset because we need to as a key, when unpacking back into a vec we'll
/// take ownership again.
#[derive(Debug)]
struct PlayerInfosetBuilder<A> {
    actions: Box<[A]>,
    prev_infoset: Option<(usize, usize)>,
}

impl<A> PlayerInfosetBuilder<A> {
    fn new(actions: impl Into<Box<[A]>>, prev_infoset: Option<(usize, usize)>) -> Self {
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
    /// (infoset, action)
    prev_infoset: Option<(usize, usize)>,
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

/// A generic player infoset without specific game type information
trait PlayerInfoset {
    fn num_actions(&self) -> usize;

    fn prev_infoset(&self) -> Option<usize>;
}

impl<I, A> PlayerInfoset for PlayerInfosetData<I, A> {
    fn num_actions(&self) -> usize {
        self.num_actions()
    }

    fn prev_infoset(&self) -> Option<usize> {
        self.prev_infoset.map(|(infoset, _action)| infoset)
    }
}

/// A generic chance infoset without specific game type information
trait ChanceInfoset {
    fn probs(&self) -> &[f64];
}

impl ChanceInfoset for ChanceInfosetData {
    fn probs(&self) -> &[f64] {
        &self.probs
    }
}

/// A compact game representation
///
/// This structure allows computing approximate equilibria, and evaluating the regret and utility
/// of strategy profiles.
#[derive(Debug)]
pub struct GameTree<Infoset, Action> {
    chance_infosets: Box<[ChanceInfosetData]>,
    player_infosets: [Box<[PlayerInfosetData<Infoset, Action>]>; 2],
    single_infosets: [Box<[(Infoset, Action)]>; 2],
    root: Node,
}

/// Two games are equal if and only if they are the same game
impl<I, A> PartialEq for GameTree<I, A> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl<I, A> Eq for GameTree<I, A> {}

impl<I: Hash + Eq, A: Hash + Eq> GameTree<I, A> {
    /// Build a [`GameTree`] from a [`Game`] state machine, so the exact (full-tree) solvers and
    /// their regret bounds apply. Only suitable for games small enough to fit in memory.
    ///
    /// This is a single validating pass: the builder walks [`into_node`][Game::into_node]
    /// recursively, so no intermediate tree is materialized -- each node's children are visited and
    /// freed as the recursion unwinds.
    ///
    /// # Errors
    ///
    /// Returns a [`GameError`] if the game is malformed: empty, contains non-zero-sum payoffs, has
    /// inconsistent infoset action sets, violates perfect recall, contains invalid chance
    /// probabilities, or has more infosets than fit in a `u32`.
    pub fn from_game<G>(root: G) -> Result<Self, GameError>
    where
        G: Game<Infoset = I, Action = A>,
        G::ChanceInfoset: Hash + Eq,
    {
        let mut chance_infosets = OptBuilder::new();
        let mut player_infosets = [Builder::new(), Builder::new()];
        let mut single_infosets = [HashMap::new(), HashMap::new()];
        let [first_player, second_player] = &mut player_infosets;
        let [first_single, second_single] = &mut single_infosets;
        let root = GameTree::recurse(
            &mut chance_infosets,
            &mut [first_player, second_player],
            &mut [first_single, second_single],
            root,
            [None; 2],
        )?;
        let chance_infosets: Box<[_]> = chance_infosets.into_iter().map(|(_, v)| v).collect();
        let player_infosets = player_infosets.map(|pinfo| {
            pinfo
                .into_iter()
                .map(|(infoset, builder)| PlayerInfosetData::new(infoset, builder))
                .collect::<Box<[_]>>()
        });
        // the solver keys its sampler on u32 infoset ids (see `key.rng(infoset as u32)`), so every
        // infoset index has to fit in a u32
        let fits_u32 = |len: usize| u32::try_from(len).is_ok();
        if !fits_u32(chance_infosets.len())
            || !player_infosets.iter().all(|pinfo| fits_u32(pinfo.len()))
        {
            return Err(GameError::TooManyInfosets);
        }
        Ok(GameTree {
            chance_infosets,
            player_infosets,
            single_infosets: single_infosets.map(|sinfo| sinfo.into_iter().collect()),
            root,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn recurse<G>(
        chance_infosets: &mut OptBuilder<G::ChanceInfoset, ChanceInfosetData>,
        player_infosets: &mut [&mut Builder<I, PlayerInfosetBuilder<A>>; 2],
        single_infosets: &mut [&mut HashMap<I, A>; 2],
        state: G,
        prev_infosets: [Option<(usize, usize)>; 2],
    ) -> Result<Node, GameError>
    where
        G: Game<Infoset = I, Action = A>,
        G::ChanceInfoset: Hash + Eq,
    {
        match state.into_node() {
            NodeType::Terminal(payoff) => Ok(Node::Terminal(payoff)),
            NodeType::Chance(info, chance) => {
                let mut probs = Vec::new();
                let mut outcomes = Vec::new();
                for (prob, next) in chance.iter() {
                    if prob > 0.0 && prob.is_finite() {
                        probs.push(prob);
                        outcomes.push(GameTree::recurse(
                            chance_infosets,
                            player_infosets,
                            single_infosets,
                            next,
                            prev_infosets,
                        )?);
                    } else {
                        return Err(GameError::NonPositiveChance);
                    }
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
                                if *data.probs == *probs {
                                    Ok(ind)
                                } else {
                                    Err(GameError::ProbabilitiesNotEqual)
                                }?
                            }
                        };
                        Ok(Node::Chance(Chance::new(outcomes, ind)))
                    }
                }
            }
            NodeType::Player(player_num, infoset, moves) => {
                let mut actions = Vec::new();
                let mut nexts = Vec::new();
                for (action, next) in moves.iter() {
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
                        }
                        let next = nexts.pop().unwrap();
                        GameTree::recurse(
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
                        let next_verts: Result<Box<[_]>, _> = nexts
                            .into_iter()
                            .enumerate()
                            .map(|(action_ind, next)| {
                                // record the action taken, so a later node sharing this infoset
                                // must also share the action (else imperfect recall)
                                let next_prev =
                                    player_num.replace(prev_infosets, Some((info_ind, action_ind)));
                                GameTree::recurse(
                                    chance_infosets,
                                    player_infosets,
                                    single_infosets,
                                    next,
                                    next_prev,
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

/// The method to use for finding approximate equilibria
///
/// When in doubt, you'll probably want to default to [External][SolveMethod::External][^mccfr] as it
/// generally converges faster than [Full][SolveMethod::Full][^cfr] or
/// [Sampled][SolveMethod::Sampled][^mccfr], which are mostly provided for completeness.
///
/// [^cfr]: [Zinkevich, Martin, et al. "Regret minimization in games with incomplete
/// information." Advances in neural information processing systems 20
/// (2007)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).
///
/// [^mccfr]: [Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive
/// games." Advances in neural information processing systems 22
/// (2009)](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveMethod {
    /// This method indicates vanilla counterfactual regret minimization, which does no random
    /// sampling. This can be good for small games, especially ones with very unlikely chance
    /// outcomes, but otherwise spends a lot of computation exploring unimportant areas of the game
    /// tree.
    Full,
    /// This method indicates chance sampled counterfactual regret minimization, which samples
    /// outcomes at chance nodes, but fully explores player actions. This often performs better
    /// than full exploration, but may produce worse results if there are infrequent but very
    /// relevant chance outcomes.
    ///
    /// Since this is sampled, there's a chance that it terminates early with a small regret bound
    /// that's slightly incorrect because it didn't sample enough chance outcomes.
    Sampled,
    /// This method indicates external sampled counterfactual regret minimization, which alternates
    /// between players, and only fully explores the actions of one player, while sampling the
    /// actions of the other according to their current strategy. This often converges faster than
    /// the other methods because it doesn't explore sections of the game tree with low value.
    ///
    /// Since this is sampled, there's a chance that it terminates early with a small regret bound
    /// that's slightly incorrect because it didn't sample enough chance outcomes.
    External,
}

/// Tuning knobs for a solve that aren't part of the regret-matching math. Every field has a sensible
/// default, so override one at a time with struct-update syntax:
/// `SolveParams { check_interval: 1000, ..Default::default() }`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveParams {
    /// The regret-matching / DCFR style. Swap this independently of everything else (for example
    /// [`RegretParams::vanilla`] for undiscounted CFR).
    pub regret: RegretParams,
    /// How often (in iterations) the regret bound is evaluated for early termination. Evaluating it
    /// is a reduction over every infoset, so checking less often is cheaper but overshoots the regret
    /// target by up to `check_interval - 1` iterations.
    pub check_interval: u64,
    /// Seed for the deterministic outcome sampler. A solve is reproducible for a fixed seed; vary it
    /// to draw independent sampled runs. Only the sampled methods ([`SolveMethod::Sampled`],
    /// [`SolveMethod::External`]) consult it.
    pub seed: u64,
    /// How many active-player decision levels the parallel external solver forks over before each
    /// subtree runs serially; higher fills the pool with more tasks. Performance only, never the
    /// result, and consulted only by [`SolveMethod::External`] with `num_threads > 1`.
    pub fork_depth: u32,
}

impl Default for SolveParams {
    fn default() -> Self {
        SolveParams {
            regret: RegretParams::default(),
            check_interval: 256,
            seed: 0,
            fork_depth: 3,
        }
    }
}

impl<I, A> GameTree<I, A> {
    /// Find an approximate Nash equilibrium of the current game
    ///
    /// Often you'll either want to run with `max_iter` as [`usize::MAX`] and `max_reg` as a meaningful
    /// regret, or `max_iter` as a number set based off of the time you have and `max_reg` set to
    /// 0.0, although setting both as a tradeoff also reasonable.
    ///
    /// # Arguments
    ///
    /// - `method` - The method of solving to use. When in doubt prefer
    ///   [External][SolveMethod::External].  See [`SolveMethod`] for details on the distinctions.
    /// - `max_iter` - The maximum number of iterations to run. The maximum regret of the found
    ///   strategy is bounded by the square-root of the number of iterations.
    /// - `max_reg` - Terminate early if the regret of the returned strategy is going to be less
    ///   than this value. With this current implementation this is only valid when the method is
    ///   [Full][SolveMethod::Full] and the params are [vanilla][RegretParams::vanilla]. If using
    ///   other parameters, this should be set lower than the desired regret.
    /// - `num_threads` - The number of threads the external solver uses. Zero selects based off of
    ///   [`thread::available_parallelism`]. One uses a single threaded variant that's more efficient
    ///   when not in a threaded environment. The full-tree and outcome-sampled methods always run
    ///   single threaded and ignore this argument.
    /// - `params` - Solver parameters bundled in a [`SolveParams`]: the [`RegretParams`] discount
    ///   style, how often the regret bound is checked, and the sampler seed. Use
    ///   [`SolveParams::default`] for sensible defaults.
    ///
    /// # Errors
    ///
    /// If a multi-threaded external solve is requested and the thread pool fails to build.
    pub fn solve(
        &self,
        method: SolveMethod,
        max_iter: u64,
        max_reg: f64,
        num_threads: usize,
        params: SolveParams,
    ) -> Result<(Strategies<'_, I, A>, RegretBound), SolveError> {
        let [first_player, second_player] = &self.player_infosets;
        let (regrets, probs) = match method {
            SolveMethod::Full => vanilla::solve_full_single(
                &self.root,
                &self.chance_infosets,
                [first_player, second_player],
                max_iter,
                max_reg,
                &params,
            ),
            SolveMethod::Sampled => vanilla::solve_sampled_single(
                &self.root,
                &self.chance_infosets,
                [first_player, second_player],
                max_iter,
                max_reg,
                &params,
            ),
            SolveMethod::External => {
                let threads = if num_threads == 0 {
                    thread::available_parallelism().map_or(1, NonZeroUsize::get)
                } else {
                    num_threads
                };
                external::solve_external_single(
                    &self.root,
                    &self.chance_infosets,
                    [first_player, second_player],
                    max_iter,
                    max_reg,
                    &params,
                    threads,
                )?
            }
        };
        Ok((Strategies { game: self, probs }, RegretBound::new(regrets)))
    }

    /// The total number of information sets for each player
    #[must_use]
    pub fn num_infosets(&self) -> usize {
        let [one, two] = &self.player_infosets;
        one.len() + two.len()
    }
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> GameTree<I, A> {
    /// Convert a named strategy into [Strategies]
    ///
    /// The input can be any set of types that vaguely iterate over pairs of information sets and
    /// then actions paired to weights. Weights can be any non-negative f64, which will be
    /// normalized in the final strategy. There are no restrictions on iteration order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// # use cfr::{Game, Moves, NodeType, Outcomes, PlayerNum, GameTree};
    /// # use std::convert::Infallible;
    /// # // a chance node leads to player one (actions A/B) or player two (actions 1/2), each at an
    /// # // "info" infoset, then a terminal -- enough to demonstrate named strategies
    /// # #[derive(Clone, Copy)]
    /// # enum Demo { Root, One, Two, End }
    /// # struct DemoDeal; // the chance node's outcomes
    /// # impl Game for Demo {
    /// #     type Action = &'static str;
    /// #     type Infoset = &'static str;
    /// #     type ChanceInfoset = Infallible;
    /// #     type Chance = DemoDeal;
    /// #     type Player = Demo;
    /// #     fn into_node(self) -> NodeType<Self> {
    /// #         match self {
    /// #             Demo::Root => NodeType::Chance(None, DemoDeal),
    /// #             Demo::One => NodeType::Player(PlayerNum::One, "info", Demo::One),
    /// #             Demo::Two => NodeType::Player(PlayerNum::Two, "info", Demo::Two),
    /// #             Demo::End => NodeType::Terminal(0.0),
    /// #         }
    /// #     }
    /// # }
    /// # impl Outcomes<Demo> for DemoDeal {
    /// #     fn len(&self) -> usize { 2 }
    /// #     fn get(&self, index: usize) -> (f64, Demo) { (0.5, [Demo::One, Demo::Two][index]) }
    /// # }
    /// # impl Moves<Demo> for Demo {
    /// #     fn len(&self) -> usize { 2 }
    /// #     fn apply(&self, _index: usize) -> Demo { Demo::End }
    /// #     fn action(&self, index: usize) -> &'static str {
    /// #         match self {
    /// #             Demo::One => ["A", "B"][index],
    /// #             Demo::Two => ["1", "2"][index],
    /// #             _ => unreachable!(),
    /// #         }
    /// #     }
    /// # }
    /// let game = // ...
    /// # GameTree::from_game(Demo::Root).unwrap();
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
    ///
    /// # Errors
    ///
    /// This will error if a valid strategy wasn't specified for every infoset, or it received
    /// invalid infosets or actions for the current [`GameTree`].
    pub fn from_named(
        &self,
        strats: [impl IntoIterator<
            Item = (
                impl Borrow<I>,
                impl IntoIterator<Item = (impl Borrow<A>, impl Borrow<f64>)>,
            ),
        >; 2],
    ) -> Result<Strategies<'_, I, A>, StratError> {
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
            for action in &info.actions {
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
                        let ind = action_inds.get(action).ok_or(StratError::InvalidAction)?;
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
                        return Err(StratError::InvalidAction);
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
        for vals in split_by_mut(&mut dense, infos.iter().map(PlayerInfosetData::num_actions)) {
            let total: f64 = vals.iter().sum();
            if total == 0.0 {
                return Err(StratError::UninitializedInfoset);
            }
            for val in vals.iter_mut() {
                *val /= total;
            }
        }
        if !singles.into_values().all(|(_, seen)| seen) {
            return Err(StratError::UninitializedInfoset);
        }

        Ok(dense)
    }
}

impl<I: Eq, A: Eq> GameTree<I, A> {
    /// Convert a named strategy into [Strategies]
    ///
    /// In case cloning is very expensive, this version doesn't require cloning or hashing, but
    /// otherwise runs in time quadratic in the number of infosets and actions, which is almost
    /// certaintly going to be worse than the cost of cloning.
    ///
    /// Also note that currently constructing the [`GameTree`] requires hashing so that relaxation is
    /// meaningless.
    ///
    /// This is otherwise the same as [`GameTree::from_named`], so see that method for examples.
    ///
    /// # Errors
    ///
    /// Same conditions as [`GameTree::from_named`]: errors if any infoset or action is unknown to the
    /// game, if probabilities for an infoset don't form a valid distribution, or if any infoset
    /// is missing.
    // NOTE this is very similar to from_named, but writing it generically with traits wasn't worth
    // that overhead
    pub fn from_named_eq(
        &self,
        strats: [impl IntoIterator<
            Item = (
                impl Borrow<I>,
                impl IntoIterator<Item = (impl Borrow<A>, impl Borrow<f64>)>,
            ),
        >; 2],
    ) -> Result<Strategies<'_, I, A>, StratError> {
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
                            .ok_or(StratError::InvalidAction)?;
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
                        return Err(StratError::InvalidAction);
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
        for vals in split_by_mut(&mut dense, infos.iter().map(PlayerInfosetData::num_actions)) {
            let total: f64 = vals.iter().sum();
            if total == 0.0 {
                return Err(StratError::UninitializedInfoset);
            }
            for val in vals.iter_mut() {
                *val /= total;
            }
        }
        if !Vec::from(seen_singles).into_iter().all(|seen| seen) {
            return Err(StratError::UninitializedInfoset);
        }

        Ok(dense)
    }
}

/// A compact strategy for both players
///
/// Strategies are tied to a specific game and maintain a reference back to them. Create these from
/// original data using [`GameTree::from_named`].
#[derive(Debug, Clone)]
pub struct Strategies<'a, Infoset, Action> {
    game: &'a GameTree<Infoset, Action>,
    probs: [Box<[f64]>; 2],
}

/// Strategies are equal if they contain identical values and are from the same game
///
/// Equality is strict since action probabilities can't be [nan][f64::NAN].
impl<I, A> PartialEq for Strategies<'_, I, A> {
    fn eq(&self, other: &Self) -> bool {
        self.game == other.game && self.probs == other.probs
    }
}

impl<I, A> Eq for Strategies<'_, I, A> {}

impl<'a, I, A> Strategies<'a, I, A> {
    /// Attach player, infoset, and action information to a strategy
    ///
    /// Use this to convert a strategy profile into an exportable format.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cfr::{Game, Moves, NodeType, PlayerNum, SolveMethod, SolveParams, GameTree};
    /// # use std::convert::Infallible;
    /// # #[derive(Clone, Copy)]
    /// # enum ExData { Start, End }
    /// # impl Game for ExData {
    /// #     type Action = ();
    /// #     type Infoset = ();
    /// #     type ChanceInfoset = Infallible;
    /// #     type Chance = Infallible;
    /// #     type Player = ExData;
    /// #     fn into_node(self) -> NodeType<Self> {
    /// #         match self {
    /// #             ExData::Start => NodeType::Player(PlayerNum::One, (), ExData::Start),
    /// #             ExData::End => NodeType::Terminal(0.0),
    /// #         }
    /// #     }
    /// # }
    /// # impl Moves<ExData> for ExData {
    /// #     fn len(&self) -> usize { 1 }
    /// #     fn apply(&self, _index: usize) -> ExData { ExData::End }
    /// #     fn action(&self, _index: usize) {}
    /// # }
    /// let game = // ...
    /// # GameTree::from_game(ExData::Start).unwrap();
    /// let (strats, _) = game.solve(
    ///     // ...
    /// # SolveMethod::External, 1, 0.0, 1, SolveParams::default()
    /// ).unwrap();
    /// let [player_one_strat, player_two_strat] = strats.as_named();
    /// for (infoset, actions) in player_one_strat {
    ///     for (action, prob) in actions {
    ///         // ...
    ///     }
    /// }
    /// ```
    #[must_use]
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
    ///
    /// Since CFR produces approximate equilibria, often it will return a strategy with very low
    /// probability of playing an action that should never actually be played. Use this to truncate
    /// the probability of small actions when they're played less than `thresh` of the time.
    pub fn truncate(&mut self, thresh: f64) {
        for (infos, box_probs) in self.game.player_infosets.iter().zip(self.probs.iter_mut()) {
            for strat in split_by_mut(
                box_probs.as_mut(),
                infos.iter().map(PlayerInfosetData::num_actions),
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
    /// This computes the avg of the l-`p` earth movers distance between the strategies for each
    /// player, thus the value is between 0 and 1 where 0 represents identical strategies, and 1
    /// represents strategies that share no support.
    ///
    /// This is only a valid distance if `p` is at least 1, which should also be the default
    /// setting.
    ///
    /// # Panics
    ///
    /// Panics if `other` isn't from the same game, or if `p` isn't positive
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn distance(&self, other: &Self, p: f64) -> [f64; 2] {
        assert!(
            self.game == other.game,
            "can only compare strategies for the same game"
        );
        assert!(p > 0.0, "`p` must be positive but got: {p}");
        let dists: Vec<_> = self
            .probs
            .iter()
            .zip(other.probs.iter())
            .zip(self.game.player_infosets.iter())
            .map(|((left, right), info)| {
                let mut dist = 0.0;
                for (left_val, right_val) in left.iter().zip(right.iter()) {
                    dist += (left_val - right_val).abs().powf(p);
                }
                dist / info.len() as f64
            })
            .collect();
        dists.try_into().unwrap()
    }

    /// Get regret and utility information for this strategy profile
    #[must_use]
    pub fn get_info(&self) -> StrategiesInfo {
        let [one_strat, two_strat] = &self.probs;
        let [one_info, two_info] = &self.game.player_infosets;
        let one_split: Box<[&[f64]]> = split_by(
            one_strat,
            one_info.iter().map(PlayerInfosetData::num_actions),
        )
        .collect();
        let two_split: Box<[&[f64]]> = split_by(
            two_strat,
            two_info.iter().map(PlayerInfosetData::num_actions),
        )
        .collect();
        let (util, regrets) = regret::regret(
            &self.game.root,
            &self.game.chance_infosets,
            [one_info, two_info],
            [&*one_split, &*two_split],
        );
        StrategiesInfo { util, regrets }
    }
}

/// Regret bound produced by solving a game
///
/// This represents an upperbound on the regrets of the returned strategy
#[derive(Debug, Clone)]
pub struct RegretBound {
    regrets: [f64; 2],
}

impl RegretBound {
    fn new(regrets: [f64; 2]) -> Self {
        RegretBound { regrets }
    }

    /// Get the regret bound of a specific player
    #[must_use]
    pub fn player_regret_bound(&self, player_num: PlayerNum) -> f64 {
        *player_num.ind(&self.regrets)
    }

    /// Get the total regret bound
    #[must_use]
    pub fn regret_bound(&self) -> f64 {
        let [one, two] = self.regrets;
        f64::max(one, two)
    }
}

/// Information about the regret and utility of a specific strategy profile
#[derive(Debug)]
pub struct StrategiesInfo {
    util: f64,
    regrets: [f64; 2],
}

impl StrategiesInfo {
    /// Get the regret of a specific player
    #[must_use]
    pub fn player_regret(&self, player_num: PlayerNum) -> f64 {
        *player_num.ind(&self.regrets)
    }

    /// Get the total regret
    #[must_use]
    pub fn regret(&self) -> f64 {
        let [one, two] = self.regrets;
        f64::max(one, two)
    }

    /// Get the utility for a specific player
    #[must_use]
    pub fn player_utility(&self, player_num: PlayerNum) -> f64 {
        match player_num {
            PlayerNum::One => self.util,
            PlayerNum::Two => -self.util,
        }
    }
}

/// An iterator over named information sets of a strategy.
///
/// This is returned when getting the names attached to [Strategies] using
/// [`as_named`][Strategies::as_named].
#[derive(Debug)]
pub struct NamedStrategyIter<'a, Infoset, Action> {
    info: &'a [PlayerInfosetData<Infoset, Action>],
    probs: &'a [f64],
    singles: slice::Iter<'a, (Infoset, Action)>,
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

impl<I, A> FusedIterator for NamedStrategyIter<'_, I, A> {}

impl<I, A> ExactSizeIterator for NamedStrategyIter<'_, I, A> {}

/// An iterator over named actions and assiciated probabilities
///
/// This is returned when getting the names attached to [Strategies] using
/// [`as_named`][Strategies::as_named].
#[derive(Debug)]
pub struct NamedStrategyActionIter<'a, Action> {
    iter: ActionType<'a, Action>,
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

impl<A> FusedIterator for NamedStrategyActionIter<'_, A> {}

impl<A> ExactSizeIterator for NamedStrategyActionIter<'_, A> {}

#[cfg(test)]
#[allow(clippy::float_cmp, unused_must_use)]
mod tests {
    use super::{
        Game, GameTree, Moves, NodeType, PlayerNum, SolveMethod, SolveParams,
    };
    use std::convert::Infallible;

    // A small game: player one ("x") has a forced action "a", then player two ("z") picks b/c, and on
    // b player one ("y") picks c/d. All payoffs are 0 -- the tests only exercise infoset/action
    // plumbing, not equilibria.
    #[derive(Debug, Clone, Copy)]
    enum Stage {
        X,
        Z,
        Y,
        Done,
    }

    #[derive(Debug, Clone, Copy)]
    struct TreeGame(Stage);

    impl Game for TreeGame {
        type Action = &'static str;
        type Infoset = &'static str;
        type ChanceInfoset = Infallible;
        type Chance = Infallible;
        type Player = TreeGame;

        fn into_node(self) -> NodeType<Self> {
            match self.0 {
                Stage::X => NodeType::Player(PlayerNum::One, "x", self),
                Stage::Z => NodeType::Player(PlayerNum::Two, "z", self),
                Stage::Y => NodeType::Player(PlayerNum::One, "y", self),
                Stage::Done => NodeType::Terminal(0.0),
            }
        }
    }

    impl Moves<TreeGame> for TreeGame {
        fn len(&self) -> usize {
            match self.0 {
                Stage::X => 1,
                Stage::Z | Stage::Y => 2,
                Stage::Done => unreachable!(),
            }
        }
        fn action(&self, index: usize) -> &'static str {
            match self.0 {
                Stage::X => ["a"][index],
                Stage::Z => ["b", "c"][index],
                Stage::Y => ["c", "d"][index],
                Stage::Done => unreachable!(),
            }
        }
        fn apply(&self, index: usize) -> TreeGame {
            TreeGame(match (self.0, index) {
                (Stage::X, _) => Stage::Z,
                (Stage::Z, 0) => Stage::Y,
                _ => Stage::Done,
            })
        }
    }

    fn create_game() -> GameTree<&'static str, &'static str> {
        GameTree::from_game(TreeGame(Stage::X)).unwrap()
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
            .from_named_eq([
                vec![("x", vec![("a", 1.0)]), ("y", vec![("c", 1.0), ("d", 2.0)])],
                vec![("z", vec![("b", 2.0), ("c", 3.0)])],
            ])
            .unwrap();
        assert_eq!(fast, slow);
        assert_eq!(fast.distance(&slow, 1.0), [0.0; 2]);

        let cloned = game.from_named(fast.as_named()).unwrap();
        assert_eq!(fast, cloned);
    }

    #[test]
    #[should_panic(expected = "same game")]
    fn test_distance_game_panic() {
        let game_one = create_game();
        let (strat_one, _) = game_one
            .solve(SolveMethod::Full, 0, 0.0, 1, SolveParams::default())
            .unwrap();

        let game_two = create_game();
        let (strat_two, _) = game_two
            .solve(SolveMethod::Full, 0, 0.0, 1, SolveParams::default())
            .unwrap();

        strat_one.distance(&strat_two, 1.0);
    }

    #[test]
    #[should_panic(expected = "`p` must be positive")]
    fn test_distance_p_panic() {
        let game = create_game();
        let (strat_one, _) = game
            .solve(SolveMethod::Full, 0, 0.0, 1, SolveParams::default())
            .unwrap();
        let (strat_two, _) = game
            .solve(SolveMethod::Full, 0, 0.0, 1, SolveParams::default())
            .unwrap();
        strat_one.distance(&strat_two, 0.0);
    }
}
