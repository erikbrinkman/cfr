#![feature(test, never_type)]
extern crate test;

use cfr::{
    Digest, Game, GameTree, History, LazySolver, Moves, NodeType, Outcomes, PlayerNum, SolveMethod,
    SolveParams,
};
use num_integer::{binomial, multinomial};
use std::fmt::Debug;
use std::hash::Hash;
use test::Bencher;

/// How an infoset is keyed, so the same liar's-dice game can be solved with either primitive.
///
/// The implementor doubles as the running per-bid accumulator (`Exact` needs none -- the `History`
/// already lives in `PlayState` for move generation), so the benches compare the two infoset-key
/// primitives head to head on one game.
trait InfoKey<const D: usize>: Clone + Default + Debug {
    /// The resulting [`Game::Infoset`] key.
    type Key: Clone + Eq + Hash + Send + Sync + Debug;

    /// Record one more bid in the running history.
    fn push(&mut self, bid: (u8, u8));
    /// Finalize the infoset key for the acting player, whose private dice are `dice`.
    fn finish(&self, bids: &History<(u8, u8)>, dice: [u8; D]) -> Self::Key;
}

/// Key the infoset by the exact bid `History` (an `Arc` cons-list): O(1) clone, exact equality.
#[derive(Clone, Default, Debug)]
struct Exact;

/// Key the infoset by an `N`-lane rolling [`Digest`] of the bids and dice: `Copy`, no allocation,
/// lossy.
#[derive(Clone, Default, Debug)]
struct Hashed<const N: usize>(Digest<N>);

impl<const D: usize> InfoKey<D> for Exact {
    type Key = ([u8; D], History<(u8, u8)>);

    // the `History` already lives in `PlayState`, so there is nothing extra to accumulate
    fn push(&mut self, _bid: (u8, u8)) {}

    fn finish(&self, bids: &History<(u8, u8)>, dice: [u8; D]) -> Self::Key {
        (dice, bids.clone())
    }
}

impl<const D: usize, const N: usize> InfoKey<D> for Hashed<N> {
    type Key = Digest<N>;

    fn push(&mut self, bid: (u8, u8)) {
        self.0 = self.0.push(&bid);
    }

    fn finish(&self, _bids: &History<(u8, u8)>, dice: [u8; D]) -> Self::Key {
        // fold the acting player's private dice into the running bid digest
        self.0.push(&dice)
    }
}

/// `D` = number of die faces, `K` = how infosets are keyed (see [`InfoKey`]).
#[derive(Debug, Clone)]
enum State<const D: usize, K: InfoKey<D>> {
    Init(u8), // dice per player
    Playing(PlayState<D, K>),
    Terminal(f64),
}

#[derive(Debug, Clone)]
struct PlayState<const D: usize, K: InfoKey<D>> {
    total_dice: u8,
    one_dice: [u8; D],       // counts per face
    two_dice: [u8; D],       // counts per face
    bids: History<(u8, u8)>, // (face, count) per bid; drives move generation
    key: K,                  // running infoset-key accumulator
}

/// Number of distinct rolls of `num` dice over `faces` faces (weak compositions of `num`).
fn roll_count(num: u8, faces: usize) -> usize {
    binomial(num as usize + faces - 1, faces - 1)
}

/// The `index`-th roll of `num` dice over `D` faces, as face counts plus its multinomial weight (the
/// number of ordered rolls that produce it). Unranks the weak composition coordinate by coordinate,
/// binary-searching each face's count; the common two-face case is a direct split.
fn nth_roll<const D: usize>(num: u8, index: usize) -> ([u8; D], u64) {
    let mut counts = [0u8; D];
    if D == 2 {
        counts[0] = index as u8;
        counts[1] = num - index as u8;
    } else {
        let mut index = index;
        let mut remaining = num;
        for (face, slot) in counts.iter_mut().enumerate().take(D - 1) {
            let rest = D - 1 - face; // faces still to fill after this one
            // completions whose face value is below `taken` number `total - roll_count(remaining -
            // taken, ..)`; binary-search the largest `taken` whose bucket starts at or before `index`
            let total = roll_count(remaining, rest + 1);
            let (mut lo, mut hi) = (0, remaining);
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if total - roll_count(remaining - mid, rest + 1) <= index {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            index -= total - roll_count(remaining - lo, rest + 1);
            remaining -= lo;
            *slot = lo;
        }
        counts[D - 1] = remaining;
    }
    (counts, multinomial(&counts.map(u64::from)))
}

/// The opening play state after dealing `per_player` dice to each player.
fn dealt<const D: usize, K: InfoKey<D>>(
    per_player: u8,
    one_dice: [u8; D],
    two_dice: [u8; D],
) -> State<D, K> {
    State::Playing(PlayState {
        total_dice: per_player * 2,
        one_dice,
        two_dice,
        bids: History::new(),
        key: K::default(),
    })
}

impl<const D: usize, K: InfoKey<D>> PlayState<D, K> {
    /// Apply a player action, returning the successor state.
    fn play(&self, action: Action) -> State<D, K> {
        match action {
            Action::Challenge => {
                let (card, num) = self.bids.head().copied().unwrap();
                let ind = card as usize;
                let player_ind = self.bids.len() % 2;
                let payoff =
                    if (self.one_dice[ind] + self.two_dice[ind] >= num) == (player_ind == 0) {
                        -1.0
                    } else {
                        1.0
                    };
                State::Terminal(payoff)
            }
            Action::Bid(card, num) => {
                let mut key = self.key.clone();
                key.push((card, num));
                State::Playing(PlayState {
                    total_dice: self.total_dice,
                    one_dice: self.one_dice,
                    two_dice: self.two_dice,
                    bids: self.bids.push((card, num)),
                    key,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Action {
    Challenge,
    Bid(u8, u8), // (face, count)
}

/// The root chance node's joint dice rolls (the carrier for [`Game::Chance`]). Holds the per-player
/// dice count and caches `single`, the number of distinct single-player rolls.
#[derive(Debug)]
struct RollOutcomes<const D: usize> {
    per_player: u8,
    single: usize,
}

impl<const D: usize> RollOutcomes<D> {
    fn new(per_player: u8) -> RollOutcomes<D> {
        RollOutcomes {
            per_player,
            single: roll_count(per_player, D),
        }
    }
}

impl<const D: usize, K: InfoKey<D>> Game for State<D, K> {
    type Action = Action;
    type Infoset = K::Key;
    type ChanceInfoset = !; // the single root roll is never correlated
    type Chance = RollOutcomes<D>;
    type Player = PlayState<D, K>;

    fn into_node(self) -> NodeType<Self> {
        match self {
            State::Terminal(payoff) => NodeType::Terminal(payoff),
            State::Init(per_player) => NodeType::Chance(None, RollOutcomes::new(per_player)),
            State::Playing(state) => {
                let (num, dice) = if state.bids.len() % 2 == 0 {
                    (PlayerNum::One, state.one_dice)
                } else {
                    (PlayerNum::Two, state.two_dice)
                };
                let infoset = state.key.finish(&state.bids, dice);
                NodeType::Player(num, infoset, state)
            }
        }
    }
}

impl<const D: usize, K: InfoKey<D>> Outcomes<State<D, K>> for RollOutcomes<D> {
    // the joint deal is two independent player rolls, so index it as (one, two) in base `single`
    fn len(&self) -> usize {
        self.single * self.single
    }

    fn get(&self, index: usize) -> (f64, State<D, K>) {
        let (one, one_weight) = nth_roll::<D>(self.per_player, index / self.single);
        let (two, two_weight) = nth_roll::<D>(self.per_player, index % self.single);
        (
            (one_weight * two_weight) as f64,
            dealt::<D, K>(self.per_player, one, two),
        )
    }
}

// Legal actions in a fixed order: challenge (if a bid stands), then same-count higher-face bids,
// then strictly-higher-count bids (every face). `len`/`action` index that order directly.
impl<const D: usize, K: InfoKey<D>> Moves<State<D, K>> for PlayState<D, K> {
    fn len(&self) -> usize {
        let last_bid = self.bids.head().copied();
        let (last_face, last_count) = last_bid.unwrap_or((D as u8 - 1, 0));
        let challenge = usize::from(last_bid.is_some());
        let higher_face = (D as u8 - 1 - last_face) as usize;
        let higher_count = D * (self.total_dice - last_count) as usize;
        challenge + higher_face + higher_count
    }

    fn action(&self, index: usize) -> Action {
        let last_bid = self.bids.head().copied();
        let challenge = usize::from(last_bid.is_some());
        if challenge == 1 && index == 0 {
            return Action::Challenge;
        }
        let (last_face, last_count) = last_bid.unwrap_or((D as u8 - 1, 0));
        let higher_face = (D as u8 - 1 - last_face) as usize;
        let bid = index - challenge;
        if bid < higher_face {
            return Action::Bid(last_face + 1 + bid as u8, last_count);
        }
        let bid = bid - higher_face;
        let per_face = (self.total_dice - last_count) as usize;
        Action::Bid((bid / per_face) as u8, last_count + 1 + (bid % per_face) as u8)
    }

    fn apply(&self, index: usize) -> State<D, K> {
        self.play(self.action(index))
    }
}

#[bench]
fn building_small(b: &mut Bencher) {
    b.iter(|| test::black_box(GameTree::from_game(State::<2, Exact>::Init(3)).unwrap()));
}

#[bench]
fn small_full_single(b: &mut Bencher) {
    // 2 2-sided dice
    let game = GameTree::from_game(State::<2, Exact>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::Full, 1000, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_sampled_single(b: &mut Bencher) {
    // 2 2-sided dice
    let game = GameTree::from_game(State::<2, Exact>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::Sampled, 1000, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_external_single(b: &mut Bencher) {
    // 2 2-sided dice
    let game = GameTree::from_game(State::<2, Exact>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 1000, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_full_multi(b: &mut Bencher) {
    // 2 2-sided dice, default thread count
    let game = GameTree::from_game(State::<2, Exact>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::Full, 1000, 0.0, 0, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_external_multi(b: &mut Bencher) {
    // 2 2-sided dice, default thread count
    let game = GameTree::from_game(State::<2, Exact>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 1000, 0.0, 0, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn medium_external_single(b: &mut Bencher) {
    // 3 2-sided dice produces a substantially larger tree
    let game = GameTree::from_game(State::<2, Exact>::Init(3)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 200, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

// The tree-free path keeps only a per-infoset regret table, so these pairs solve the same game and
// iteration count with each infoset-key primitive -- the exact `History` vs the lossy `Digest` --
// to compare them head to head.
#[bench]
fn small_lazy_history(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Exact>::Init(2), 1000);
        test::black_box(solver)
    });
}

#[bench]
fn small_lazy_digest(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Hashed<1>>::Init(2), 1000);
        test::black_box(solver)
    });
}

// the wider digest halves the collision odds again for an extra lane of hashing per push
#[bench]
fn small_lazy_digest_wide(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Hashed<2>>::Init(2), 1000);
        test::black_box(solver)
    });
}

#[bench]
fn medium_lazy_history(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Exact>::Init(3), 200);
        test::black_box(solver)
    });
}

#[bench]
fn medium_lazy_digest(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Hashed<1>>::Init(3), 200);
        test::black_box(solver)
    });
}

#[bench]
fn medium_lazy_digest_wide(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Hashed<2>>::Init(3), 200);
        test::black_box(solver)
    });
}

// 8 2-sided dice per player: a deep game (long bid histories) the lazy path explores only along
// sampled trajectories -- the regime where History's growing Arc chains cost most against Digest's
// flat key
#[bench]
fn large_lazy_history(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Exact>::Init(8), 100);
        test::black_box(solver)
    });
}

#[bench]
fn large_lazy_digest(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-1.0, 1.0);
        solver.run(&State::<2, Hashed<1>>::Init(8), 100);
        test::black_box(solver)
    });
}

#[cfg(test)]
mod tests {
    use super::{multinomial, nth_roll, roll_count};
    use std::collections::HashSet;

    // every index maps to a distinct valid roll, and the indices cover them all exactly once
    fn is_bijection<const D: usize>(num: u8) {
        let single = roll_count(num, D);
        let mut seen = HashSet::new();
        for index in 0..single {
            let (counts, weight) = nth_roll::<D>(num, index);
            assert_eq!(u8::try_from(counts.iter().map(|&c| usize::from(c)).sum::<usize>()), Ok(num));
            assert_eq!(weight, multinomial(&counts.map(u64::from)));
            assert!(seen.insert(counts), "duplicate roll at index {index}");
        }
        assert_eq!(seen.len(), single);
    }

    #[test]
    fn nth_roll_unranks_every_roll() {
        for num in 0..=6 {
            is_bijection::<2>(num); // the direct two-face path
            is_bijection::<3>(num); // the general binary-search path
            is_bijection::<4>(num);
        }
    }
}
