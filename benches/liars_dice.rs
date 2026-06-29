#![feature(test, never_type)]
extern crate test;

use cfr::{Game, Moves, NodeType, Outcomes, PlayerNum, SolveMethod, SolveParams, GameTree};
use num_integer::{binomial, multinomial};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;
use test::Bencher;

/// `D` = number of die faces.
#[derive(Debug, Clone, PartialEq)]
enum State<const D: usize> {
    Init(u8), // dice per player
    Playing(PlayState<D>),
    Terminal(f64),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlayState<const D: usize> {
    total_dice: u8,
    one_dice: [u8; D], // counts per face
    two_dice: [u8; D], // counts per face
    bids: Bids,
}

/// Bid history as a persistent stack; the shared `Rc` tail makes push and clone O(1), and each node
/// caches the stack's rolling hash so an infoset hashes in O(1) instead of walking the chain.
#[derive(Debug, Clone, Default)]
struct Bids(Option<Rc<BidNode>>);

#[derive(Debug)]
struct BidNode {
    bid: (u8, u8), // (face, count)
    len: usize,
    digest: u64,
    rest: Bids,
}

impl Bids {
    fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |node| node.len)
    }

    fn last(&self) -> Option<(u8, u8)> {
        self.0.as_ref().map(|node| node.bid)
    }

    fn digest(&self) -> u64 {
        self.0.as_ref().map_or(0, |node| node.digest)
    }

    fn push(&self, bid: (u8, u8)) -> Bids {
        let mut hasher = DefaultHasher::new();
        (self.digest(), bid).hash(&mut hasher);
        Bids(Some(Rc::new(BidNode {
            bid,
            len: self.len() + 1,
            digest: hasher.finish(),
            rest: self.clone(),
        })))
    }
}

impl Hash for Bids {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.digest());
    }
}

impl PartialEq for Bids {
    fn eq(&self, other: &Bids) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            // shared tail or mismatched digest settles it in O(1); else fall back to structure
            (Some(left), Some(right)) => {
                Rc::ptr_eq(left, right)
                    || (left.digest == right.digest
                        && left.bid == right.bid
                        && left.rest == right.rest)
            }
            _ => false,
        }
    }
}

impl Eq for Bids {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Action {
    Challenge,
    Bid(u8, u8), // (face, count)
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
fn dealt<const D: usize>(per_player: u8, one_dice: [u8; D], two_dice: [u8; D]) -> State<D> {
    State::Playing(PlayState {
        total_dice: per_player * 2,
        one_dice,
        two_dice,
        bids: Bids::default(),
    })
}

impl<const D: usize> PlayState<D> {
    /// Apply a player action, returning the successor state.
    fn play(&self, action: Action) -> State<D> {
        match action {
            Action::Challenge => {
                let (card, num) = self.bids.last().unwrap();
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
            Action::Bid(card, num) => State::Playing(PlayState {
                total_dice: self.total_dice,
                one_dice: self.one_dice,
                two_dice: self.two_dice,
                bids: self.bids.push((card, num)),
            }),
        }
    }
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

impl<const D: usize> Game for State<D> {
    type Action = Action;
    type Infoset = ([u8; D], Bids);
    type ChanceInfoset = !; // the single root roll is never correlated
    type Chance = RollOutcomes<D>;
    type Player = PlayState<D>;

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
                let infoset = (dice, state.bids.clone());
                NodeType::Player(num, infoset, state)
            }
        }
    }
}

impl<const D: usize> Outcomes<State<D>> for RollOutcomes<D> {
    // the joint deal is two independent player rolls, so index it as (one, two) in base `single`
    fn len(&self) -> usize {
        self.single * self.single
    }

    fn get(&self, index: usize) -> (f64, State<D>) {
        let (one, one_weight) = nth_roll::<D>(self.per_player, index / self.single);
        let (two, two_weight) = nth_roll::<D>(self.per_player, index % self.single);
        ((one_weight * two_weight) as f64, dealt(self.per_player, one, two))
    }
}

// Legal actions in a fixed order: challenge (if a bid stands), then same-count higher-face bids,
// then strictly-higher-count bids (every face). `len`/`action` index that order directly.
impl<const D: usize> Moves<State<D>> for PlayState<D> {
    fn len(&self) -> usize {
        let last_bid = self.bids.last();
        let (last_face, last_count) = last_bid.unwrap_or((D as u8 - 1, 0));
        let challenge = usize::from(last_bid.is_some());
        let higher_face = (D as u8 - 1 - last_face) as usize;
        let higher_count = D * (self.total_dice - last_count) as usize;
        challenge + higher_face + higher_count
    }

    fn action(&self, index: usize) -> Action {
        let last_bid = self.bids.last();
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

    fn apply(&self, index: usize) -> State<D> {
        self.play(self.action(index))
    }
}

#[bench]
fn building_small(b: &mut Bencher) {
    b.iter(|| test::black_box(GameTree::from_game(State::<2>::Init(3)).unwrap()));
}

#[bench]
fn small_full_single(b: &mut Bencher) {
    // 2 2-sided dice
    let game = GameTree::from_game(State::<2>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::Full, 1000, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_sampled_single(b: &mut Bencher) {
    // 2 2-sided dice
    let game = GameTree::from_game(State::<2>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::Sampled, 1000, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_external_single(b: &mut Bencher) {
    // 2 2-sided dice
    let game = GameTree::from_game(State::<2>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 1000, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_full_multi(b: &mut Bencher) {
    // 2 2-sided dice, default thread count
    let game = GameTree::from_game(State::<2>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::Full, 1000, 0.0, 0, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn small_external_multi(b: &mut Bencher) {
    // 2 2-sided dice, default thread count
    let game = GameTree::from_game(State::<2>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 1000, 0.0, 0, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn medium_external_single(b: &mut Bencher) {
    // 3 2-sided dice produces a substantially larger tree
    let game = GameTree::from_game(State::<2>::Init(3)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 200, 0.0, 1, SolveParams::default())
            .unwrap()
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
