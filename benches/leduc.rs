#![feature(test, never_type)]
extern crate test;

use cfr::{
    Digest, Game, GameTree, History, LazySolver, Moves, NodeType, Outcomes, PlayerNum, SolveMethod,
    SolveParams,
};
use std::fmt::Debug;
use std::hash::Hash;
use test::Bencher;

// Leduc hold'em: a deck of `copies` copies of each of `ranks` ranks; each player antes, is dealt one
// private card, bets, then a public card is revealed, players bet again, and there is a showdown --
// pairing the public card beats a high card, else the higher card wins.

/// The rules of a Leduc variant, carried in the game state.
///
/// Vary these to scale the game for benchmarking (more ranks/copies, deeper betting).
#[derive(Clone, Copy, Debug)]
struct Spec {
    ranks: u8,      // distinct ranks, ordered low to high
    copies: u8,     // copies of each rank in the deck
    ante: u32,      // forced ante per player
    bet: [u32; 2],  // bet/raise size in rounds 0 and 1
    max_raises: u8, // bet + raise cap per round
}

/// Standard Leduc: 3 ranks, 2 copies, ante 1, bets 2/4, 2 raises.
///
/// Its first-player Nash value is ~-0.0856.
const STANDARD: Spec = Spec {
    ranks: 3,
    copies: 2,
    ante: 1,
    bet: [2, 4],
    max_raises: 2,
};

/// A larger variant: 6 ranks, 3 copies, deeper betting -- a tree too big to enjoy materializing, so
/// it exercises the lazy solver and the infoset-key footprint.
const BIG: Spec = Spec {
    ranks: 6,
    copies: 3,
    ante: 1,
    bet: [2, 4],
    max_raises: 3,
};

impl Spec {
    /// The worst-case contribution by one player (ante plus the capped raises in both rounds), which
    /// bounds the payoff magnitude the lazy solver needs for pruning.
    fn max_paid(&self) -> f64 {
        let [round0, round1] = self.bet;
        f64::from(self.ante + u32::from(self.max_raises) * (round0 + round1))
    }
}

// repr(u8) so `move as u8` is the move's discriminant -- a distinct code folded into the infoset key
// to record the betting history
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Move {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
}

/// How an infoset is keyed, so the same game can be solved with either primitive (see liar's dice).
///
/// The implementor doubles as the running accumulator of the betting history.
trait InfoKey: Clone + Default + Debug {
    /// The resulting [`Game::Infoset`] key.
    type Key: Clone + Eq + Hash + Send + Sync + Debug;

    /// Record one move (its [`Move`] discriminant) in the running history.
    fn push(&mut self, code: u8);
    /// Finalize the key for the acting player, given their private card and the public card.
    fn finish(&self, private: u8, public: Option<u8>) -> Self::Key;
}

/// Key the infoset by the exact betting `History` plus the visible cards.
#[derive(Clone, Default, Debug)]
struct Exact(History<u8>);

/// Key the infoset by an `N`-lane rolling [`Digest`] of the betting history and visible cards.
#[derive(Clone, Default, Debug)]
struct Hashed<const N: usize>(Digest<N>);

impl InfoKey for Exact {
    type Key = (u8, Option<u8>, History<u8>);

    fn push(&mut self, code: u8) {
        self.0 = self.0.push(code);
    }

    fn finish(&self, private: u8, public: Option<u8>) -> Self::Key {
        (private, public, self.0.clone())
    }
}

impl<const N: usize> InfoKey for Hashed<N> {
    type Key = Digest<N>;

    fn push(&mut self, code: u8) {
        self.0 = self.0.push(&code);
    }

    fn finish(&self, private: u8, public: Option<u8>) -> Self::Key {
        self.0.push(&(private, public))
    }
}

/// A non-terminal hand: the rules, the cards, the betting state, and the infoset-key accumulator.
#[derive(Debug, Clone)]
struct Hand<K: InfoKey> {
    spec: Spec,
    one_card: u8,
    two_card: u8,
    public: Option<u8>,
    round: usize,      // 0 (pre-board) or 1 (post-board)
    paid: [u32; 2],    // total contributions, including antes
    commit: [u32; 2],  // this-round contributions
    bets: u8,          // bet/raise count this round
    acted: bool,       // whether a move has been made this round
    to_act: usize,     // 0 = player one, 1 = player two
    key: K,            // running infoset-key accumulator
}

impl<K: InfoKey> Hand<K> {
    /// The opening hand after the private deal (`one`/`two` are the ranks, antes posted).
    fn start(spec: Spec, one: u8, two: u8) -> Hand<K> {
        Hand {
            spec,
            one_card: one,
            two_card: two,
            public: None,
            round: 0,
            paid: [spec.ante, spec.ante],
            commit: [0, 0],
            bets: 0,
            acted: false,
            to_act: 0,
            key: K::default(),
        }
    }

    /// The acting player's private card.
    fn private(&self) -> u8 {
        if self.to_act == 0 {
            self.one_card
        } else {
            self.two_card
        }
    }

    /// What the acting player must add to match the opponent (0 if no outstanding bet).
    fn to_call(&self) -> u32 {
        self.commit[1 - self.to_act].saturating_sub(self.commit[self.to_act])
    }

    /// Copies of `rank` still in the deck after the two private cards.
    fn copies_left(&self, rank: u8) -> u8 {
        self.spec.copies - u8::from(self.one_card == rank) - u8::from(self.two_card == rank)
    }

    /// The successor after the acting player makes `mv`.
    fn act(&self, mv: Move) -> Leduc<K> {
        let me = self.to_act;
        let to_call = self.to_call();
        let size = self.spec.bet[self.round];
        let mut next = self.clone();
        next.key.push(mv as u8);
        let [one_paid, two_paid] = self.paid;
        match mv {
            // the folder forfeits its contribution to the opponent
            Move::Fold => Leduc::Terminal(if me == 0 {
                -f64::from(one_paid)
            } else {
                f64::from(two_paid)
            }),
            // a closing check (opponent already checked) ends the round; otherwise pass the turn
            Move::Check if self.acted => close(next),
            Move::Check => {
                next.acted = true;
                next.to_act = 1 - me;
                Leduc::Act(next)
            }
            Move::Call => {
                next.commit[me] += to_call;
                next.paid[me] += to_call;
                close(next)
            }
            Move::Bet => {
                next.commit[me] += size;
                next.paid[me] += size;
                next.bets = 1;
                next.acted = true;
                next.to_act = 1 - me;
                Leduc::Act(next)
            }
            Move::Raise => {
                next.commit[me] += to_call + size;
                next.paid[me] += to_call + size;
                next.bets += 1;
                next.acted = true;
                next.to_act = 1 - me;
                Leduc::Act(next)
            }
        }
    }
}

/// End the current betting round: deal the public card after round 0, otherwise go to showdown.
fn close<K: InfoKey>(hand: Hand<K>) -> Leduc<K> {
    if hand.round == 0 {
        Leduc::DealPublic(hand)
    } else {
        Leduc::Terminal(showdown(&hand))
    }
}

/// Player one's net payoff at a showdown (pairing the public card wins, else the higher card wins).
fn showdown<K: InfoKey>(hand: &Hand<K>) -> f64 {
    let public = hand.public.unwrap();
    let one_pair = hand.one_card == public;
    let two_pair = hand.two_card == public;
    let one_wins = if one_pair != two_pair {
        one_pair // exactly one pairs the board (both is impossible -- a rank has at most COPIES cards)
    } else {
        hand.one_card > hand.two_card // neither pairs: higher card wins
    };
    let [one_paid, two_paid] = hand.paid;
    if hand.one_card == hand.two_card {
        0.0 // identical ranks and neither pairs: a tie
    } else if one_wins {
        f64::from(two_paid)
    } else {
        -f64::from(one_paid)
    }
}

/// A chance node: either the initial private deal (carrying the rules) or the public-card deal.
#[derive(Debug, Clone)]
enum Deal<K: InfoKey> {
    Hands(Spec),
    Public(Hand<K>),
}

impl<K: InfoKey> Outcomes<Leduc<K>> for Deal<K> {
    fn len(&self) -> usize {
        match self {
            // every ordered pair of ranks (the weights in `get` account for the copies)
            Deal::Hands(spec) => usize::from(spec.ranks) * usize::from(spec.ranks),
            Deal::Public(hand) => {
                (0..hand.spec.ranks).filter(|&rank| hand.copies_left(rank) > 0).count()
            }
        }
    }

    fn get(&self, index: usize) -> (f64, Leduc<K>) {
        match self {
            Deal::Hands(spec) => {
                let one = (index / usize::from(spec.ranks)) as u8;
                let two = (index % usize::from(spec.ranks)) as u8;
                // ordered ways to deal these ranks: same rank has copies*(copies-1), distinct copies^2
                let weight = if one == two {
                    spec.copies * (spec.copies - 1)
                } else {
                    spec.copies * spec.copies
                };
                (f64::from(weight), Leduc::Act(Hand::start(*spec, one, two)))
            }
            Deal::Public(hand) => {
                // the index-th rank still in the deck, weighted by its remaining copies
                let rank = (0..hand.spec.ranks)
                    .filter(|&rank| hand.copies_left(rank) > 0)
                    .nth(index)
                    .unwrap();
                let mut next = hand.clone();
                next.public = Some(rank);
                next.round = 1;
                next.commit = [0, 0];
                next.bets = 0;
                next.acted = false;
                next.to_act = 0;
                (f64::from(hand.copies_left(rank)), Leduc::Act(next))
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Leduc<K: InfoKey> {
    DealHands(Spec),
    DealPublic(Hand<K>),
    Act(Hand<K>),
    Terminal(f64),
}

impl<K: InfoKey> Leduc<K> {
    /// The root node of a game with the given rules.
    fn new(spec: Spec) -> Leduc<K> {
        Leduc::DealHands(spec)
    }
}

impl<K: InfoKey> Game for Leduc<K> {
    type Action = Move;
    type Infoset = K::Key;
    type ChanceInfoset = !; // each deal is sampled per trajectory, never correlated
    type Chance = Deal<K>;
    type Player = Hand<K>;

    fn into_node(self) -> NodeType<Self> {
        match self {
            Leduc::Terminal(payoff) => NodeType::Terminal(payoff),
            Leduc::DealHands(spec) => NodeType::Chance(None, Deal::Hands(spec)),
            Leduc::DealPublic(hand) => NodeType::Chance(None, Deal::Public(hand)),
            Leduc::Act(hand) => {
                let num = if hand.to_act == 0 {
                    PlayerNum::One
                } else {
                    PlayerNum::Two
                };
                let key = hand.key.finish(hand.private(), hand.public);
                NodeType::Player(num, key, hand)
            }
        }
    }
}

// Actions in a fixed order: with no bet to face, [check, bet?]; otherwise [fold, call, raise?].
impl<K: InfoKey> Moves<Leduc<K>> for Hand<K> {
    fn len(&self) -> usize {
        let raise = usize::from(self.bets < self.spec.max_raises);
        if self.to_call() == 0 {
            1 + raise
        } else {
            2 + raise
        }
    }

    fn action(&self, index: usize) -> Move {
        if self.to_call() == 0 {
            if index == 0 { Move::Check } else { Move::Bet }
        } else {
            match index {
                0 => Move::Fold,
                1 => Move::Call,
                _ => Move::Raise,
            }
        }
    }

    fn apply(&self, index: usize) -> Leduc<K> {
        self.act(self.action(index))
    }
}

#[bench]
fn building(b: &mut Bencher) {
    b.iter(|| test::black_box(GameTree::from_game(Leduc::<Exact>::new(STANDARD)).unwrap()));
}

#[bench]
fn full_single(b: &mut Bencher) {
    let game = GameTree::from_game(Leduc::<Exact>::new(STANDARD)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::Full, 200, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

#[bench]
fn external_single(b: &mut Bencher) {
    let game = GameTree::from_game(Leduc::<Exact>::new(STANDARD)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 1000, 0.0, 1, SolveParams::default())
            .unwrap()
    });
}

// the tree-free path on standard Leduc, keyed by each infoset-key primitive
#[bench]
fn standard_lazy_history(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-STANDARD.max_paid(), STANDARD.max_paid());
        solver.run(&Leduc::<Exact>::new(STANDARD), 1000);
        test::black_box(solver)
    });
}

#[bench]
fn standard_lazy_digest(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-STANDARD.max_paid(), STANDARD.max_paid());
        solver.run(&Leduc::<Hashed<1>>::new(STANDARD), 1000);
        test::black_box(solver)
    });
}

#[bench]
fn standard_lazy_digest_wide(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-STANDARD.max_paid(), STANDARD.max_paid());
        solver.run(&Leduc::<Hashed<2>>::new(STANDARD), 1000);
        test::black_box(solver)
    });
}

// the larger variant -- too big to materialize, so only the lazy path runs it
#[bench]
fn big_lazy_history(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-BIG.max_paid(), BIG.max_paid());
        solver.run(&Leduc::<Exact>::new(BIG), 300);
        test::black_box(solver)
    });
}

#[bench]
fn big_lazy_digest(b: &mut Bencher) {
    b.iter(|| {
        let mut solver = LazySolver::new(-BIG.max_paid(), BIG.max_paid());
        solver.run(&Leduc::<Hashed<1>>::new(BIG), 300);
        test::black_box(solver)
    });
}

#[cfg(test)]
mod tests {
    use super::{Exact, Leduc, STANDARD};
    use cfr::{GameTree, PlayerNum, SolveMethod, SolveParams};

    #[test]
    fn leduc_matches_known_value() {
        // exact CFR must converge AND reproduce standard Leduc's known first-player Nash value
        // (~-0.0856) -- a strong check that the betting state machine, chance weights, and showdown
        // are all faithful; a malformed game converges to some other value
        let game = GameTree::from_game(Leduc::<Exact>::new(STANDARD)).unwrap();
        let (strategies, bound) = game
            .solve(SolveMethod::Full, 5000, 0.0, 1, SolveParams::default())
            .unwrap();
        for player in [PlayerNum::One, PlayerNum::Two] {
            assert!(
                bound.player_regret_bound(player) < 0.1,
                "player {player:?} did not converge"
            );
        }
        let value = strategies.get_info().player_utility(PlayerNum::One);
        assert!(
            (value - -0.0856).abs() < 0.005,
            "Leduc value {value} is off from the known -0.0856"
        );
    }
}
