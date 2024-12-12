#![feature(test, never_type, type_alias_impl_trait)]
extern crate test;

use cfr::{Game, GameNode, IntoGameNode, PlayerNum, SolveMethod};
use std::hash::Hash;
use std::iter;
use test::Bencher;

#[derive(Debug, PartialEq)]
enum State<const D: usize> {
    Init(u8),
    Playing(PlayState<D>),
    Terminal(f64),
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct PlayState<const D: usize> {
    total_dice: u8,
    one_dice: [u8; D],
    two_dice: [u8; D],
    bids: Vec<(u8, u8)>,
}

type Combinations<const D: usize> = impl Iterator<Item = ([u8; D], u64)>;

fn combinations<const D: usize>(num: u8) -> Combinations<D> {
    assert_ne!(D, 0);
    let mut vals = [0; D];
    vals[0] = num;
    let mut inds = Vec::with_capacity(D - 1);
    inds.push(0);
    let mut prod = 1;
    iter::once((vals, prod)).chain(iter::from_fn(move || {
        inds.pop().map(|i| {
            prod *= vals[i] as u64;
            vals[i] -= 1;
            vals[i + 1] += 1;
            prod /= vals[i + 1] as u64;
            vals.swap(0, i);
            if vals[i + 1] == 1 && i < D - 2 {
                inds.push(i + 1);
            }
            if vals[0] > 0 {
                inds.push(0);
            }
            (vals, prod)
        })
    }))
}

type Rolls<const D: usize> = impl Iterator<Item = (f64, State<D>)>;

fn roll<const D: usize>(player_dice: u8) -> Rolls<D> {
    combinations(player_dice).flat_map(move |(one_dice, p_one)| {
        combinations(player_dice).map(move |(two_dice, p_two)| {
            (
                (p_one * p_two) as f64,
                State::Playing(PlayState {
                    total_dice: player_dice * 2,
                    one_dice,
                    two_dice,
                    bids: Vec::new(),
                }),
            )
        })
    })
}

type Actions<const D: usize> = impl Iterator<Item = (Action, State<D>)>;

impl<const D: usize> PlayState<D> {
    fn into_actions(self) -> Actions<D> {
        let PlayState {
            total_dice,
            one_dice,
            two_dice,
            bids,
        } = self;
        let player_ind = bids.len() % 2;
        let last_bid = bids.last().copied();

        // challenge action
        let challenge = last_bid.map(move |(card, num)| {
            let ind = card as usize;
            let payoff = if (one_dice[ind] + two_dice[ind] >= num) == (player_ind == 0) {
                -1.0
            } else {
                1.0
            };
            (Action::Challenge, State::Terminal(payoff))
        });

        // bid action
        let (last_card, last_num) = last_bid.unwrap_or((D as u8 - 1, 0));
        let higher_cards = (last_card + 1)..(D as u8);
        let cards = 0..(D as u8);
        let higher_nums = (last_num + 1)..=total_dice;

        // bids with same number but higher card
        let higher_card_bids = higher_cards.map(move |card| (card, last_num));
        // bids with any card but higher num
        let higher_num_bids = cards
            .into_iter()
            .flat_map(move |card| higher_nums.clone().map(move |num| (card, num)));
        // actual action objects
        let bid_iter = higher_card_bids
            .chain(higher_num_bids)
            .map(move |(card, num)| {
                let mut bids = bids.clone();
                bids.push((card, num));
                let next_state = PlayState {
                    total_dice,
                    one_dice,
                    two_dice,
                    bids,
                };
                (Action::Bid(card, num), State::Playing(next_state))
            });
        challenge.into_iter().chain(bid_iter)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum Action {
    Challenge,
    Bid(u8, u8),
}

impl<const D: usize> IntoGameNode for State<D> {
    type PlayerInfo = ([u8; D], Vec<(u8, u8)>);
    type Action = Action;
    type ChanceInfo = !;
    type Outcomes = Rolls<D>;
    type Actions = Actions<D>;

    fn into_game_node(self) -> GameNode<Self> {
        match self {
            State::Init(dice) => GameNode::Chance(None, roll(dice)),
            State::Terminal(payoff) => GameNode::Terminal(payoff),
            State::Playing(state) => {
                let (player_num, dice) = if state.bids.len() % 2 == 0 {
                    (PlayerNum::One, state.one_dice)
                } else {
                    (PlayerNum::Two, state.two_dice)
                };
                GameNode::Player(player_num, (dice, state.bids.clone()), state.into_actions())
            }
        }
    }
}

#[bench]
fn building_small(b: &mut Bencher) {
    b.iter(|| test::black_box(Game::from_root(State::<2>::Init(3)).unwrap()));
}

#[bench]
fn small_external_single(b: &mut Bencher) {
    // 2 2-sided dice
    let game = Game::from_root(State::<2>::Init(2)).unwrap();
    b.iter(|| {
        game.solve(SolveMethod::External, 1000, 0.0, 1, None)
            .unwrap()
    });
}
