Counterfactual Regret (CFR)
===========================
[![crates.io](https://img.shields.io/crates/v/cfr)](https://crates.io/crates/cfr)
[![docs](https://img.shields.io/badge/docs-docs.rs-blue)](https://docs.rs/cfr/latest/cfr/)
[![build](https://github.com/erikbrinkman/cfr/actions/workflows/rust.yml/badge.svg)](https://github.com/erikbrinkman/cfr/actions/workflows/rust.yml)
[![license](https://img.shields.io/github/license/erikbrinkman/cfr)](LICENSE)


Counterfactual regret minimization of two-player zero-sum
incomplete-information games in rust. This is a rust library and binary for
computing approximate nash-equilibria of two-player zero-sum
incomplete-information games.

Usage
-----

### Library

To use the library part of this crate, add the following to your `Cargo.toml`.
```toml
[dependencies]
cfr = { version = "0.0.1", default-features = false }
```

Then implement [`TerminalNode`](https://docs.rs/cfr/latest/cfr/trait.TerminalNode.html), [`ChanceNode`](https://docs.rs/cfr/latest/cfr/trait.ChanceNode.html), and [`PlayerNode`](https://docs.rs/cfr/latest/cfr/trait.PlayerNode.html) for your relevant
objects in the game tree.

Finally execute:
```rust
use cfr::{Game, Solution};
let game = Game::from_node(...)?;
let Solution {regret, strategy} = game.solve();
let named = game.name_strategy(&strategy)?;
```

### Binary

This package can also be used as a binary utilizing json format. To use first
install it with

```bash
$ cargo install cfr
```

Game trees are defined using a JSON domain specific language that should be
passed in via `stdin`. The resulting strategy for each player in addition to
information about the strategy pair will be written to `stdout`.

```bash
$ cfr < game.json
{
  "expected_utility": 0.05555555727916178,
  "player_one_regret": 1.186075528125663e-06,
  "player_two_regret": 8.061797025377127e-05,
  "regret": 8.061797025377127e-05,
  "player_one_strategy": { ... },
  "player_two_regret": { ... }
}
```

The DSL is defined by:
```bison
node      ::= terminal || chance || player
terminal  ::= { "terminal": <number> }
chance    ::= {
                "chance": {
                  "<named outcome>": { "prob": <number>, "state": node },
                  ...
                }
              }
player    ::= {
                "player": {
                  "player_one": <bool>,
                  "infoset": <string>,
                  "actions": { "<named action>": node, ... }
                }
              }
```

A minimal example highlighting all types of nodes, but of an uninteresting game is:
```json
{
  "chance": {
    "single": {
      "prob": 1.0,
      "state": {
        "player": {
          "player_one": true,
          "infoset": "none",
          "actions": {
            "only": {
              "terminal": 0.0
            }
          }
        }
      }
    }
  }
}
```
In this game there's a 100% chance of the `"single"` outcome, followed by a
move by player one where they have information `"none"` and only have one
action: `"only"`. After selecting that action, they get payoff 0.

Errors
------

### Json Error

This error occurs when the json doesn't match the expected format for reading.
See [binary](#binary) for details on the specification, and make sure that the
json you're providing matches that specification.

### Game Error

This error occurs if there were problems with the game specification that made
creating a compact game impossible. There are two different ways this could
fail, and all will come with extra information:

- `ChanceNotProbability` : This gets thrown if there's a chance node who's
  total probability between all outcomes isn't one.
- `ActionsNotEqual` : This gets thrown if two different states with the same
  information sets had different actions available to them. It will report what
  information set, player, and actions caused the error.

To Do
-----

- [ ] Efficiently allocating the memory in a way that is compact, requires few 
      allocations, and uses only safe code in rust was note easy to figure out,
      so the current implementation does a lot of small heap allocations to
      initialize the data structures. This could be improved in the future, but
      doing so isn't obvious.
