Counterfactual Regret (CFR)
===========================
[![crates.io](https://img.shields.io/crates/v/cfr)](https://crates.io/crates/cfr)
[![docs](https://img.shields.io/badge/docs-docs.rs-blue)](https://docs.rs/cfr/latest/cfr/)
[![build](https://github.com/erikbrinkman/cfr/actions/workflows/rust.yml/badge.svg)](https://github.com/erikbrinkman/cfr/actions/workflows/rust.yml)
[![license](https://img.shields.io/github/license/erikbrinkman/cfr)](LICENSE)

Counterfactual regret minimization solver for two-player zero-sum
incomplete-information games in rust. This is a rust library and binary for
computing approximate nash-equilibria of two-player zero-sum
incomplete-information games.

Usage
-----

### Library

To use the library part of this crate, add the following to your `Cargo.toml`.
```toml
[dependencies]
cfr = { version = "0.1.0", default-features = false }
```

Then implement [`IntoGameNode`](FIXME), for a type that represents a node in your game tree (or alternatively can generate new nodes).

Finally execute:
```rust
use cfr::{Game, IntoGameNode};
struct MyData { ... }
impl IntoGameNode for MyData { ... }
let game = Game::from_root(...)?;
let strategies = game.solve_external();
let info = strategies.get_info();
let regret = info.regret();
let [player_one, player_two] = strategies.as_named();
```

### Binary

This package can also be used as a binary with a few different input formats.
install it with

```bash
$ cargo install cfr
```

Solving a game will produce the expected utility to player one, as well as the
regrets and the strategy found in json format.

```bash
$ cfr -i game_file
{
  "expected_one_utility": 0.05555555727916178,
  "player_one_regret": 1.186075528125663e-06,
  "player_two_regret": 8.061797025377127e-05,
  "regret": 8.061797025377127e-05,
  "player_one_strategy": { ... },
  "player_two_regret": { ... }
}
```

The command line tool can interpret a custom [json dsl](#json-format), and
[gambit `.efg`](#gambit-format) files.

#### JSON Format

The DSL is defined by:
```bison
node     ::= terminal || chance || player
terminal ::= { "terminal": <number> }
chance   ::= {
               "chance": {
                 "infoset"?: <string>,
                 "outcomes": {
                   "<named outcome>": { "prob": <number>, "state": node },
                   ...
                 }
               }
             }
player   ::= {
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
    "outcomes": {
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
}
```
In this game there's a 100% chance of the `"single"` outcome, followed by a
move by player one where they have information `"none"` and only have one
action: `"only"`. After selecting that action, they get payoff 0.

#### Gambit Format

The gambit format follows the standard [gambit extensive form game
format](https://gambitproject.readthedocs.io/en/v16.0.2/formats.html), with
some mild restrictions.

- Gambit specifies that actions are optional, but this requires every player and chance node specifies thier actions.
- Actions must be unique, and there are some very mild restrictions on
  non-conflicting information set names (See [duplicate infosets](#duplicate-infosets)).
- The gambit format allows for arbitrary player, non-constant-sum extensive
  form games, but this only allows two-player constant-sum perfect recall
  games.
- For efficiency this uses double precision floats, because equilibria are
  approximate, thus in extreme circumstances payoffs might not be
  representable.

Errors
------

This section has more details on errors the command line might return.

### Json Error

This error occurs when the json doesn't match the expected format for reading.
See [JSON Format](#json-format) for details on the specification, and make sure
that the json you're providing matches that specification.

### Gambit Error

This error occurs when the gambit file can't be parsed. There should be more
info about exactly where the error occured. See [gambit format](#gambit-format)
for more details on the format.

### Duplicate Infosets

Gambit `.efg` files don't require naming infosets, but `cfr` requires string
names for every infoset. It will default to useing the string version of the
infoset number, but this will fail if that infoset name is already taken. For
example:

```
...
p 1 1 "2" { ... } 0
p 1 2 { ... } 0
...
```

will throw this error as long as a name for infoset 2 isn't defined elsewhere.

### Constant Sum

Counterfactual regret minimization only works on constant sum games. Since
gambit files define payoffs independently, this verifies that the range of the
sum of every profile is less than 0.1% of the range of the of the payoffs for a
single player. If you encounter this error, `cfr` will not for this game. 

### Game Error

This error occurs if there were problems with the game specification that made
creating a compact game impossible. See the documentation of
[`GameError`](https://docs.rs/cfr/latest/cfr/enum.GameError.html) for more
details.

### Auto Error

The game file couldn't be parsed by any known game format. In order to get more
detailed errors regarding the parsing failure, try rerunning again with
`--input-format <format>` to get more precise errors

To Do
-----

- [ ] Implement multi threaded variants.
- [ ] A lot of guarantees around memory safety are guaranteed by the nature of
  the tree objects we traverse, but rust can't verify these memory guarantees.
  We currently use standard rust runtime checks like `RefCell` and `Mutex`, but
  we shouldn't need these in all circumstances, and we will be more perofrmant
  by switching to more unsafe rust.
