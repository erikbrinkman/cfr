use cfr::{Game, Moves, NodeType, Outcomes, PlayerNum};
use gambit_parser::{Chance, EscapedStr, ExtensiveFormGame, Node, Player};
use num_traits::cast::ToPrimitive;
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Gambit's [`Game`]: a cursor into a parsed [`ExtensiveFormGame`]. Build one with [`TryFrom`].
#[derive(Clone)]
pub struct GambitNode<'a, 'g> {
    node: Node<'a, 'g>,
    info: Rc<GlobalInfo>,
    cum_payoff: f64,
}

impl<'a, 'g> TryFrom<&'g ExtensiveFormGame<'a>> for GambitNode<'a, 'g> {
    type Error = GambitError;

    fn try_from(game: &'g ExtensiveFormGame<'a>) -> Result<Self, GambitError> {
        let players = game.player_names().len();
        if players != 2 {
            return Err(GambitError::PlayerCount(players));
        }
        Ok(GambitNode {
            node: game.root(),
            info: Rc::new(get_global_info(game)?),
            cum_payoff: 0.0,
        })
    }
}

impl GambitNode<'_, '_> {
    /// The constant-sum offset to apply to reported utilities.
    pub fn sum(&self) -> f64 {
        self.info.sum
    }
}

/// A gambit player infoset: keyed internally by its `id`, displayed by its name.
#[derive(Debug, Clone, Copy)]
pub struct GambitInfoset<'a> {
    id: u64,
    name: &'a EscapedStr,
}

impl PartialEq for GambitInfoset<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for GambitInfoset<'_> {}

impl Hash for GambitInfoset<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Display for GambitInfoset<'_> {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.name, out)
    }
}

/// Payoff data shared across every node of a parsed game: the per-outcome player-one payoff and the
/// constant-sum offset.
#[derive(Debug)]
struct GlobalInfo {
    outcomes: HashMap<u64, f64>,
    sum: f64,
}

/// Why a parsed gambit game can't be interpreted as a two-player zero-sum game.
#[derive(Debug)]
pub enum GambitError {
    /// The game doesn't have exactly two players.
    PlayerCount(usize),
    /// A payoff didn't fit in an `f64`.
    NonFinitePayoff,
    /// The game isn't constant sum.
    NotConstantSum,
}

impl Display for GambitError {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            GambitError::PlayerCount(players) => write!(
                out,
                "game file has {players} players, but `cfr` only supports two player games"
            ),
            GambitError::NonFinitePayoff => write!(
                out,
                "received non-finite payoffs in gambit format; make sure payoffs fit in a double"
            ),
            GambitError::NotConstantSum => write!(
                out,
                "gambit file wasn't constant sum : https://github.com/erikbrinkman/cfr#constant-sum"
            ),
        }
    }
}

impl std::error::Error for GambitError {}

/// The player-one payoff a non-zero gambit outcome contributes (outcome 0 means none).
fn outcome_bonus(info: &GlobalInfo, outcome: u64) -> f64 {
    if outcome == 0 {
        0.0
    } else {
        info.outcomes[&outcome]
    }
}

/// A gambit chance node's outcomes, the carrier for [`Game::Chance`].
#[derive(Clone)]
pub struct GambitOutcomes<'a, 'g> {
    info: Rc<GlobalInfo>,
    cum_payoff: f64,
    chance: Chance<'a, 'g>,
}

/// A gambit player node's moves, the carrier for [`Game::Player`].
#[derive(Clone)]
pub struct GambitMoves<'a, 'g> {
    info: Rc<GlobalInfo>,
    cum_payoff: f64,
    player: Player<'a, 'g>,
}

impl<'a, 'g> Game for GambitNode<'a, 'g> {
    type Action = &'a EscapedStr;
    type Infoset = GambitInfoset<'a>;
    type ChanceInfoset = u64;
    type Chance = GambitOutcomes<'a, 'g>;
    type Player = GambitMoves<'a, 'g>;

    fn into_node(self) -> NodeType<Self> {
        match self.node {
            Node::Terminal(term) => {
                let cum_payoff = self.cum_payoff + self.info.outcomes[&term.outcome()];
                NodeType::Terminal(cum_payoff - self.info.sum)
            }
            Node::Chance(chance) => {
                let cum_payoff = self.cum_payoff + outcome_bonus(&self.info, chance.outcome());
                NodeType::Chance(
                    Some(chance.infoset()),
                    GambitOutcomes {
                        info: self.info,
                        cum_payoff,
                        chance,
                    },
                )
            }
            Node::Player(player) => {
                let cum_payoff = self.cum_payoff + outcome_bonus(&self.info, player.outcome());
                let num = match player.player_num() {
                    1 => PlayerNum::One,
                    2 => PlayerNum::Two,
                    _ => panic!("internal error: invalid player number"),
                };
                let infoset = GambitInfoset {
                    id: player.infoset(),
                    name: player.infoset_name(),
                };
                NodeType::Player(
                    num,
                    infoset,
                    GambitMoves {
                        info: self.info,
                        cum_payoff,
                        player,
                    },
                )
            }
        }
    }
}

impl<'a, 'g> Outcomes<GambitNode<'a, 'g>> for GambitOutcomes<'a, 'g> {
    fn len(&self) -> usize {
        self.chance.len()
    }

    fn get(&self, index: usize) -> (f64, GambitNode<'a, 'g>) {
        let (_, prob, node) = self.chance.action_at(index).unwrap();
        (
            prob.to_f64().unwrap(),
            GambitNode {
                node,
                info: self.info.clone(),
                cum_payoff: self.cum_payoff,
            },
        )
    }
}

impl<'a, 'g> Moves<GambitNode<'a, 'g>> for GambitMoves<'a, 'g> {
    fn len(&self) -> usize {
        self.player.len()
    }

    fn action(&self, index: usize) -> &'a EscapedStr {
        self.player.action_at(index).unwrap().0
    }

    fn apply(&self, index: usize) -> GambitNode<'a, 'g> {
        let (_, node) = self.player.action_at(index).unwrap();
        GambitNode {
            node,
            info: self.info.clone(),
            cum_payoff: self.cum_payoff,
        }
    }
}

/// The exactly-two player-one/player-two payoffs of an outcome.
fn payoff_pair<R: ToPrimitive>(payoffs: &[R]) -> [f64; 2] {
    [payoffs[0].to_f64().unwrap(), payoffs[1].to_f64().unwrap()]
}

/// Compute a game's payoff info, validating the cfr requirements (finite payoffs and constant sum).
fn get_global_info(game: &ExtensiveFormGame<'_>) -> Result<GlobalInfo, GambitError> {
    // resolve every outcome id to its payoffs; outcomes may be declared on one node and referenced
    // by id from others
    let mut outcomes: HashMap<u64, [f64; 2]> = HashMap::new();
    let mut queue = vec![game.root()];
    while let Some(node) = queue.pop() {
        match node {
            Node::Terminal(term) => {
                outcomes.insert(term.outcome(), payoff_pair(term.outcome_payoffs()));
            }
            Node::Chance(chance) => {
                if let Some(payoffs) = chance.outcome_payoffs() {
                    outcomes.insert(chance.outcome(), payoff_pair(payoffs));
                }
                queue.extend(chance.actions().map(|(_, _, child)| child));
            }
            Node::Player(player) => {
                if let Some(payoffs) = player.outcome_payoffs() {
                    outcomes.insert(player.outcome(), payoff_pair(payoffs));
                }
                queue.extend(player.actions().map(|(_, child)| child));
            }
        }
    }

    // accumulate payoffs down to each terminal to find the constant-sum value and confirm the game
    // really is constant sum
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut one_min = f64::INFINITY;
    let mut one_max = f64::NEG_INFINITY;
    let mut queue = vec![(game.root(), [0.0; 2])];
    while let Some((node, mut cum)) = queue.pop() {
        match node {
            Node::Terminal(term) => {
                let [one, two] = outcomes[&term.outcome()];
                cum[0] += one;
                cum[1] += two;
                let [one, two] = cum;
                let sum = one + (two - one) / 2.0;
                if !sum.is_finite() {
                    return Err(GambitError::NonFinitePayoff);
                }
                min = min.min(sum);
                max = max.max(sum);
                one_min = one_min.min(one);
                one_max = one_max.max(one);
            }
            Node::Chance(chance) => {
                if chance.outcome() != 0 {
                    let [one, two] = outcomes[&chance.outcome()];
                    cum[0] += one;
                    cum[1] += two;
                }
                queue.extend(chance.actions().map(|(_, _, child)| (child, cum)));
            }
            Node::Player(player) => {
                if player.outcome() != 0 {
                    let [one, two] = outcomes[&player.outcome()];
                    cum[0] += one;
                    cum[1] += two;
                }
                queue.extend(player.actions().map(|(_, child)| (child, cum)));
            }
        }
    }
    if (max - min) * 1000.0 > (one_max - one_min) {
        return Err(GambitError::NotConstantSum);
    }

    let sum = min + (max - min) / 2.0;
    Ok(GlobalInfo {
        outcomes: outcomes
            .into_iter()
            .map(|(id, [one, _])| (id, one))
            .collect(),
        sum,
    })
}

#[cfg(test)]
mod tests {
    use super::{GambitError, GambitNode};
    use cfr::{GameTree, PlayerNum, SolveMethod, SolveParams};
    use gambit_parser::ExtensiveFormGame;

    fn parse(raw: &str) -> ExtensiveFormGame<'_> {
        ExtensiveFormGame::try_from(raw).unwrap()
    }

    #[test]
    fn error_messages_render() {
        for err in [
            GambitError::PlayerCount(3),
            GambitError::NonFinitePayoff,
            GambitError::NotConstantSum,
        ] {
            assert!(!err.to_string().is_empty(), "{err:?} rendered empty");
        }
    }

    #[test]
    fn solves_matching_pennies() {
        // player one picks a side, player two answers from one shared infoset; matching pays player
        // one. Solving the materialized game exercises the player into_node branch, the move carrier,
        // and the named-strategy output (GambitInfoset Display).
        let game = parse(
            r#"EFG 2 R "" { "" "" } p "" 1 1 "p1" { "h" "t" } 0 p "" 2 2 "p2" { "h" "t" } 0 t "" 1 { 1 -1 } t "" 2 { -1 1 } p "" 2 2 "p2" { "h" "t" } 0 t "" 3 { -1 1 } t "" 4 { 1 -1 }"#,
        );
        let node = GambitNode::try_from(&game).unwrap();
        let tree = GameTree::from_game(node).unwrap();
        let (strats, bound) = tree
            .solve(SolveMethod::Full, 50_000, 0.0, 1, SolveParams::default())
            .unwrap();
        for player in [PlayerNum::One, PlayerNum::Two] {
            assert!(
                bound.player_regret_bound(player) < 0.02,
                "player {player:?} not converged"
            );
        }
        for named in strats.as_named() {
            for (infoset, actions) in named {
                assert!(!infoset.to_string().is_empty());
                for (action, prob) in actions {
                    let _ = action.to_string();
                    assert!((prob - 0.5).abs() < 0.05, "not ~50/50: {prob}");
                }
            }
        }
    }

    #[test]
    fn handles_chance_nodes() {
        // a chance node exercises the chance into_node branch and the outcome carrier
        let game = parse(
            r#"EFG 2 R "" { "" "" } c "" 1 "ch" { "x" 1/2 "y" 1/2 } 0 t "" 1 { 1 -1 } t "" 2 { -1 1 }"#,
        );
        let node = GambitNode::try_from(&game).unwrap();
        GameTree::from_game(node).unwrap();
    }

    #[test]
    fn rejects_malformed() {
        assert!(ExtensiveFormGame::try_from("EFG 2 F").is_err());
    }

    #[test]
    fn rejects_non_finite_payoffs() {
        let game = parse(r#"EFG 2 R "" { "" "" } t "" 1 { 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0 }"#);
        assert!(matches!(
            GambitNode::try_from(&game),
            Err(GambitError::NonFinitePayoff)
        ));
    }

    #[test]
    fn rejects_three_players() {
        let game = parse(r#"EFG 2 R "" { "" "" "" } t "" 1 { 0 0 0 }"#);
        assert!(matches!(
            GambitNode::try_from(&game),
            Err(GambitError::PlayerCount(3))
        ));
    }

    #[test]
    fn rejects_non_constant_sum() {
        let game = parse(
            r#"EFG 2 R "" { "" "" } c "" 1 "c1" { "x" 1/2 "y" 1/2 } 0 t "" 1 { 0 0 } t "" 2 { 1 1 }"#,
        );
        assert!(matches!(
            GambitNode::try_from(&game),
            Err(GambitError::NotConstantSum)
        ));
    }

    #[test]
    fn rejects_invalid_game() {
        // infoset 1 is both an ancestor and descendant of itself -- imperfect recall
        let game = parse(
            r#"EFG 2 R "" { "" "" } p "" 1 1 "i1" { "a" "b" } 0 t "" 1 { 0 0 } p "" 1 1 "i1" { "a" "b" } 0 t "" 1 { 0 0 } t "" 1 { 0 0 }"#,
        );
        let node = GambitNode::try_from(&game).unwrap();
        assert!(GameTree::from_game(node).is_err());
    }

    #[test]
    fn constant_sum_offset() {
        let game = parse(r#"EFG 2 R "" { "" "" } t "" 2 { 1 1 }"#);
        assert_eq!(GambitNode::try_from(&game).unwrap().sum(), 1.0);
    }
}
