use cfr::{Game, GameNode, IntoGameNode, PlayerNum};
use gambit_parser::{Error, ExtensiveFormGame, Node};
use num_traits::cast::ToPrimitive;
use std::collections::{HashMap, HashSet};
use std::io::Read;

struct JoinedNode<'a> {
    node: &'a Node<'a>,
    info: &'a GlobalInfo,
    cum_payoff: f64,
}

struct GlobalInfo {
    infoset_names: [HashMap<u64, String>; 2],
    outcomes: HashMap<u64, f64>,
    sum: f64,
}

impl IntoGameNode for JoinedNode<'_> {
    type PlayerInfo = String;
    type Action = String;
    type ChanceInfo = u64;
    // NOTE these are vectors because we need to sort by action to preserve infosets
    type Outcomes = Vec<(f64, Self)>;
    type Actions = Vec<(String, Self)>;

    fn into_game_node(self) -> GameNode<Self> {
        match self.node {
            Node::Terminal(term) => {
                let node_payoff = self.info.outcomes.get(&term.outcome()).unwrap();
                GameNode::Terminal(self.cum_payoff + node_payoff - self.info.sum)
            }
            Node::Chance(chance) => {
                let node_payoff = if chance.outcome() == 0 {
                    0.0
                } else {
                    *self.info.outcomes.get(&chance.outcome()).unwrap()
                };
                let mut outcomes: Vec<_> = chance
                    .actions()
                    .iter()
                    .map(|(act, prob, node)| {
                        (
                            act.to_string(),
                            prob.to_f64().unwrap(),
                            JoinedNode {
                                node,
                                info: self.info,
                                cum_payoff: self.cum_payoff + node_payoff,
                            },
                        )
                    })
                    .collect();
                // NOTE also sort by probs in case actions have identical names, note this will
                // still error when the game is parsed
                outcomes.sort_unstable_by(|(act1, prob1, _), (act2, prob2, _)| {
                    (act1, prob1).partial_cmp(&(act2, prob2)).unwrap()
                });
                GameNode::Chance(
                    Some(chance.infoset()),
                    outcomes
                        .into_iter()
                        .map(|(_, prob, node)| (prob, node))
                        .collect(),
                )
            }
            Node::Player(player) => {
                let player_num = match player.player_num() {
                    1 => PlayerNum::One,
                    2 => PlayerNum::Two,
                    _ => panic!("internal error: invalid player number"),
                };
                let node_payoff = if player.outcome() == 0 {
                    0.0
                } else {
                    *self.info.outcomes.get(&player.outcome()).unwrap()
                };
                let infoset = self.info.infoset_names[player.player_num() - 1]
                    .get(&player.infoset())
                    .unwrap()
                    .to_string();

                let mut actions: Vec<_> = player
                    .actions()
                    .iter()
                    .map(|(act, node)| {
                        (
                            act.to_string(),
                            JoinedNode {
                                node,
                                info: self.info,
                                cum_payoff: self.cum_payoff + node_payoff,
                            },
                        )
                    })
                    .collect();
                actions.sort_unstable_by(|(act1, _), (act2, _)| act1.cmp(act2));
                GameNode::Player(player_num, infoset, actions)
            }
        }
    }
}

/// Gets global game information
///
/// This assumes that the earlier validation was correct and will produce undefined behavior if
/// it's not validated
fn get_global_info(root: &Node<'_>) -> GlobalInfo {
    // get outcomes and named infosets
    let mut infoset_names: [HashMap<_, _>; 2] = Default::default();
    let mut infosets: [HashSet<_>; 2] = Default::default();
    let mut outcomes: HashMap<_, [f64; 2]> = HashMap::new();
    let mut queue = vec![root];
    while let Some(node) = queue.pop() {
        match node {
            Node::Terminal(terminal) => {
                let floats: Vec<f64> = terminal
                    .outcome_payoffs()
                    .iter()
                    .map(|r| r.to_f64().unwrap())
                    .collect();
                outcomes.insert(terminal.outcome(), floats.try_into().unwrap());
            }
            Node::Chance(chance) => {
                if let Some(pays) = chance.outcome_payoffs() {
                    let floats: Vec<f64> = pays.iter().map(|r| r.to_f64().unwrap()).collect();
                    outcomes.insert(chance.outcome(), floats.try_into().unwrap());
                }
                queue.extend(chance.actions().iter().map(|(_, _, next)| next));
            }
            Node::Player(player) => {
                if let Some(pays) = player.outcome_payoffs() {
                    let floats: Vec<f64> = pays.iter().map(|r| r.to_f64().unwrap()).collect();
                    outcomes.insert(player.outcome(), floats.try_into().unwrap());
                }
                if let Some(name) = player.infoset_name() {
                    infoset_names[player.player_num() - 1]
                        .entry(player.infoset())
                        .or_insert_with(|| name.to_string());
                }
                infosets[player.player_num() - 1].insert(player.infoset());
                queue.extend(player.actions().iter().map(|(_, next)| next));
            }
        }
    }

    // for unnamed infosets convert to their number
    for (given, mut seen) in infoset_names.iter_mut().zip(infosets) {
        let mut used_names = HashSet::new();
        for (infoset, name) in given.iter() {
            seen.remove(infoset);
            used_names.insert(name);
        }

        let mut number_names = Vec::new();
        for info in seen {
            let name = info.to_string();
            if used_names.contains(&name) {
                panic!("some infosets had no names, but the string of their number was used as the name of another infoset : https://github.com/erikbrinkman/cfr#duplicate-infosets");
            } else {
                number_names.push((info, name));
            }
        }
        given.extend(number_names);
    }

    // go through terminal payoffs to determine value of constant sum
    let mut min = f64::INFINITY;
    let mut max = -f64::INFINITY;
    let mut one_min = f64::INFINITY;
    let mut one_max = -f64::INFINITY;
    let mut queue = vec![(root, [0.0; 2])];
    while let Some((node, mut cum_pays)) = queue.pop() {
        match node {
            Node::Terminal(terminal) => {
                for (cum, out) in cum_pays
                    .iter_mut()
                    .zip(outcomes.get(&terminal.outcome()).unwrap())
                {
                    *cum += *out
                }
                let [one, two] = cum_pays;
                let sum = one + (two - one) / 2.0;
                if !sum.is_finite() {
                    panic!(
                        "received non-finite payoffs in gambit format; make sure payoffs fit in a double"
                    );
                }
                min = f64::min(min, sum);
                max = f64::max(max, sum);
                one_min = f64::min(one_min, one);
                one_max = f64::max(one_max, one);
            }
            Node::Chance(chance) => {
                if chance.outcome() != 0 {
                    for (cum, out) in cum_pays
                        .iter_mut()
                        .zip(outcomes.get(&chance.outcome()).unwrap())
                    {
                        *cum += *out
                    }
                }
                queue.extend(chance.actions().iter().map(|(_, _, next)| (next, cum_pays)));
            }
            Node::Player(player) => {
                if player.outcome() != 0 {
                    for (cum, out) in cum_pays
                        .iter_mut()
                        .zip(outcomes.get(&player.outcome()).unwrap())
                    {
                        *cum += *out
                    }
                }
                queue.extend(player.actions().iter().map(|(_, next)| (next, cum_pays)));
            }
        }
    }
    if (max - min) * 1000.0 > (one_max - one_min) {
        panic!(
            "gambit file wasn't constant sum : https://github.com/erikbrinkman/cfr#constant-sum"
        );
    }

    // merge results
    GlobalInfo {
        infoset_names,
        outcomes: outcomes
            .into_iter()
            .map(|(info, [one, _])| (info, one))
            .collect(),
        sum: min + (max - min) / 2.0,
    }
}

/// This returns an error on parsing indicating another format should be tried
pub fn from_str(raw: &str) -> Result<(Game<String, String>, f64), Error<'_>> {
    let gambit = ExtensiveFormGame::try_from(raw)?;
    if gambit.player_names().len() != 2 {
        panic!(
            "game file has {} players, but `cfr` only supports two player games",
            gambit.player_names().len()
        );
    }
    let info = get_global_info(gambit.root());
    let game = Game::from_root(JoinedNode {
            node: gambit.root(),
            info: &info,
            cum_payoff: 0.0,
        })
        .expect("couldn't extract a compact game representation due to problems with the structure : https://github.com/erikbrinkman/cfr#game-error");
    Ok((game, info.sum))
}

pub fn from_reader(reader: &mut impl Read) -> (Game<String, String>, f64) {
    let mut buff = String::new();
    reader.read_to_string(&mut buff).unwrap();
    from_str(&buff).expect(
        "couldn't parse gambit game definition : https://github.com/erikbrinkman/cfr#gambit-error",
    )
}

#[cfg(test)]
mod tests {
    #[test]
    #[should_panic(expected = "couldn't parse gambit game definition")]
    fn test_parse() {
        super::from_reader(&mut "EFG 2 F".as_bytes());
    }

    #[test]
    #[should_panic(expected = "received non-finite payoffs in gambit format")]
    fn test_finite_payoffs() {
        super::from_str(r#"EFG 2 R "" { "" "" } t "" 1 { 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0 }"#)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "game file has 3 players")]
    fn test_three_players() {
        super::from_str(r#"EFG 2 R "" { "" "" "" } t "" 1 { 0 0 0 }"#).unwrap();
    }

    #[test]
    #[should_panic(expected = "gambit file wasn't constant sum")]
    fn test_constant_sum_error() {
        super::from_str(
            r#"EFG 2 R "" { "" "" } c "" 1 { "" 1/2 "" 1/2 } 0 t "" 1 { 0 0 } t "" 2 { 1 1 }"#,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(
        expected = "some infosets had no names, but the string of their number was used as the name of another infoset"
    )]
    fn test_infoset_names() {
        super::from_str(
            r#"EFG 2 R "" { "" "" } p "" 1 1 "2" { "" "" } 0 t "" 1 { 0 0 } p "" 1 2 { "" } 0 t "" 1 { 0 0 }"#,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(
        expected = "couldn't extract a compact game representation due to problems with the structure"
    )]
    fn test_repr() {
        super::from_str(
            r#"EFG 2 R "" { "" "" } p "" 1 1 { "" "" } 0 t "" 1 { 0 0 } p "" 1 1 { "" "" } 0 t "" 1 { 0 0 } t "" 1 { 0 0 }"#,
        )
        .unwrap();
    }

    #[test]
    fn test_constant_sum() {
        let (_, sum) = super::from_str(r#"EFG 2 R "" { "" "" } t "" 2 { 1 1 }"#).unwrap();
        assert_eq!(sum, 1.0);
    }
}
