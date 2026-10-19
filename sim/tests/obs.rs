//! Tests of the learning interface (`sim::obs`).

use sim::data::{Back, BlindKind, Edition, HandType, Joker};
use sim::inject::Inject;
use sim::game::{Game, Phase};
use sim::obs::{self, ObsMut};
use sim::Action;

struct Buf {
    ti: Vec<i32>,
    tf: Vec<f32>,
    gi: Vec<i32>,
    gf: Vec<f32>,
}

impl Buf {
    fn new() -> Buf {
        Buf {
            ti: vec![0; obs::MAX_TOKENS * obs::TOK_INT],
            tf: vec![0.0; obs::MAX_TOKENS * obs::TOK_FLOAT],
            gi: vec![0; obs::GLOB_INT],
            gf: vec![0.0; obs::GLOB_FLOAT],
        }
    }
    /// Panics with the first differing token / global if `self != o`.
    fn assert_same(&self, o: &Buf, what: &str) {
        self.first_diff("tok_i", &self.ti, &o.ti, obs::TOK_INT, what);
        self.first_diff("tok_f", &self.tf, &o.tf, obs::TOK_FLOAT, what);
        self.first_diff("glob_i", &self.gi, &o.gi, 1, what);
        self.first_diff("glob_f", &self.gf, &o.gf, 1, what);
    }

    /// Panics at the first index where `a` and `b` differ (`w` features per
    /// row; token rows are shown with their categorical features).
    fn first_diff<T: PartialEq + std::fmt::Debug>(&self, name: &str, a: &[T], b: &[T], w: usize, what: &str) {
        if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
            let tok = i / w;
            let row = if w > 1 { format!(" token {tok} ints {:?}", &self.ti[tok * obs::TOK_INT..(tok + 1) * obs::TOK_INT]) } else { String::new() };
            panic!("{what}: {name}[{i}] (feature {}){row}: {:?} != {:?}", i % w, a[i], b[i]);
        }
    }

    fn encode(&mut self, g: &Game) -> obs::Layout {
        obs::encode_opts(g, &mut ObsMut { tok_i: &mut self.ti, tok_f: &mut self.tf, glob_i: &mut self.gi, glob_f: &mut self.gf }, true)
    }
}

/// Plays a game with a simple seeded policy, calling `f` at every decision.
fn walk(seed: &str, steps: usize, mut f: impl FnMut(&Game)) {
    let mut g = Game::new_run(seed, Back::Red, 1);
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut acts = Vec::new();
    for _ in 0..steps {
        if g.is_terminal() {
            break;
        }
        f(&g);
        g.legal_actions(&mut acts);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        // Prefer progress: play more often than anything else in rounds.
        let plays: Vec<Action> = acts.iter().copied().filter(|a| matches!(a, Action::Play(_))).collect();
        let a = if g.phase == Phase::Round && !x.is_multiple_of(4) && !plays.is_empty() {
            plays[(x >> 8) as usize % plays.len()]
        } else {
            acts[(x >> 8) as usize % acts.len()]
        };
        g.step(a).unwrap();
    }
}

#[test]
fn preview_matches_full_step() {
    let alt = sim::rng::PseudoRandom::new("PREVIEW0");
    let mut checked = 0;
    for s in 0..6 {
        walk(&format!("PV{s}"), 150, |g| {
            if g.phase != Phase::Round {
                return;
            }
            let mut acts = Vec::new();
            g.legal_actions(&mut acts);
            for a in acts.iter().filter(|a| matches!(a, Action::Play(_))).take(25) {
                let Action::Play(m) = *a else { unreachable!() };
                let (p, _) = obs::preview_score(g, m, &alt).unwrap();
                let mut c = g.clone();
                c.rng = alt.clone();
                c.step(*a).unwrap();
                assert_eq!(p, c.last_hand_score, "mask {m:b}");
                checked += 1;
            }
        });
    }
    assert!(checked > 100);
}

#[test]
fn deck_order_is_not_observable() {
    let mut g = Game::new_run("ORDER", Back::Red, 1);
    g.step(Action::SelectBlind).unwrap();
    let mut a = Buf::new();
    a.encode(&g);
    g.deck.reverse();
    let mut b = Buf::new();
    b.encode(&g);
    assert_eq!(a.ti, b.ti);
    assert_eq!(a.tf, b.tf);
    assert_eq!(a.gf, b.gf);
}

#[test]
fn every_op_has_a_legal_completion() {
    let mut ops = Vec::new();
    let mut buf = Buf::new();
    let mut n = 0;
    for s in 0..40 {
        walk(&format!("OP{s}"), 300, |g| {
            let lay = buf.encode(g);
            obs::ops(g, &lay, 4, &mut ops);
            assert!(!ops.is_empty(), "no ops in {:?}", g.phase);
            for o in &ops {
                // minimal card set: required + lowest allowed cards
                let mut m = o.required;
                let mut i = 0;
                while (m.count_ones() as u8) < o.card_min && i < 32 {
                    if o.allowed >> i & 1 == 1 {
                        m |= 1 << i;
                    }
                    i += 1;
                }
                let dest = if o.dest > 0 { if o.index == 0 && o.kind != obs::op::MOVE_HAND { 1 } else if o.kind == obs::op::MOVE_HAND { (m.trailing_zeros() + 1) % o.dest as u32 } else { 0 } } else { 0 };
                assert!(obs::decode(g, o, m, dest).is_some(), "op {o:?} has no legal completion in {:?}", g.phase);
                n += 1;
            }
        });
    }
    assert!(n > 500, "only {n} ops checked");
}

#[test]
fn face_down_cards_are_hidden() {
    let mut g = Game::new_run("HIDE", Back::Red, 1);
    g.step(Action::SelectBlind).unwrap();
    for c in g.hand.iter_mut() {
        c.face_down = true;
    }
    let mut b = Buf::new();
    let lay = b.encode(&g);
    for i in 0..lay.hand {
        let t = &b.ti[i * obs::TOK_INT..(i + 1) * obs::TOK_INT];
        assert_eq!(t[1], obs::center::HIDDEN);
        assert_eq!(&t[2..7], &[0, 0, 0, 0, 0]);
    }
    assert!(obs::top_plays(&g, 8).is_empty(), "face-down cards must not be previewed");
}

#[test]
fn face_down_identity_is_not_observable() {
    let mut g = Game::new_run("HIDE2", Back::Red, 1);
    g.step(Action::SelectBlind).unwrap();
    g.hand[0].face_down = true;
    let mut a = Buf::new();
    a.encode(&g);
    // Swap the face-down card with an unseen deck card: same observation.
    let k = g.deck.len() / 2;
    std::mem::swap(&mut g.hand[0], &mut g.deck[k]);
    g.hand[0].face_down = true;
    g.deck[k].face_down = false;
    let mut b = Buf::new();
    b.encode(&g);
    a.assert_same(&b, "swapped face-down card");
}

#[test]
fn resample_hidden_keeps_the_observation() {
    let mut n = 0;
    for s in 0..40 {
        let mut states = Vec::new();
        walk(&format!("RS{s}"), 200, |g| states.push(g.clone()));
        // Every other state, alternately with face-down hand cards.
        for (k, g) in states.into_iter().step_by(2).enumerate() {
            let mut g = g;
            if k % 2 == 0 {
                for c in g.hand.iter_mut().step_by(3) {
                    c.face_down = true;
                }
            }
            let mut a = Buf::new();
            a.encode(&g);
            let mut c = g.clone();
            c.resample_hidden(&format!("X{k}"));
            let mut b = Buf::new();
            b.encode(&c);
            a.assert_same(&b, &format!("resample in {:?}", g.phase));
            // The resampled game is still playable.
            let mut acts = Vec::new();
            c.legal_actions(&mut acts);
            if let Some(&a0) = acts.first() {
                c.step(a0).unwrap();
            }
            n += 1;
        }
    }
    assert!(n > 100, "only {n} states checked");
}

#[test]
fn candidates_are_consistent() {
    let mut n = 0;
    for s in 0..20 {
        walk(&format!("CD{s}"), 150, |g| {
            let a = obs::top_plays(g, 8);
            let b = obs::top_plays(g, 8);
            assert_eq!(a, b, "candidates must be deterministic");
            for c in &a {
                assert!(c.min <= c.mean + 1e-9 && c.mean <= c.max + 1e-9, "{c:?}");
                assert!((0.0..=1.0).contains(&c.p_clear));
                n += 1;
            }
            for w in a.windows(2) {
                assert!(w[0].p_clear >= w[1].p_clear);
            }
        });
    }
    assert!(n > 200, "only {n} candidates checked");
}

/// The token at `i` (ints).
fn tok(b: &Buf, i: usize) -> &[i32] {
    &b.ti[i * obs::TOK_INT..(i + 1) * obs::TOK_INT]
}

#[test]
fn deck_token_ties_do_not_leak_order() {
    let mut g = Game::new_run("TIES", Back::Red, 1);
    g.step(Action::SelectBlind).unwrap();
    // Two cards that agree on every categorical feature but not on a bonus.
    let last = g.deck.len() - 1;
    let (suit, rank) = (g.deck[0].suit, g.deck[0].rank);
    g.deck[last].suit = suit;
    g.deck[last].rank = rank;
    g.deck[last].ab.perma_bonus = 10.0;
    for deck_tokens in [false, true] {
        let enc = |g: &Game| {
            let mut b = Buf::new();
            obs::encode_opts(g, &mut ObsMut { tok_i: &mut b.ti, tok_f: &mut b.tf, glob_i: &mut b.gi, glob_f: &mut b.gf }, deck_tokens);
            b
        };
        let a = enc(&g);
        let mut r = g.clone();
        r.deck.reverse();
        a.assert_same(&enc(&r), &format!("reversed deck (deck_tokens={deck_tokens})"));
    }
}

#[test]
fn blind_tokens_show_the_hand_they_name() {
    let mut g = Game::new_run("NAMED", Back::Red, 1);
    g.boss_choice = BlindKind::Ox;
    g.most_played_hand = HandType::Flush;
    let mut b = Buf::new();
    b.encode(&g);
    let boss = (0..obs::MAX_TOKENS).find(|&i| tok(&b, i)[0] == obs::zone::BLIND && tok(&b, i)[7] == 2).unwrap();
    assert_eq!(tok(&b, boss)[3], HandType::Flush as i32 + 1);
    // The Mouth, once a hand has fixed it.
    g.boss_choice = BlindKind::Mouth;
    g.blind.kind = Some(BlindKind::Mouth);
    g.blind.only_hand = Some(HandType::Pair);
    b.encode(&g);
    assert_eq!(tok(&b, boss)[3], HandType::Pair as i32 + 1);
}

#[test]
fn hand_can_be_reordered_in_packs() {
    let mut buf = Buf::new();
    let mut ops = Vec::new();
    let mut packs = 0;
    for s in 0..60 {
        walk(&format!("PK{s}"), 300, |g| {
            if g.phase == Phase::Pack && g.hand.len() >= 2 {
                let lay = buf.encode(g);
                obs::ops(g, &lay, 0, &mut ops);
                for k in [obs::op::SORT_RANK, obs::op::SORT_SUIT, obs::op::MOVE_HAND] {
                    assert!(ops.iter().any(|o| o.kind == k), "op {k} missing in a pack with the hand drawn");
                }
                packs += 1;
            }
        });
    }
    assert!(packs > 0, "no pack opened with the hand drawn");
}

#[test]
fn reorder_destinations_are_checked() {
    let mut g = Game::new_run("DEST", Back::Red, 1);
    g.step(Action::SelectBlind).unwrap();
    let mut buf = Buf::new();
    let lay = buf.encode(&g);
    let mut ops = Vec::new();
    obs::ops(&g, &lay, 0, &mut ops);
    let mv = ops.iter().find(|o| o.kind == obs::op::MOVE_HAND).unwrap();
    assert_eq!(obs::decode(&g, mv, 1, 2), Some(Action::MoveHand { from: 0, to: 2 }));
    // 258 would wrap to 2 as a u8.
    assert_eq!(obs::decode(&g, mv, 1, 258), None);
    assert_eq!(obs::decode(&g, mv, 1, mv.dest as u32), None);
}

#[test]
fn candidates_never_crowd_out_other_ops() {
    let mut g = Game::new_run("ROOM", Back::Red, 1);
    g.step(Action::SelectBlind).unwrap();
    let mut buf = Buf::new();
    let lay = buf.encode(&g);
    let (mut few, mut many) = (Vec::new(), Vec::new());
    obs::ops(&g, &lay, 0, &mut few);
    obs::ops(&g, &lay, obs::MAX_OPS, &mut many);
    assert!(many.len() <= obs::MAX_OPS);
    let base: Vec<_> = many.iter().filter(|o| o.kind != obs::op::PLAY_TOP).copied().collect();
    assert_eq!(base, few, "candidates replaced other ops");
    assert!(many.len() > few.len(), "no candidates");
}

#[test]
fn candidates_ignore_hidden_card_ids() {
    let mut n = 0;
    for s in 0..10 {
        walk(&format!("ID{s}"), 120, |g| {
            if g.phase != Phase::Round {
                return;
            }
            // Misprint makes every score depend on the chance scenarios.
            let mut a = g.clone();
            a.inject(&Inject::Joker(Joker::Misprint, Edition::None, false));
            let mut r = a.clone();
            for c in r.hand.iter_mut() {
                c.id += 100_000;
            }
            let top = obs::top_plays(&a, 8);
            assert!(top.iter().all(|c| c.min < c.max), "Misprint plays should vary");
            assert_eq!(top, obs::top_plays(&r, 8), "candidates depend on card ids");
            n += 1;
        });
    }
    assert!(n > 20, "only {n} rounds checked");
}
