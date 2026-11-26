//! The joker-order gain on the joker-move ops and the labels at played hands.

use sim::data::{Back, Edition, Joker};
use sim::game::{Game, Phase, Scratch};
use sim::inject::Inject;
use sim::obs::{self, ORDER_JOKERS, PREVIEW_SAMPLES, Scenarios};
use sim::rng::{PseudoRandom, Xorshift};
use sim::{Action, CardMask, Stake, num};

use crate::obs_and_ops;

/// Jokers whose place in the row changes the score (additive and
/// multiplicative effects, copies, retriggers), ones that draw random
/// numbers, and ones that pay or create cards when a hand is played.
const POOL: [Joker; 24] = [
	Joker::Joker,
	Joker::Cavendish,
	Joker::Blueprint,
	Joker::Brainstorm,
	Joker::Misprint,
	Joker::Bloodstone,
	Joker::Business,
	Joker::J8Ball,
	Joker::Vagabond,
	Joker::Dna,
	Joker::Superposition,
	Joker::SixthSense,
	Joker::Photograph,
	Joker::HangingChad,
	Joker::Banner,
	Joker::Abstract,
	Joker::Stencil,
	Joker::Baron,
	Joker::Half,
	Joker::RoughGem,
	Joker::ToTheMoon,
	Joker::Ramen,
	Joker::GreenJoker,
	Joker::Jolly,
];

/// Calls `f` at every decision of `games` games (on every stake) that start
/// with 1 to 7 jokers of `POOL`, keep them and play the top candidate most
/// of the time (joker moves, other actions and editions mixed in).
fn walk_jokers(prefix: &str, games: u8, steps: usize, mut f: impl FnMut(&Game)) {
	for game in 0..games {
		let stake = Stake::ALL[usize::from(game) % Stake::ALL.len()];
		let mut g = Game::new_run(&format!("{prefix}{game}"), Back::Red, stake);
		let mut rng = Xorshift::new(
			0x9e37_79b9_7f4a_7c15 ^ (u64::from(game) + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9),
		);
		let mut roll = move || rng.next_u64() >> 8;
		for _ in 0..=num::pick(roll(), 7) {
			g.inject(Inject::Joker {
				joker: POOL[num::pick(roll(), POOL.len())],
				edition: [None, None, Some(Edition::Holo), Some(Edition::Polychrome)]
					[num::pick(roll(), 4)],
				eternal: false,
			});
		}
		let mut acts = Vec::new();
		for _ in 0..steps {
			if g.is_terminal() {
				break;
			}
			f(&g);
			let roll = roll();
			let slots = g.areas.jokers.len();
			let top = obs::top_plays(&g, 1);
			let action = if roll.is_multiple_of(5) && slots >= 2 {
				let from = num::pick(roll / 5, slots);
				Action::MoveJoker {
					from: num::byte(from),
					to: num::byte((from + 1 + num::pick(roll / 64, slots - 1)) % slots),
				}
			} else if let Some(c) = top.first().filter(|_| !roll.is_multiple_of(11)) {
				Action::Play(c.mask)
			} else {
				g.legal_actions(&mut acts);
				acts.retain(|a| !matches!(a, Action::SellJoker(_)));
				acts[num::pick(roll, acts.len())]
			};
			g.step(action).expect("a legal action steps");
		}
	}
}

fn gain(o: &obs::Op) -> f32 {
	o.feats[obs::OP_ORDER_GAIN]
}

/// The play of the top candidate op, if `ops` starts with one.
fn top_play(ops: &[obs::Op]) -> Option<CardMask> {
	ops.first()
		.filter(|o| o.kind == obs::OpKind::PlayTop)
		.and_then(|o| o.cards)
		.map(|c| c.required)
}

/// The chips `g` still needs to clear its blind.
fn need(g: &Game) -> f64 {
	g.blind.chips - g.round.chips
}

/// What `obs::order_gain` and `obs::order_label` require of a state.
fn scored(g: &Game) -> bool {
	g.flow.phase == Phase::Round
		&& (2..=ORDER_JOKERS).contains(&g.areas.jokers.len())
		&& g.flow.rearranges < obs::MAX_REARRANGES
		&& !obs::hidden_scoring_state(&g.areas)
		&& !g.areas.jokers.iter().any(|c| c.is_joker(Joker::Ceremonial))
}

/// What a play leaves besides its score: money, jokers, consumables, cards
/// in hand and deck.
fn side(g: &Game) -> (u64, usize, usize, usize) {
	(
		g.run.dollars.to_bits(),
		g.areas.jokers.len(),
		g.areas.consumables.len(),
		g.areas.hand.cards.len() + g.areas.deck.len(),
	)
}

/// A play in one scenario.
#[derive(Clone, Copy)]
struct Played {
	score: f64,
	/// Whether the score drew a random number.
	chance: bool,
	/// What the hand leaves (see [`side`]).
	side: (u64, usize, usize, usize),
}

/// Plays `mask` in scenario `k` after the joker move `mv` (from, to), through
/// the game's own actions.
fn played(sc: &Scenarios, k: usize, mask: CardMask, mv: Option<(usize, usize)>) -> Played {
	let mut world = sc.view(k).clone();
	if let Some((from, to)) = mv {
		world
			.step(Action::MoveJoker {
				from: num::byte(from),
				to: num::byte(to),
			})
			.expect("a joker moves to another slot");
	}
	let alt = sc.alt(k);
	let preview =
		obs::preview_score(&mut Scratch::default(), &world, mask, alt).expect("the play is legal");
	world.rng.clone_from(alt);
	world.preview_hand(mask).expect("the play is legal");
	Played {
		score: preview.score,
		chance: preview.chance,
		side: side(&world),
	}
}

fn ratio(capped: f64, base: f64) -> f64 {
	(capped + 1.0) / (base + 1.0)
}

#[test]
fn order_gain_is_zero_outside_its_cases_and_bounded() {
	let scale = num::feat(obs::ORDER_GAIN_SCALE);
	let least = scale * 1.01f32.ln() - 1e-3;
	let (mut states, mut gaining, mut excluded) = (0, 0, 0);
	walk_jokers("OZ", 60, 160, |g| {
		let (_, ops) = obs_and_ops(g);
		let moves: Vec<&obs::Op> = ops
			.iter()
			.filter(|o| o.kind == obs::OpKind::MoveJoker)
			.collect();
		for o in &ops {
			assert!(
				o.kind == obs::OpKind::MoveJoker || gain(o) == 0.0,
				"gain on {o:?}"
			);
		}
		for o in &moves {
			let x = gain(o);
			assert!(x == 0.0 || (least..=2.0 * scale).contains(&x), "gain {x}");
		}
		let sc = Scenarios::new(g, 1);
		let open = scored(g)
			&& top_play(&ops).is_some_and(|top| played(&sc, 0, top, None).score < need(g));
		if !open {
			assert!(
				moves.iter().all(|o| gain(o) == 0.0),
				"gain outside the stated cases in {:?}",
				g.flow.phase
			);
		}
		states += 1;
		if moves.iter().all(|o| gain(o) == 0.0) {
			return;
		}
		gaining += 1;
		// The same state with one condition broken has no gain.
		let mut broken: Vec<Game> = (0..4).map(|_| g.clone()).collect();
		broken[0].areas.jokers[0].core.face_down = true;
		broken[1].areas.hand.cards[0].core.face_down = true;
		broken[2].round.chips = g.blind.chips;
		broken[3].inject(Inject::joker(Joker::Ceremonial));
		for (k, b) in broken.iter().enumerate() {
			let (_, ops) = obs_and_ops(b);
			assert!(
				ops.iter().all(|o| gain(o) == 0.0),
				"gain with condition {k} broken"
			);
			excluded += 1;
		}
	});
	assert!(
		states > 2000 && gaining > 100 && excluded > 400,
		"{states} states, {gaining} with a gain, {excluded} exclusions"
	);
}

#[test]
fn order_gain_matches_a_brute_force_over_every_move() {
	let (mut states, mut gaining, mut dirty) = (0, 0, 0);
	walk_jokers("OB", 60, 160, |g| {
		let (_, ops) = obs_and_ops(g);
		let Some(mask) = top_play(&ops) else {
			return;
		};
		if !scored(g) {
			return;
		}
		let sc = Scenarios::new(g, 1);
		let need = need(g);
		let unmoved = played(&sc, 0, mask, None);
		let base = unmoved.score.min(need);
		let n = g.areas.jokers.len();
		for o in ops.iter().filter(|o| o.kind == obs::OpKind::MoveJoker) {
			let from = usize::from(o.index);
			let mut best = base;
			for to in (0..n).filter(|&to| to != from) {
				let moved = played(&sc, 0, mask, Some((from, to)));
				if moved.side == unmoved.side {
					best = best.max(moved.score.min(need));
				} else if ratio(moved.score.min(need), base) >= 1.01 {
					dirty += 1;
				}
			}
			let want = if unmoved.score < need && ratio(best, base) >= 1.01 {
				num::feat(obs::ORDER_GAIN_SCALE * ratio(best, base).ln().min(2.0))
			} else {
				0.0
			};
			assert_eq!(
				gain(o),
				want,
				"joker {from} of {n}, base {}, need {need}",
				unmoved.score
			);
			gaining += usize::from(want > 0.0);
		}
		states += 1;
	});
	assert!(
		states > 300 && gaining > 100 && dirty > 5,
		"{states} states, {gaining} gains, {dirty} gaining moves with other effects"
	);
}

/// One scenario's plays: in the current order, and after each single move.
struct Run {
	unmoved: Played,
	moved: Vec<Played>,
}

/// The label of playing `mask` in `g`, from every single move played out.
fn brute_label(g: &Game, mask: CardMask) -> Option<[u8; ORDER_JOKERS]> {
	if !scored(g) || g.check(Action::Play(mask)).is_err() {
		return None;
	}
	let sc = Scenarios::new(g, PREVIEW_SAMPLES);
	let n = g.areas.jokers.len();
	let need = need(g);
	let moves: Vec<(usize, usize)> = (0..n)
		.flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
		.collect();
	let run = |k: usize| Run {
		unmoved: played(&sc, k, mask, None),
		moved: moves
			.iter()
			.map(|&mv| played(&sc, k, mask, Some(mv)))
			.collect(),
	};
	let mut runs = vec![run(0)];
	if runs[0].unmoved.chance || runs[0].moved.iter().any(|p| p.chance) {
		runs.extend((1..PREVIEW_SAMPLES).map(run));
	} else if runs[0].unmoved.score >= need {
		return None;
	}
	let count = num::float(runs.len());
	let capped = |p: &Played| p.score.min(need);
	let base = runs.iter().map(|r| capped(&r.unmoved)).sum::<f64>() / count;
	if base >= need {
		return None;
	}
	let gains: Vec<Option<f64>> = (0..moves.len())
		.map(|m| {
			let s = runs.iter().map(|r| capped(&r.moved[m])).sum::<f64>() / count;
			let ok = runs.iter().all(|r| {
				capped(&r.moved[m]) >= capped(&r.unmoved) && r.moved[m].side == r.unmoved.side
			});
			(ok && ratio(s, base) >= 1.01).then_some(s)
		})
		.collect();
	let best = gains.iter().flatten().copied().fold(0.0, f64::max);
	let mut dest = [0u8; ORDER_JOKERS];
	for (&(from, to), s) in moves.iter().zip(&gains) {
		if s.is_some_and(|s| s >= 0.999 * best) {
			dest[from] |= 1 << to;
		}
	}
	Some(dest)
}

#[test]
fn order_labels_match_a_brute_force_over_every_move() {
	let (mut labels, mut positive, mut chance, mut tied) = (0, 0, 0, 0);
	walk_jokers("OL", 60, 160, |g| {
		if g.flow.phase != Phase::Round {
			return;
		}
		// The top candidate and a weaker play (which more often stays below the blind).
		let top = obs::top_plays(g, 8);
		for c in top.iter().take(1).chain(top.last()) {
			let got = obs::order_label(g, c.mask, obs::MAX_REARRANGES);
			assert_eq!(got, brute_label(g, c.mask), "play {:b}", c.mask.bits());
			let Some(dest) = got else { continue };
			labels += 1;
			positive += usize::from(dest != [0; ORDER_JOKERS]);
			chance += usize::from(c.min < c.max);
			tied += usize::from(dest.iter().filter(|&&d| d != 0).count() > 1);
		}
	});
	assert!(
		labels > 300 && positive > 60 && chance > 30 && tied > 20,
		"{labels} labels, {positive} positive, {chance} on chance plays, {tied} with tied sources"
	);
}

#[test]
fn order_gains_and_labels_do_not_reveal_hidden_state() {
	let (mut states, mut gaining, mut positive) = (0, 0, 0);
	walk_jokers("OH", 40, 160, |g| {
		if g.flow.phase != Phase::Round {
			return;
		}
		let (_, ops) = obs_and_ops(g);
		let mut permuted = g.clone();
		permuted.areas.deck.reverse();
		permuted.rng = PseudoRandom::new("ELSEWHERE");
		let areas = &mut permuted.areas;
		for c in areas.deck.iter_mut().chain(&mut areas.hand.cards) {
			c.core.id += 100_000;
		}
		let mut resampled = g.clone();
		resampled.resample_hidden("OTHERWORLD");
		let mask = obs::top_plays(g, 8).last().map(|c| c.mask);
		for (what, h) in [("permuted", &permuted), ("resampled", &resampled)] {
			assert_eq!(ops, obs_and_ops(h).1, "op features, hidden state {what}");
			if let Some(m) = mask {
				assert_eq!(
					obs::order_label(g, m, obs::MAX_REARRANGES),
					obs::order_label(h, m, obs::MAX_REARRANGES),
					"label, hidden state {what}"
				);
			}
		}
		states += 1;
		gaining += usize::from(ops.iter().any(|o| gain(o) > 0.0));
		positive += usize::from(mask.is_some_and(|m| {
			obs::order_label(g, m, obs::MAX_REARRANGES).is_some_and(|d| d != [0; ORDER_JOKERS])
		}));
	});
	assert!(
		states > 500 && gaining > 50 && positive > 30,
		"{states} states, {gaining} with a gain, {positive} positive labels"
	);
}

#[test]
fn preview_hand_leaves_what_the_played_hand_leaves() {
	// While the round goes on, a full step only draws cards after the hand:
	// the score, the money and the cards must be the preview's.
	let (mut checked, mut paid, mut created) = (0, 0, 0);
	let mut scratch = Scratch::default();
	walk_jokers("PH", 40, 160, |g| {
		if g.flow.phase != Phase::Round {
			return;
		}
		for c in obs::top_plays(g, 3) {
			let alt = PseudoRandom::new("HAND");
			let mut preview = g.clone();
			preview.rng = alt.clone();
			let score = preview.preview_hand(c.mask).unwrap();
			assert_eq!(
				score,
				obs::preview_score(&mut scratch, g, c.mask, &alt)
					.unwrap()
					.score,
				"play {:b}",
				c.mask.bits()
			);
			let mut full = g.clone();
			full.rng = alt;
			full.step(Action::Play(c.mask)).unwrap();
			if full.flow.phase != Phase::Round {
				continue;
			}
			assert_eq!(full.round.chips, g.round.chips + score, "chips");
			assert_eq!(
				side(&preview),
				side(&full),
				"play {:b} with {:?}",
				c.mask.bits(),
				g.areas.jokers.iter().map(|j| j.kind).collect::<Vec<_>>()
			);
			checked += 1;
			paid += usize::from(preview.run.dollars != g.run.dollars);
			created += usize::from(preview.areas.consumables.len() != g.areas.consumables.len());
		}
	});
	assert!(
		checked > 500 && paid > 30 && created > 10,
		"{checked} plays, {paid} that paid, {created} that created a consumable"
	);
}
