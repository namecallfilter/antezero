//! What a build can score over a round (`obs::capacity`) and the rounds the previews play.

use sim::data::{Back, BlindKind, Edition, Joker, Planet};
use sim::game::{Game, Phase};
use sim::inject::Inject;
use sim::obs::{self};
use sim::{Action, Kind, num};

use crate::first_shop;

#[test]
#[expect(clippy::float_cmp, reason = "hand counts and flags are exact values")]
fn capacity_orders_jokers_and_counts_grabber() {
	// At a shop: a x3 joker adds more than a +4 mult joker, which adds
	// something; Grabber adds a hand and nothing else.
	let mut g = Game::new_run("CAP", Back::Red, 1);
	g.inject(&Inject::Dollars(20.0));
	for s in 0.. {
		// Best-candidate play reaches the first shop on most seeds.
		g = Game::new_run(&format!("CAP{s}"), Back::Red, 1);
		g.step(Action::SelectBlind).unwrap();
		while g.phase == Phase::Round || g.phase == Phase::RoundEval {
			let a = if g.phase == Phase::Round {
				Action::Play(obs::top_plays(&g, 1)[0].mask)
			} else {
				Action::CashOut
			};
			g.step(a).unwrap();
		}
		if g.phase == Phase::Shop {
			break;
		}
	}
	let seeds = [1u64, 2, 3, 4];
	let boss = g.boss_choice;
	let base = obs::capacity(&g, boss, &seeds);
	assert!(base.mean > 0.0 && base.hands == 4.0 && base.chips > 0.0 && base.ratio() > 0.0);
	let mut plus = g.clone();
	plus.inject(&Inject::Joker {
		joker: Joker::Joker,
		edition: Edition::None,
		eternal: false,
	});
	let mut times = g.clone();
	times.inject(&Inject::Joker {
		joker: Joker::Cavendish,
		edition: Edition::None,
		eternal: false,
	});
	let (cp, ct) = (
		obs::capacity(&plus, boss, &seeds),
		obs::capacity(&times, boss, &seeds),
	);
	assert!(
		ct.mean > cp.mean && cp.mean > base.mean,
		"{base:?} {cp:?} {ct:?}"
	);
	let mut grab = g.clone();
	grab.inject(&Inject::Voucher(sim::data::Voucher::Grabber));
	let cg = obs::capacity(&grab, boss, &seeds);
	assert_eq!(cg.hands, 5.0);
	assert_eq!(cg.mean, base.mean, "the same hands are dealt");
	// As a shop item, Grabber's gain is its extra hand (same hands dealt).
	let mut shop = g.clone();
	let mut grabber = sim::Card::new(1 << 30, sim::Kind::Voucher(sim::data::Voucher::Grabber));
	grabber.cost = 10.0;
	shop.shop_vouchers.push(grabber);
	let mut cache = Vec::new();
	assert!(obs::build_preview(&shop, &mut cache));
	let f = cache[0]
		.item_feats(&shop, obs::zone::SHOP_VOUCHER, shop.shop_vouchers.len() - 1)
		.unwrap();
	assert!(
		(f[0] - (5.0f32 / 4.0).ln()).abs() < 0.01 && f[3] == 1.0,
		"Grabber's preview {f:?}"
	);
	// The Psychic: only five-card plays score, and they are previewed.
	for s in 0..8 {
		let v = obs::virtual_round(&g, BlindKind::Psychic, s);
		assert!(
			obs::best_play(&v) > 0.0,
			"no scoring play previewed against The Psychic (seed {s})"
		);
	}
	// The virtual round is a playable first decision against the boss.
	let v = obs::virtual_round(&g, boss, 7);
	assert_eq!(v.phase, Phase::Round);
	assert_eq!(v.hand.len(), num::trunc(g.hand_limit.limit));
	assert_eq!(v.blind.kind, Some(boss));
	assert!(v.hand.iter().all(|c| !c.face_down) && v.play.is_empty() && v.discard.is_empty());
	assert_eq!(v.deck.len() + v.hand.len(), g.playing_card_count());
}

#[test]
fn preview_hands_differ_between_games() {
	use std::collections::{HashMap, HashSet};

	// Two games at the same ante and round, with the same cards and the same boss, must not preview on the same
	// sampled hands (their histories differ): otherwise an item's preview is the same wrong number in every game.
	let mut by_boss: HashMap<BlindKind, HashSet<u64>> = HashMap::new();
	let mut from = 0;
	for _ in 0..64 {
		let mut g = first_shop("DV", from);
		from = g
			.rng
			.seed()
			.trim_start_matches("DV")
			.parse::<u64>()
			.unwrap()
			+ 1;
		g.inject(&Inject::Joker {
			joker: Joker::Jolly,
			edition: Edition::None,
			eternal: false,
		});
		let mut cache = Vec::new();
		assert!(obs::build_preview(&g, &mut cache));
		by_boss
			.entry(g.boss_choice)
			.or_default()
			.insert(cache[0].cap_boss.mean.to_bits());
	}
	// The Pillar debuffs the cards a game has played, so its capacities differ anyway.
	let most = by_boss
		.iter()
		.filter(|x| *x.0 != BlindKind::Pillar)
		.map(|x| x.1.len())
		.max()
		.unwrap();
	assert!(
		most >= 4,
		"at most {most} different capacities among the games facing the same boss"
	);
}

#[test]
#[expect(
	clippy::many_single_char_names,
	reason = "short names for the games being compared"
)]
#[expect(clippy::float_cmp, reason = "halving the target is exact")]
fn previews_start_the_round_as_burglar_and_chicot_do() {
	// Burglar trades the discards for three hands when the blind is selected, and Chicot disables the boss: the
	// previewed round must start like the real one.
	let mut g = first_shop("BG", 0);
	g.inject(&Inject::Joker {
		joker: Joker::Burglar,
		edition: Edition::None,
		eternal: false,
	});
	let v = obs::virtual_round(&g, BlindKind::Big, 3);
	let mut real = g.clone();
	real.step(Action::EndShop).unwrap();
	real.step(Action::SelectBlind).unwrap();
	assert_eq!(
		(v.hands_left, v.discards_left),
		(real.hands_left, real.discards_left)
	);
	assert_eq!((v.hands_left, v.discards_left), (7.0, 0.0));
	assert_eq!(obs::capacity(&g, BlindKind::Big, &[1, 2]).hands, 7.0);
	// A copy joker next to Burglar repeats it: ten hands, in the preview whatever the current order.
	let mut b = first_shop("BG", 0);
	for j in [Joker::Blueprint, Joker::Burglar] {
		b.inject(&Inject::Joker {
			joker: j,
			edition: Edition::None,
			eternal: false,
		});
	}
	let mut real = b.clone();
	real.step(Action::EndShop).unwrap();
	real.step(Action::SelectBlind).unwrap();
	assert_eq!(real.hands_left, 10.0);
	b.jokers.reverse();
	assert_eq!(obs::virtual_round(&b, BlindKind::Big, 3).hands_left, 10.0);
	let mut c = first_shop("BG", 0);
	let wall = obs::virtual_round(&c, BlindKind::Wall, 3).blind.chips;
	c.inject(&Inject::Joker {
		joker: Joker::Chicot,
		edition: Edition::None,
		eternal: false,
	});
	let v = obs::virtual_round(&c, BlindKind::Wall, 3);
	assert!(
		v.blind.disabled && v.blind.chips == wall / 2.0,
		"The Wall with Chicot: {} chips (without: {wall})",
		v.blind.chips
	);
	assert!(!obs::virtual_round(&c, BlindKind::Big, 3).blind.disabled);
	// As a shop item against The Wall, Chicot's gain counts the halved target.
	let mut shop = first_shop("BG", 0);
	shop.boss_choice = BlindKind::Wall;
	shop.dollars = 50.0;
	shop.shop_jokers.clear();
	shop.shop_jokers.push(c.jokers.last().unwrap().clone());
	let mut cache = Vec::new();
	assert!(obs::build_preview(&shop, &mut cache));
	let f = cache[0].item_feats(&shop, obs::zone::SHOP_CARD, 0).unwrap();
	assert!(
		(f[0] - 2f32.ln()).abs() < 0.02,
		"Chicot's preview against The Wall {f:?}"
	);
}

#[test]
#[expect(clippy::float_cmp, reason = "hand counts are exact values")]
fn capacity_against_the_eye_and_the_mouth_is_a_whole_round() {
	// The Eye scores no hand type twice and The Mouth only the first hand's type, so the round is worth less than
	// its hands' best plays added up; one hand alone is worth the same.
	let (mut eye_less, mut mouth_less) = (0, 0);
	for s in 0..12u64 {
		let mut g = first_shop("EM", s * 7);
		g.inject(&Inject::Joker {
			joker: Joker::Jolly,
			edition: Edition::None,
			eternal: false,
		});
		let seeds = [s + 1, s + 2, s + 3, s + 4];
		let free = obs::capacity(&g, BlindKind::Small, &seeds).mean;
		for (boss, less) in [
			(BlindKind::Eye, &mut eye_less),
			(BlindKind::Mouth, &mut mouth_less),
		] {
			let c = obs::capacity(&g, boss, &seeds);
			assert!(
				c.mean <= free + 1e-9,
				"{boss:?}: {} above the unrestricted {free}",
				c.mean
			);
			*less += usize::from(c.mean < free * 0.98);
			// Four hands, four sampled hands: one whole round, which clears exactly when its capacity covers the target.
			assert_eq!(
				c.p_clear,
				f64::from(u8::from(c.ratio() >= 1.0)),
				"{boss:?}: clear fraction {} at ratio {}",
				c.p_clear,
				c.ratio()
			);
			// A longer round (Burglar: seven hands) keeps using up hand types past the four sampled hands.
			let mut long = g.clone();
			long.inject(&Inject::Joker {
				joker: Joker::Burglar,
				edition: Edition::None,
				eternal: false,
			});
			let (l, f) = (
				obs::capacity(&long, boss, &seeds),
				obs::capacity(&long, BlindKind::Small, &seeds),
			);
			assert!(
				l.hands == 7.0
					&& l.mean <= f.mean + 1e-9
					&& (boss != BlindKind::Eye || l.mean <= c.mean + 1e-9),
				"{boss:?}: seven hands {l:?} vs four {c:?}"
			);
			assert_eq!(
				obs::capacity(&g, boss, &seeds[..1]).mean,
				obs::capacity(&g, BlindKind::Small, &seeds[..1]).mean,
				"{boss:?}: one hand"
			);
			// A round of one hand has nothing to restrict, however many hands are sampled.
			let mut one = g.clone();
			one.rr_hands = 1.0;
			assert_eq!(
				obs::capacity(&one, boss, &seeds).mean,
				obs::capacity(&one, BlindKind::Small, &seeds).mean,
				"{boss:?}: rounds of one hand"
			);
		}
	}
	assert!(
		eye_less >= 9 && mouth_less >= 6,
		"the restriction costs something in only {eye_less} (Eye) and {mouth_less} (Mouth) of 12 builds"
	);
}

#[test]
fn item_capacity_against_the_eye_is_on_the_rounds_scale() {
	// An item's capacity columns are on the round's scale: against The Eye its ratio is lower than with the boss
	// disabled (Chicot owned, which scores nothing itself) by what the rule costs the build.
	let mut scale = 0;
	for s in 0..12u64 {
		let item = |chicot: bool| {
			let mut g = first_shop("EM", s * 7);
			g.boss_choice = BlindKind::Eye;
			g.dollars = 50.0;
			g.inject(&Inject::Joker {
				joker: Joker::Jolly,
				edition: Edition::None,
				eternal: false,
			});
			if chicot {
				g.inject(&Inject::Joker {
					joker: Joker::Chicot,
					edition: Edition::None,
					eternal: false,
				});
			}
			g.shop_jokers.clear();
			let mut p = sim::Card::new(999, Kind::Planet(Planet::Pluto));
			(p.cost, p.base_cost) = (3.0, 3.0);
			g.shop_jokers.push(p);
			let mut cache = Vec::new();
			assert!(obs::build_preview(&g, &mut cache));
			(
				f64::from(cache[0].item_feats(&g, obs::zone::SHOP_CARD, 0).unwrap()[1]),
				cache[0].cap_boss.mean,
			)
		};
		let ((r_eye, m_eye), (r_free, m_free)) = (item(false), item(true));
		if r_free < 3.9 && m_eye < m_free * 0.98 {
			assert!(
				(r_eye / r_free - m_eye / m_free).abs() < 0.01,
				"item ratio {r_eye} vs {r_free} with the boss disabled; the build's capacity {m_eye} vs {m_free}"
			);
			scale += 1;
		}
	}
	assert!(
		scale >= 5,
		"only {scale} builds where The Eye costs something"
	);
}
