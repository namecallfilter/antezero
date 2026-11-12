//! The build previews of shop and pack items.

use sim::data::{Back, BlindKind, Edition, HandType, Joker, Voucher};
use sim::game::{Game, Phase};
use sim::inject::Inject;
use sim::obs::{self};
use sim::{Action, Kind, num};

use crate::{
	G_HAS_PREVIEW, G_PREVIEW, best_candidate, best_legal_play, first_shop, obs_and_ops, tok,
	walk_build,
};

#[test]
#[expect(clippy::float_cmp, reason = "the test compares exact values")]
fn build_previews_do_not_reveal_hidden_state() {
	// Shops, packs and blind select: permuting everything the player cannot
	// see (deck order, card ids, the RNG, a face-down hand card's identity)
	// must leave the observation and every op feature unchanged. Misprint
	// and Crimson Heart make the previewed scoring draw random numbers.
	let mut n = 0;
	for s in 0..30 {
		walk_build(&format!("BP{s}"), 250, |g| {
			if !matches!(g.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect) {
				return;
			}
			let mut a = g.clone();
			if !a.hand.is_empty() && !a.deck.is_empty() {
				a.hand[0].face_down = true;
			}
			if n % 2 == 0 {
				a.inject(&Inject::Joker {
					joker: Joker::Misprint,
					edition: Edition::None,
					eternal: false,
				});
			}
			if n % 3 == 0 {
				a.boss_choice = BlindKind::FinalHeart;
			}
			let (oa, pa) = obs_and_ops(&a);
			assert_eq!(oa.gf[G_HAS_PREVIEW], 1.0, "no preview in {:?}", g.phase);
			let mut b = a.clone();
			b.deck.reverse();
			b.rng = sim::rng::PseudoRandom::new("ELSEWHERE");
			// A bijection that also scrambles the ids' order.
			for c in b
				.deck
				.iter_mut()
				.chain(b.hand.iter_mut())
				.chain(b.discard.iter_mut())
				.chain(b.jokers.iter_mut())
				.chain(b.consumables.iter_mut())
				.chain(b.shop_jokers.iter_mut())
				.chain(b.shop_vouchers.iter_mut())
				.chain(b.shop_boosters.iter_mut())
				.chain(b.pack_cards.iter_mut())
			{
				c.id = c.id.wrapping_mul(0x9e37_79b1) ^ 0x5bd1_e995;
			}
			if b.hand.first().is_some_and(|c| c.face_down) {
				let k = b.deck.len() / 2;
				std::mem::swap(&mut b.hand[0], &mut b.deck[k]);
				b.hand[0].face_down = true;
				b.deck[k].face_down = true;
			}
			let (ob, pb) = obs_and_ops(&b);
			oa.assert_same(&ob, &format!("hidden state permuted in {:?}", g.phase));
			assert_eq!(pa, pb, "op features reveal hidden state in {:?}", g.phase);
			n += 1;
		});
	}
	assert!(n > 100, "only {n} states checked");
}

#[test]
#[expect(
	clippy::float_cmp,
	reason = "the flags and zeroed columns are exact values"
)]
fn build_previews_are_consistent_and_bounded() {
	let item_zone = |z: i32| {
		matches!(
			z,
			obs::zone::JOKER
				| obs::zone::CONSUMABLE
				| obs::zone::SHOP_CARD
				| obs::zone::SHOP_VOUCHER
				| obs::zone::PACK
		)
	};
	let (mut previewed, mut states) = (0, 0);
	for s in 0..40 {
		walk_build(&format!("PB{s}"), 250, |g| {
			let (a, ops) = obs_and_ops(g);
			let (b, ops2) = obs_and_ops(g);
			a.assert_same(&b, "repeated encoding");
			assert_eq!(ops, ops2);
			let shopping = matches!(g.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect);
			assert_eq!(a.gf[G_HAS_PREVIEW], if shopping { 1.0 } else { 0.0 });
			for k in 0..2 {
				let (lr, r, p) = (
					a.gf[G_PREVIEW + 3 * k],
					a.gf[G_PREVIEW + 3 * k + 1],
					a.gf[G_PREVIEW + 3 * k + 2],
				);
				assert!(
					lr.is_finite()
						&& (-4.0..=4.0).contains(&lr)
						&& (0.0..=4.0).contains(&r)
						&& (0.0..=1.0).contains(&p),
					"{lr} {r} {p}"
				);
				if !shopping {
					assert_eq!((lr, r, p), (0.0, 0.0, 0.0));
				}
			}
			for i in 0..obs::MAX_TOKENS {
				let z = tok(&a, i)[0];
				let f = &a.tf[i * obs::TOK_FLOAT + obs::TOK_PREVIEW..(i + 1) * obs::TOK_FLOAT];
				if f[3] == 1.0 {
					assert!(shopping && item_zone(z), "preview on token {i} zone {z}");
					assert!(
						f[0].is_finite()
							&& (-4.0..=4.0).contains(&f[0])
							&& (0.0..=4.0).contains(&f[1])
							&& (0.0..=1.0).contains(&f[2]),
						"{f:?}"
					);
					previewed += 1;
				} else {
					assert_eq!(f, &[0.0; obs::PREVIEW_TOK], "token {i} zone {z}");
				}
			}
			for o in &ops {
				assert!(
					o.kind == obs::op::PLAY_TOP || o.feats == [0.0; obs::OP_FLOAT],
					"{o:?}"
				);
			}
			states += 1;
		});
	}
	assert!(
		states > 200 && previewed > 100,
		"{states} states, {previewed} previewed items"
	);
}

#[test]
fn item_previews_see_planets_for_other_hands() {
	// Every hand dealt is the same 8 cards (a Pair of Queens); a Pluto makes a
	// lone Ace the best play, so it must preview as a gain.
	use sim::data::{Planet, Suit};
	let mut g = Game::new_run("PLUTO", Back::Red, 1);
	g.phase = Phase::Shop;
	g.boss_choice = BlindKind::Small;
	g.dollars = 50.0;
	g.deck.clear();
	let specs = [
		(12, Suit::Spades),
		(12, Suit::Hearts),
		(14, Suit::Diamonds),
		(13, Suit::Clubs),
		(8, Suit::Clubs),
		(6, Suit::Spades),
		(4, Suit::Hearts),
		(2, Suit::Diamonds),
	];
	for (i, &(rank, suit)) in specs.iter().enumerate() {
		let mut c = sim::Card::new(100 + num::index32(i), sim::Kind::Playing);
		c.rank = rank;
		c.suit = suit;
		g.deck.push(c);
	}
	let h = &mut g.hands[HandType::HighCard as usize];
	(h.level, h.chips, h.mult) = (2.0, 15.0, 2.0);
	let mut p = sim::Card::new(999, sim::Kind::Planet(Planet::Pluto));
	(p.cost, p.base_cost) = (3.0, 3.0);
	g.shop_jokers.push(p);
	let mut cache = Vec::new();
	assert!(obs::build_preview(&g, &mut cache));
	let base = cache[0].cap_boss;
	let f = cache[0].item_feats(&g, obs::zone::SHOP_CARD, 0).unwrap();
	let mut after = g.clone();
	let h = &mut after.hands[HandType::HighCard as usize];
	(h.level, h.chips, h.mult) = (3.0, 25.0, 3.0);
	let fresh = obs::capacity(&after, BlindKind::Small, &[1, 2, 3, 4]);
	assert!(fresh.mean > base.mean, "{base:?} {fresh:?}");
	let want = num::feat(((fresh.hands * fresh.mean + 1.0) / (base.hands * base.mean + 1.0)).ln());
	assert!(
		(f[0] - want).abs() < 0.01,
		"Pluto's preview {f:?}, want gain {want}"
	);
}

#[test]
fn kickers_are_tried_for_blackboard_and_raised_fist() {
	// Blackboard and Raised Fist pay for what is not held, so the best play often carries kickers; the shop preview's
	// best play and the round's scored candidates must find it (the scoring cores alone miss it on most hands).
	for j in [Joker::Blackboard, Joker::RaisedFist] {
		let mut g = first_shop("KK", 0);
		g.inject(&Inject::Joker {
			joker: j,
			edition: Edition::None,
			eternal: false,
		});
		let (mut found, mut offered, mut kicker, n) = (0, 0, 0, 40u64);
		for s in 0..n {
			let v = obs::virtual_round(&g, BlindKind::Small, s);
			let (got, full) = (obs::best_play(&v), best_legal_play(&v));
			found += usize::from(got >= full * 0.999);
			offered += usize::from(best_candidate(&v) >= full * 0.999);
			let mut acts = Vec::new();
			v.legal_actions(&mut acts);
			let small = acts
				.iter()
				.filter_map(|a| {
					if let Action::Play(m) = *a {
						(m.count_ones() <= 2)
							.then(|| v.clone().preview_play(m).ok())
							.flatten()
					} else {
						None
					}
				})
				.fold(0.0, f64::max);
			kicker += usize::from(full > small * 1.5);
		}
		assert!(
			found as u64 >= n * 9 / 10,
			"{j:?}: the previewed play is the best legal one in only {found} of {n} hands"
		);
		assert!(
			offered as u64 >= n * 9 / 10,
			"{j:?}: the best legal play is among the round's scored candidates in only {offered} of {n} hands"
		);
		assert!(
			kicker >= 8,
			"{j:?}: kickers decide only {kicker} of {n} hands (the test needs hands where they do)"
		);
	}
}

#[test]
fn previews_place_copy_jokers_where_the_hand_scores_most() {
	// Blueprint copies its right neighbour and Brainstorm the leftmost joker: the previewed row must score as much as
	// the best of all its orders (three jokers: every order is tried here).
	for copy in [Joker::Blueprint, Joker::Brainstorm] {
		let mut g = first_shop("CP", 0);
		for j in [Joker::Joker, Joker::Cavendish, copy] {
			g.inject(&Inject::Joker {
				joker: j,
				edition: Edition::None,
				eternal: false,
			});
		}
		let mut matters = 0;
		for s in 0..16u64 {
			let v = obs::virtual_round(&g, BlindKind::Small, s);
			let got = obs::best_play(&v);
			let (mut best, mut worst) = (0.0f64, f64::INFINITY);
			for p in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [
				2, 1, 0,
			]] {
				let mut w = v.clone();
				w.jokers = p.iter().map(|&i| v.jokers[i].clone()).collect();
				for (i, c) in w.jokers.iter_mut().enumerate() {
					c.id = 5_000_000 + num::index32(i);
				}
				let x = obs::best_play(&w);
				(best, worst) = (best.max(x), worst.min(x));
			}
			assert!(
				got >= best * 0.9999,
				"{copy:?} seed {s}: previewed {got}, the best order scores {best}"
			);
			matters += usize::from(best > worst * 1.2);
		}
		assert!(
			matters >= 8,
			"{copy:?}: the order matters in only {matters} hands"
		);
	}
}

#[test]
#[expect(clippy::float_cmp, reason = "the previewed flag is exactly 1")]
fn hieroglyph_preview_shows_its_cost_and_the_nearer_target() {
	// Hieroglyph costs a hand and lowers the ante (the same hand is dealt with and without it).
	let mut g = first_shop("HG", 0);
	g.inject(&Inject::Joker {
		joker: Joker::Joker,
		edition: Edition::None,
		eternal: false,
	});
	let mut v = sim::Card::new(1 << 30, Kind::Voucher(Voucher::Hieroglyph));
	v.cost = 10.0;
	g.shop_vouchers.push(v);
	let mut cache = Vec::new();
	assert!(obs::build_preview(&g, &mut cache));
	let f = cache[0]
		.item_feats(&g, obs::zone::SHOP_VOUCHER, g.shop_vouchers.len() - 1)
		.unwrap();
	// 3 hands for 4: the gain is the cost (the last boss is no nearer); the ratio column is against the
	// ante-0 boss, a third of the ante-1 target.
	assert!(
		(f[0] - 0.75f32.ln()).abs() < 0.02 && f[3] == 1.0,
		"Hieroglyph's preview {f:?}"
	);
	let sold = cache[0]
		.item_feats(&g, obs::zone::JOKER, g.jokers.len() - 1)
		.unwrap()[1];
	assert!(
		f[1] > sold,
		"Hieroglyph's ratio {} is against the lower target (selling a joker leaves {sold})",
		f[1]
	);
}

#[test]
fn item_previews_count_the_growth_a_purchase_or_sale_triggers() {
	// Using a planet grows Constellation (x0.1) and selling a card grows Campfire (x0.25): a preview of the item
	// includes that growth, so it is worth exactly that much more than the same item without the joker.
	use sim::data::Planet;
	let gain = |with: Option<Joker>, zone: i32| {
		let mut g = first_shop("GR", 0);
		g.shop_jokers.clear();
		g.inject(&Inject::Joker {
			joker: Joker::Joker,
			edition: Edition::None,
			eternal: false,
		});
		if let Some(j) = with {
			g.inject(&Inject::Joker {
				joker: j,
				edition: Edition::None,
				eternal: false,
			});
		}
		let mut p = sim::Card::new(999, Kind::Planet(Planet::Pluto));
		(p.cost, p.base_cost) = (3.0, 3.0);
		g.shop_jokers.push(p);
		let mut cache = Vec::new();
		assert!(obs::build_preview(&g, &mut cache));
		cache[0].item_feats(&g, zone, 0).unwrap()[0]
	};
	let planet =
		gain(Some(Joker::Constellation), obs::zone::SHOP_CARD) - gain(None, obs::zone::SHOP_CARD);
	assert!(
		(planet - 1.1f32.ln()).abs() < 0.01,
		"Constellation adds {planet} to a planet's preview"
	);
	// Selling the +4 Mult joker: Campfire's x1.25 comes on top of the loss.
	let sale = gain(Some(Joker::Campfire), obs::zone::JOKER) - gain(None, obs::zone::JOKER);
	assert!(
		(sale - 1.25f32.ln()).abs() < 0.01,
		"Campfire adds {sale} to a sale's preview"
	);
}

#[test]
#[expect(
	clippy::float_cmp,
	reason = "the test looks for columns that changed at all"
)]
fn satellite_shows_its_payout() {
	// Satellite pays a dollar per distinct planet used this run; the count is on its card, so in its token.
	let mut g = first_shop("SAT", 0);
	g.inject(&Inject::Joker {
		joker: Joker::Satellite,
		edition: Edition::None,
		eternal: false,
	});
	let (a, _) = obs_and_ops(&g);
	g.planets_used = 0b101;
	let (b, _) = obs_and_ops(&g);
	let changed: Vec<usize> = (0..a.tf.len()).filter(|&i| a.tf[i] != b.tf[i]).collect();
	assert_eq!(changed.len(), 1, "one token column changes: {changed:?}");
	assert_eq!((a.tf[changed[0]], b.tf[changed[0]]), (0.0, 0.2));
	assert!(a.ti == b.ti && a.gi == b.gi && a.gf == b.gf);
}
