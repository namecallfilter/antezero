//! The ids and the `config.extra` values of the observation, matching
//! `RLBalatro`'s tables (ids are alphabetical by key within each key prefix).

use std::iter;
use std::sync::OnceLock;

use crate::card::{Ability, Card, Kind};
use crate::data::{
	ALL_BACKS, ALL_BLINDS, ALL_BOOSTERS, ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAGS,
	ALL_TAROTS, ALL_VOUCHERS, Back, BlindKind, Booster, CENTER_COUNT, Center, Enhancement, Joker,
	Spectral, TagKind, Tarot, Voucher,
};
use crate::game::Game;
use crate::num;

pub(super) const N_JOKER_KEYS: usize = ALL_JOKERS.len();
pub(super) const N_VOUCHER_KEYS: usize = ALL_VOUCHERS.len();
/// The center feature of a card shown face down.
pub(super) const CARD_BACK_CENTER_F: f32 = 300.0;

struct IdTables {
	center_id: [u16; CENTER_COUNT],
	back_id: [u16; ALL_BACKS.len()],
	blind_id: [u16; ALL_BLINDS.len()],
	tag_id: [u16; ALL_TAGS.len()],
	joker_by_alpha: [Joker; N_JOKER_KEYS],
	voucher_alpha: [usize; N_VOUCHER_KEYS],
	voucher_by_alpha: [Voucher; N_VOUCHER_KEYS],
}

/// A position in one of the sorted key lists, all of which are shorter than
/// 300, as an id.
fn id16(n: usize) -> u16 {
	u16::try_from(n).expect("the key lists hold fewer than 300 keys")
}

fn build_tables() -> IdTables {
	let mut t = IdTables {
		center_id: [0; CENTER_COUNT],
		back_id: [0; ALL_BACKS.len()],
		blind_id: [0; ALL_BLINDS.len()],
		tag_id: [0; ALL_TAGS.len()],
		joker_by_alpha: [Joker::Joker; N_JOKER_KEYS],
		voucher_alpha: [0; N_VOUCHER_KEYS],
		voucher_by_alpha: [Voucher::OverstockNorm; N_VOUCHER_KEYS],
	};

	let mut backs: Vec<Back> = ALL_BACKS.to_vec();
	backs.sort_by_key(|b| b.key());
	for (rank, b) in backs.into_iter().enumerate() {
		t.back_id[b as usize] = id16(rank);
	}

	let mut c_centers: Vec<Center> = ALL_TAROTS
		.iter()
		.map(|&x| Center::Tarot(x))
		.chain(ALL_PLANETS.iter().map(|&x| Center::Planet(x)))
		.chain(ALL_SPECTRALS.iter().map(|&x| Center::Spectral(x)))
		.chain(iter::once(Center::Enhanced(Enhancement::Base)))
		.collect();
	c_centers.sort_by_key(|c| c.key());
	for (rank, c) in c_centers.into_iter().enumerate() {
		t.center_id[c.index()] = id16(16 + rank);
	}

	let mut jokers: Vec<Joker> = ALL_JOKERS.to_vec();
	jokers.sort_by_key(|j| j.key());
	for (rank, j) in jokers.into_iter().enumerate() {
		t.joker_by_alpha[rank] = j;
		t.center_id[Center::Joker(j).index()] = id16(74 + rank);
	}

	let mut enhs: Vec<Enhancement> = Enhancement::POOL.to_vec();
	enhs.sort_by_key(|e| e.key());
	for (rank, e) in enhs.into_iter().enumerate() {
		t.center_id[Center::Enhanced(e).index()] = id16(224 + rank);
	}

	let mut boosters: Vec<Booster> = ALL_BOOSTERS.to_vec();
	boosters.sort_by_key(|b| b.key());
	for (rank, b) in boosters.into_iter().enumerate() {
		t.center_id[Center::Booster(b).index()] = id16(232 + rank);
	}

	let mut vouchers: Vec<Voucher> = ALL_VOUCHERS.to_vec();
	vouchers.sort_by_key(|v| v.key());
	for (rank, v) in vouchers.into_iter().enumerate() {
		t.voucher_alpha[v as usize] = rank;
		t.voucher_by_alpha[rank] = v;
		t.center_id[Center::Voucher(v).index()] = id16(267 + rank);
	}

	let mut blinds: Vec<BlindKind> = ALL_BLINDS.to_vec();
	blinds.sort_by_key(|b| b.key());
	for (rank, b) in blinds.into_iter().enumerate() {
		t.blind_id[b as usize] = id16(rank);
	}

	let mut tags: Vec<TagKind> = ALL_TAGS.to_vec();
	tags.sort_by_key(|tk| tk.key());
	for (rank, tk) in tags.into_iter().enumerate() {
		t.tag_id[tk as usize] = id16(rank);
	}
	t
}

#[inline]
fn tables() -> &'static IdTables {
	static TABLES: OnceLock<IdTables> = OnceLock::new();
	TABLES.get_or_init(build_tables)
}

/// The jokers in the order the observation numbers them (alphabetical by
/// key): the order of the `stickers` and `target` groups, and of the joker
/// center ids.
#[must_use]
pub fn jokers_by_key() -> &'static [Joker] {
	&tables().joker_by_alpha
}

/// The vouchers in the order the observation numbers them (alphabetical by
/// key): the order of the owned vouchers in the `round_ctx` group.
#[must_use]
pub fn vouchers_by_key() -> &'static [Voucher] {
	&tables().voucher_by_alpha
}

#[inline]
pub(super) fn center_id(c: Center) -> u16 {
	tables().center_id[c.index()]
}

#[inline]
pub(super) fn back_id(b: Back) -> u16 {
	tables().back_id[b as usize]
}

#[inline]
pub(super) fn blind_id(b: BlindKind) -> u16 {
	tables().blind_id[b as usize]
}

#[inline]
pub(super) fn tag_id(t: TagKind) -> u16 {
	tables().tag_id[t as usize]
}

#[inline]
pub(super) fn voucher_alpha(v: Voucher) -> usize {
	tables().voucher_alpha[v as usize]
}

/// Static + dynamic `(extra0, extra1)` features (`f[17]`, `f[18]`) matching
/// `RLBalatro`'s card features.
pub(super) fn card_extras(card: &Card) -> (f32, f32) {
	let ab = &card.ab;
	match card.kind {
		Kind::Booster(b) => (f32::from(b.def().extra), 0.0),
		Kind::Voucher(v) => (voucher_extra(v), 0.0),
		Kind::Planet(_) => (0.0, 0.0),
		Kind::Tarot(t) => match t {
			Tarot::Hermit => (20.0, 0.0),
			Tarot::Temperance => (50.0, 0.0),
			Tarot::WheelOfFortune => (4.0, 0.0),
			_ => (0.0, 0.0),
		},
		Kind::Spectral(s) => match s {
			Spectral::Ankh | Spectral::Cryptid | Spectral::Grim | Spectral::Hex => (2.0, 0.0),
			Spectral::Familiar => (3.0, 0.0),
			Spectral::Incantation => (4.0, 0.0),
			Spectral::Immolate => (5.0, 20.0),
			_ => (0.0, 0.0),
		},
		Kind::Joker(j) => joker_extras(j, ab),
	}
}

fn voucher_extra(v: Voucher) -> f32 {
	use Voucher as V;
	match v {
		V::OverstockNorm | V::OverstockPlus => 0.0,
		V::Antimatter => 15.0,
		V::Blank => 5.0,
		V::ClearanceSale => 25.0,
		V::Liquidation | V::SeedMoney => 50.0,
		V::MoneyTree => 100.0,
		V::CrystalBall | V::Telescope => 3.0,
		V::DirectorsCut | V::Retcon => 10.0,
		V::GlowUp | V::Illusion | V::MagicTrick | V::OmenGlobe => 4.0,
		V::Hone | V::RerollGlut | V::RerollSurplus => 2.0,
		V::Observatory => 1.5,
		V::PlanetMerchant | V::TarotMerchant => 2.4,
		V::PlanetTycoon | V::TarotTycoon => 8.0,
		V::Grabber
		| V::Hieroglyph
		| V::NachoTong
		| V::PaintBrush
		| V::Palette
		| V::Petroglyph
		| V::Recyclomancy
		| V::Wasteful => 1.0,
	}
}

fn joker_extras(j: Joker, ab: &Ability) -> (f32, f32) {
	use Joker as J;
	match j {
		J::Castle => (3.0, num::feat(ab.chips)),
		J::IceCream => (5.0, num::feat(ab.chips)),
		J::Runner => (15.0, num::feat(ab.chips)),
		J::Square => (4.0, num::feat(ab.chips)),
		J::Wee => (8.0, num::feat(ab.chips)),
		J::Rocket => (num::feat(ab.dollars), 2.0),
		J::Selzer => (num::feat(ab.extra), 0.0),
		J::TurtleBean => (1.0, num::feat(ab.extra)),
		J::TodoList => (4.0, ab.to_do.map_or(0.0, |h| f32::from(h as u8 + 1))),
		J::Bloodstone => (1.5, 2.0),
		J::Bootstraps | J::Faceless => (5.0, if j == J::Bootstraps { 2.0 } else { 3.0 }),
		J::Cavendish => (3.0, 1000.0),
		J::GreenJoker | J::ReservedParking => (1.0, if j == J::GreenJoker { 1.0 } else { 2.0 }),
		J::GrosMichel => (15.0, 6.0),
		J::Half => (20.0, 3.0),
		J::LoyaltyCard => (4.0, 5.0),
		J::Misprint => (23.0, 0.0),
		J::MysticSummit => (0.0, 15.0),
		J::Scholar => (20.0, 4.0),
		J::Stuntman => (250.0, 2.0),
		J::Troubadour => (-1.0, 2.0),
		J::WalkieTalkie => (10.0, 4.0),
		J::Yorick => (23.0, 1.0),
		_ => (joker_scalar_extra(j), 0.0),
	}
}

fn joker_scalar_extra(j: Joker) -> f32 {
	use Joker as J;
	match j {
		J::Ramen => 0.01,
		J::Constellation | J::Vampire => 0.1,
		J::Obelisk | J::SteelJoker => 0.2,
		J::Campfire | J::Hologram | J::LuckyCat | J::Throwback => 0.25,
		J::HitTheRoad | J::Madness => 0.5,
		J::Glass => 0.75,
		J::Caino
		| J::Chaos
		| J::Cloud9
		| J::Dusk
		| J::FortuneTeller
		| J::Gift
		| J::Hack
		| J::Marble
		| J::Mime
		| J::RideTheBus
		| J::RoughGem
		| J::Satellite
		| J::SockAndBuskin
		| J::Supernova
		| J::ToTheMoon => 1.0,
		J::Ancient | J::Baron | J::Baseball => 1.5,
		J::BlueJoker
		| J::Bull
		| J::Business
		| J::DelayedGrat
		| J::Flash
		| J::Hallucination
		| J::HangingChad
		| J::Idol
		| J::Invisible
		| J::Photograph
		| J::RiffRaff
		| J::SeeingDouble
		| J::Triboulet
		| J::Trousers => 2.0,
		J::Abstract
		| J::Acrobat
		| J::Blackboard
		| J::Burglar
		| J::CardSharp
		| J::DriversLicense
		| J::Egg
		| J::FlowerPot
		| J::GluttenousJoker
		| J::GreedyJoker
		| J::LustyJoker
		| J::RedCard
		| J::Trading
		| J::WrathfulJoker => 3.0,
		J::J8Ball
		| J::Burnt
		| J::Erosion
		| J::EvenSteven
		| J::Golden
		| J::Popcorn
		| J::Space
		| J::Ticket
		| J::Vagabond => 4.0,
		J::Hiker | J::Mail | J::Smiley => 5.0,
		J::OnyxAgate => 7.0,
		J::Fibonacci | J::Matador => 8.0,
		J::ShootTheMoon => 13.0,
		J::CreditCard => 20.0,
		J::Stone => 25.0,
		J::Banner | J::ScaryFace => 30.0,
		J::OddTodd => 31.0,
		J::Arrowhead => 50.0,
		_ => 0.0,
	}
}

/// Observation v2's `(joker_value, joker_count)` (`f[22]`, `f[23]`), matching
/// `RLBalatro`'s `joker_live`.
pub(super) fn joker_live(g: &Game, card: &Card) -> (f32, f32) {
	use Joker as J;
	let Kind::Joker(j) = card.kind else {
		return (0.0, 0.0);
	};
	let ab = &card.ab;
	let dollars = g.run.dollars.max(0.0);
	match j {
		J::Caino => (num::feat(ab.caino_xmult), 0.0),
		J::LoyaltyCard => {
			let n = num::feat(ab.loyalty_remaining);
			let v = if ab.loyalty_remaining == 0.0 {
				4.0
			} else {
				1.0
			};
			(v, n)
		}
		J::Invisible => {
			let v = num::feat(ab.invis_rounds);
			let n = num::feat((2.0 - ab.invis_rounds).max(0.0));
			(v, n)
		}
		J::Yorick => (num::feat(ab.x_mult), num::feat(ab.yorick_discards)),
		J::Selzer => (0.0, num::feat(ab.extra)),
		J::SteelJoker => (num::feat(1.0 + 0.2 * ab.steel_tally), 0.0),
		J::Stone => (num::feat(25.0 * ab.stone_tally), 0.0),
		J::Cloud9 => (num::feat(ab.nine_tally), 0.0),
		J::DriversLicense => {
			let v = num::feat(ab.driver_tally);
			let n = num::feat((16.0 - ab.driver_tally).max(0.0));
			(v, n)
		}
		J::FortuneTeller => (num::feat(g.history.consumable_usage_tarot), 0.0),
		J::Satellite => (num::feat(f64::from(g.history.planets_used.count())), 0.0),
		J::Throwback => (num::feat(1.0 + 0.25 * g.run.skips), 0.0),
		J::Abstract => (num::feat(3.0 * num::float(g.areas.jokers.len())), 0.0),
		J::BlueJoker => (num::feat(2.0 * num::float(g.areas.deck.len())), 0.0),
		J::Erosion => (
			num::feat(
				(4.0 * (g.run.starting_deck_size - num::float(g.areas.playing_card_count())))
					.max(0.0),
			),
			0.0,
		),
		J::Banner => (num::feat(30.0 * g.round.discards_left), 0.0),
		J::Bull => (num::feat(2.0 * dollars), 0.0),
		J::Bootstraps => (num::feat(2.0 * (dollars / 5.0).floor()), 0.0),
		J::IceCream | J::Runner | J::Square | J::Wee | J::Castle => (num::feat(ab.chips), 0.0),
		J::Golden => (4.0, 0.0),
		J::Rocket => (num::feat(ab.dollars), 0.0),
		J::DelayedGrat => (
			if g.round.discards_used == 0.0 {
				num::feat(2.0 * g.round.discards_left)
			} else {
				0.0
			},
			0.0,
		),
		J::MysticSummit => (
			if g.round.discards_left == 0.0 {
				15.0
			} else {
				0.0
			},
			0.0,
		),
		J::TurtleBean => (num::feat(ab.extra), 0.0),
		J::Swashbuckler
		| J::GreenJoker
		| J::RideTheBus
		| J::Trousers
		| J::RedCard
		| J::Flash
		| J::Popcorn
		| J::Ceremonial => (num::feat(ab.mult), 0.0),
		J::Stencil
		| J::Hologram
		| J::Vampire
		| J::Obelisk
		| J::LuckyCat
		| J::Constellation
		| J::Madness
		| J::Campfire
		| J::Glass
		| J::HitTheRoad
		| J::Ramen => (num::feat(ab.x_mult), 0.0),
		_ => (0.0, 0.0),
	}
}
