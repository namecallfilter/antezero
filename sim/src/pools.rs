//! Pools and seeded sampling (`common_events.lua`).

use crate::card::{Card, Kind};
use crate::card_lifecycle;
use crate::card_lifecycle::Pricing;
use crate::data::{
	ALL_BLINDS, ALL_BOOSTERS, ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAGS, ALL_TAROTS,
	ALL_VOUCHERS, BlindKind, Booster, Boss, Center, Edition, Enhancement, Joker, P_CARDS_SORTED,
	Planet, PoolFlag, RARITY_POOL_1, RARITY_POOL_2, RARITY_POOL_3, RARITY_POOL_4, Rank, Rarity,
	Spectral, Suit, TagKind, Tarot, Voucher,
};
use crate::enum_sets::CenterSet;
use crate::event_queue::EditionTiming;
use crate::game::{Area, Areas, BlindSelect, CardSet, Game, HandLevels};
use crate::playing_card::PlayingCard;
use crate::rng::{Key, PseudoRandom};
use crate::run::{History, Modifiers, Run};
use crate::shop::Shop;

/// Which editions a poll can give (`poll_edition`'s `no_neg`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditionPool {
	Any,
	NoNegative,
}

/// `poll_edition(key, mod, no_neg)`.
pub(crate) fn poll_edition(
	rng: &mut PseudoRandom, edition_rate: f64, key: &Key, m: f64, pool: EditionPool,
) -> Option<Edition> {
	let p = rng.random(key);
	if p > 1.0 - 0.003 * m && pool == EditionPool::Any {
		Some(Edition::Negative)
	} else if p > 1.0 - 0.006 * edition_rate * m {
		Some(Edition::Polychrome)
	} else if p > 1.0 - 0.02 * edition_rate * m {
		Some(Edition::Holo)
	} else if p > 1.0 - 0.04 * edition_rate * m {
		Some(Edition::Foil)
	} else {
		None
	}
}

/// `poll_edition(key, nil, true, true)`: the guaranteed poll of Aura and
/// The Wheel of Fortune, which never gives Negative (its Foil threshold,
/// `1 - 0.04 * 25`, is exactly 0).
pub(crate) fn poll_guaranteed_edition(rng: &mut PseudoRandom, key: &Key) -> Option<Edition> {
	poll_edition(rng, 1.0, key, 25.0, EditionPool::NoNegative)
}

/// What keeps a card out of the pools: having been used
/// (`G.GAME.used_jokers`) unless Showman is owned, and, for the jokers whose
/// pool flag names it, whether Gros Michel is extinct.
#[derive(Clone, Copy, Debug)]
struct PoolFilter {
	used_jokers: CenterSet,
	showman: bool,
	gros_michel_extinct: bool,
}

impl PoolFilter {
	fn new(areas: &Areas, history: &History) -> PoolFilter {
		PoolFilter {
			used_jokers: history.used_jokers,
			showman: areas.has_joker(Joker::RingMaster),
			gros_michel_extinct: history.gros_michel_extinct,
		}
	}

	/// Whether `c` is left out for having been used.
	fn is_used(self, c: Center) -> bool {
		self.used_jokers.contains(c) && !self.showman
	}
}

/// A pool in the game's order. `None` is an entry `get_current_pool` marks
/// `'UNAVAILABLE'`: it keeps its slot, and landing on it resamples.
struct Pool {
	entries: [Option<Kind>; Pool::ROOM],
	len: usize,
}

impl Pool {
	/// Room for any pool: the joker pools are parts of the joker table, and
	/// the consumable tables are smaller.
	const ROOM: usize = ALL_JOKERS.len();

	fn new() -> Pool {
		Pool {
			entries: [None; Pool::ROOM],
			len: 0,
		}
	}

	fn push(&mut self, kind: Kind, available: bool) {
		if available {
			self.entries[self.len] = Some(kind);
		}
		self.len += 1;
	}
}

/// `pseudorandom_element(pool, pseudoseed(key))` with the game's resample
/// loop: an unavailable entry (`None`) draws again from
/// `key .. '_resample' .. n`. A pool without an available entry gives
/// `fallback`, after the draw the game makes from its one-entry fallback
/// pool with the same key.
fn sample_or<T: Copy>(rng: &mut PseudoRandom, key: Key, pool: &[Option<T>], fallback: T) -> T {
	if pool.iter().all(Option::is_none) {
		rng.element(&key, 1);
		return fallback;
	}
	let mut i = rng.element(&key, pool.len());
	let mut it = 1;
	loop {
		if let Some(entry) = pool[i] {
			return entry;
		}
		it += 1;
		let mut k = key;
		k.push("_resample").push_int(it);
		i = rng.element(&k, pool.len());
	}
}

/// The card sets with a pool to sample, with the roll that decides a
/// joker's rarity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PoolSet {
	Joker(RarityRoll),
	Tarot,
	Planet,
	Spectral,
}

impl PoolSet {
	/// The pool of `set`, a joker's rarity drawn from the seeded stream.
	fn seeded(set: CardSet) -> PoolSet {
		match set {
			CardSet::Joker => PoolSet::Joker(RarityRoll::Seeded),
			CardSet::Tarot => PoolSet::Tarot,
			CardSet::Planet => PoolSet::Planet,
			CardSet::Spectral => PoolSet::Spectral,
		}
	}

	fn card_set(self) -> CardSet {
		match self {
			PoolSet::Joker(_) => CardSet::Joker,
			PoolSet::Tarot => CardSet::Tarot,
			PoolSet::Planet => CardSet::Planet,
			PoolSet::Spectral => CardSet::Spectral,
		}
	}
}

/// The joker pool of the rolled rarity; returns the rarity.
fn joker_pool(
	areas: &Areas, ante: i32, filter: PoolFilter, rng: &mut PseudoRandom, roll: RarityRoll,
	append: &str, pool: &mut Pool,
) -> Rarity {
	let r = match roll {
		RarityRoll::Fixed(r) => r,
		RarityRoll::Legendary { rolled: false } => 0.0,
		RarityRoll::Seeded | RarityRoll::Legendary { rolled: true } => {
			let mut k = Key::from("rarity");
			k.push_int(i64::from(ante)).push(append);
			rng.random(&k)
		}
	};
	let rarity = if matches!(roll, RarityRoll::Legendary { .. }) {
		Rarity::Legendary
	} else if r > 0.95 {
		Rarity::Rare
	} else if r > 0.7 {
		Rarity::Uncommon
	} else {
		Rarity::Common
	};
	let src: &[Joker] = match rarity {
		Rarity::Common => &RARITY_POOL_1,
		Rarity::Uncommon => &RARITY_POOL_2,
		Rarity::Rare => &RARITY_POOL_3,
		Rarity::Legendary => &RARITY_POOL_4,
	};
	for &j in src {
		let mut add = false;
		if !filter.is_used(Center::Joker(j)) {
			let d = j.def();
			add = match d.enhancement_gate {
				Some(gate) => areas.playing_cards().any(|c| c.enhancement == gate),
				None => true,
			};
			match j.pool_flag() {
				Some(PoolFlag::GrosMichelAlive) if filter.gros_michel_extinct => add = false,
				Some(PoolFlag::GrosMichelExtinct) if !filter.gros_michel_extinct => add = false,
				_ => {}
			}
		}
		pool.push(Kind::Joker(j), add);
	}
	rarity
}

/// Builds the pool of `set` and samples it, including the resample loop
/// (`get_current_pool` + `pseudorandom_element`).
fn sample_pool(
	areas: &Areas, ante: i32, filter: PoolFilter, hands: &HandLevels, rng: &mut PseudoRandom,
	set: PoolSet, append: &str,
) -> Kind {
	let mut pool = Pool::new();
	let mut key = Key::new();
	match set {
		PoolSet::Joker(roll) => {
			let rarity = joker_pool(areas, ante, filter, rng, roll, append, &mut pool);
			key.push("Joker").push_int(i64::from(rarity.id()));
			if !matches!(roll, RarityRoll::Legendary { .. }) {
				key.push(append);
			}
		}
		PoolSet::Tarot => {
			key.push("Tarot").push(append);
			for &t in &ALL_TAROTS {
				pool.push(Kind::Tarot(t), !filter.is_used(Center::Tarot(t)));
			}
		}
		PoolSet::Planet => {
			key.push("Planet").push(append);
			for &p in &ALL_PLANETS {
				let mut add = false;
				if !filter.is_used(Center::Planet(p)) {
					let softlock = matches!(p, Planet::PlanetX | Planet::Ceres | Planet::Eris);
					add = !softlock || hands[p.hand()].played > 0.0;
				}
				pool.push(Kind::Planet(p), add);
			}
		}
		PoolSet::Spectral => {
			key.push("Spectral").push(append);
			for &s in &ALL_SPECTRALS {
				let add = !filter.is_used(Center::Spectral(s))
					&& !matches!(s, Spectral::Soul | Spectral::BlackHole);
				pool.push(Kind::Spectral(s), add);
			}
		}
	}
	if !matches!(set, PoolSet::Joker(RarityRoll::Legendary { .. })) {
		key.push_int(i64::from(ante));
	}
	let fallback = match set {
		PoolSet::Joker(_) => Kind::Joker(Joker::Joker),
		PoolSet::Tarot => Kind::Tarot(Tarot::Strength),
		PoolSet::Planet => Kind::Planet(Planet::Pluto),
		PoolSet::Spectral => Kind::Spectral(Spectral::Incantation),
	};
	sample_or(rng, key, &pool.entries[..pool.len], fallback)
}

/// The enhancement chosen by sampling the `Enhanced` pool, which is never
/// filtered by usage (so its first draw always lands).
fn sample_enhancement(run: &Run, rng: &mut PseudoRandom, append: &str) -> Enhancement {
	let key = run.key_append_ante("Enhanced", append);
	Enhancement::POOL[rng.element(&key, Enhancement::POOL.len())]
}

/// What [`create_card`] is asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NewCard<A = Area> {
	pub source: CardSource,
	/// The area the card is created for (shop and pack jokers get stickers).
	pub area: A,
	/// Suffix of the seeds the polls use (`key_append`).
	pub append: &'static str,
}

/// Where [`create_card`] takes its card from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CardSource {
	/// The pool of `set` (`_type`).
	Pool {
		set: PoolSet,
		/// The Soul or Black Hole can take the card's place.
		soulable: bool,
	},
	/// This card (`forced_key`).
	Forced(Kind),
}

impl<A> NewCard<A> {
	/// A card sampled from `set`, with a polled rarity if it is a joker.
	pub(crate) fn new(set: CardSet, area: A, append: &'static str) -> NewCard<A> {
		NewCard {
			source: CardSource::Pool {
				set: PoolSet::seeded(set),
				soulable: false,
			},
			area,
			append,
		}
	}

	/// A card sampled from `set`, unless The Soul or Black Hole takes its
	/// place.
	pub(crate) fn soulable(set: CardSet, area: A, append: &'static str) -> NewCard<A> {
		NewCard {
			source: CardSource::Pool {
				set: PoolSet::seeded(set),
				soulable: true,
			},
			area,
			append,
		}
	}

	/// A joker of the rarity `rarity` decides.
	pub(crate) fn joker(rarity: RarityRoll, area: A, append: &'static str) -> NewCard<A> {
		NewCard {
			source: CardSource::Pool {
				set: PoolSet::Joker(rarity),
				soulable: false,
			},
			area,
			append,
		}
	}

	/// The card `kind`.
	pub(crate) fn forced(kind: Kind, area: A, append: &'static str) -> NewCard<A> {
		NewCard {
			source: CardSource::Forced(kind),
			area,
			append,
		}
	}
}

/// An area of the cards the player owns, which [`create_owned`] creates
/// cards into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OwnedArea {
	Jokers,
	Consumables,
}

impl From<OwnedArea> for Area {
	fn from(area: OwnedArea) -> Area {
		match area {
			OwnedArea::Jokers => Area::Jokers,
			OwnedArea::Consumables => Area::Consumables,
		}
	}
}

/// The roll that decides a created joker's rarity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RarityRoll {
	/// Drawn from the seeded stream.
	Seeded,
	/// This value, with nothing drawn (`_rarity`).
	Fixed(f64),
	/// `legendary`: a legendary joker. Vanilla draws the rarity from the
	/// seeded stream all the same and ignores it (`rolled`); Steamodded
	/// draws nothing.
	Legendary { rolled: bool },
}

/// The Soul or Black Hole taking the place of a card of `set`: a 0.3% roll
/// each, Black Hole's last.
fn roll_soul(filter: PoolFilter, run: &Run, rng: &mut PseudoRandom, set: CardSet) -> Option<Kind> {
	let key = run.key_append_ante("soul_", set.name());
	let mut forced = None;
	if matches!(set, CardSet::Tarot | CardSet::Spectral)
		&& !filter.is_used(Center::Spectral(Spectral::Soul))
		&& rng.random(&key) > 0.997
	{
		forced = Some(Kind::Spectral(Spectral::Soul));
	}
	if matches!(set, CardSet::Planet | CardSet::Spectral)
		&& !filter.is_used(Center::Spectral(Spectral::BlackHole))
		&& rng.random(&key) > 0.997
	{
		forced = Some(Kind::Spectral(Spectral::BlackHole));
	}
	forced
}

/// The stake's stickers for a joker created for the shop or a pack: eternal
/// or perishable from one roll, rental from another.
fn roll_stickers(
	modifiers: Modifiers, jokers: &[Card], run: &Run, discount_percent: f64,
	rng: &mut PseudoRandom, card: &mut Card, area: Area,
) {
	let from_pack = area == Area::PackCards;
	let k = run.key_ante(if from_pack { "packetper" } else { "etperpoll" });
	let poll = rng.random(&k);
	if modifiers.enable_eternals_in_shop && poll > 0.7 {
		card.make_eternal();
	} else if modifiers.enable_perishables_in_shop && poll > 0.4 && poll <= 0.7 {
		card.make_perishable(run.perishable_rounds);
	}
	if modifiers.enable_rentals_in_shop {
		let k = run.key_ante(if from_pack { "packssjr" } else { "ssjr" });
		if rng.random(&k) > 0.7 {
			card.ab.rental = true;
			Pricing::new(discount_percent, jokers).set_cost(card, None);
		}
	}
}

/// `create_card(_type, area, legendary, _rarity, skip_materialize, soulable, forced_key, key_append)`
/// for a card that is not a playing card.
pub(crate) fn create_card(g: &mut Game, spec: NewCard) -> Card {
	let NewCard {
		source,
		area,
		append,
	} = spec;
	let kind = match source {
		CardSource::Forced(kind) => kind,
		CardSource::Pool { set, soulable } => {
			let filter = PoolFilter::new(&g.areas, &g.history);
			let soul = if soulable {
				roll_soul(filter, &g.run, &mut g.rng, set.card_set())
			} else {
				None
			};
			soul.unwrap_or_else(|| {
				sample_pool(
					&g.areas, g.run.ante, filter, &g.hands, &mut g.rng, set, append,
				)
			})
		}
	};
	let mut card = card_lifecycle::make_card(g, kind);
	if card.kind.is_consumable() && area != Area::PackCards {
		g.events.materialize(1.0);
	}
	// A joker set always gives a joker, and a forced or soul card takes the
	// set of its kind (`_type = center.set`).
	if let Kind::Joker(_) = card.kind {
		if matches!(area, Area::ShopJokers | Area::PackCards) {
			roll_stickers(
				g.setup.modifiers,
				&g.areas.jokers,
				&g.run,
				g.shop.discount_percent,
				&mut g.rng,
				&mut card,
				area,
			);
		}
		let k = g.run.key_append_ante("edi", append);
		let e = poll_edition(&mut g.rng, g.shop.edition_rate, &k, 1.0, EditionPool::Any);
		if e.is_none() {
			Pricing::new(g.shop.discount_percent, &g.areas.jokers).set_cost(&mut card, None);
		} else {
			card_lifecycle::set_edition(
				&mut card,
				e,
				None,
				Pricing::new(g.shop.discount_percent, &g.areas.jokers),
				&mut g.areas.limits,
			);
			g.events.edition_fx(e, EditionTiming::Delayed);
		}
	}
	card
}

/// The sets of playing cards `create_card` can sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlayingSet {
	/// A plain front.
	Base,
	/// A front with an enhancement from the `Enhanced` pool.
	Enhanced,
}

/// `create_card('Base' or 'Enhanced', ..., key_append)`: a playing card
/// with a random front.
pub(crate) fn create_playing_card(
	areas: &mut Areas, run: &Run, discount_percent: f64, rng: &mut PseudoRandom, set: PlayingSet,
	append: &str,
) -> PlayingCard {
	let enh = match set {
		PlayingSet::Base => Enhancement::Base,
		PlayingSet::Enhanced => sample_enhancement(run, rng, append),
	};
	let k = run.key_append_ante("front", append);
	let (suit, rank) = random_front(rng, &k);
	card_lifecycle::make_playing_card(areas, run.hands_played, discount_percent, suit, rank, enh)
}

/// `pseudorandom_element(G.P_CARDS, pseudoseed(key))`: a random playing-card
/// front.
pub(crate) fn random_front(rng: &mut PseudoRandom, key: &Key) -> (Suit, Rank) {
	P_CARDS_SORTED[rng.element(key, P_CARDS_SORTED.len())]
}

/// [`create_card`], `add_to_deck` and `emplace` into the area the card is
/// created for.
pub(crate) fn create_owned(g: &mut Game, spec: NewCard<OwnedArea>) {
	let NewCard {
		source,
		area,
		append,
	} = spec;
	let mut c = create_card(g, NewCard {
		source,
		area: area.into(),
		append,
	});
	card_lifecycle::add_to_deck(g, &mut c);
	match area {
		OwnedArea::Jokers => g.areas.jokers.push(c),
		OwnedArea::Consumables => g.areas.consumables.push(c),
	}
}

/// `get_new_boss()`.
pub(crate) fn get_new_boss(
	win_ante: i32, ante: i32, blind_select: &mut BlindSelect, rng: &mut PseudoRandom,
) -> BlindKind {
	let clamped_ante = ante.max(1);
	let eligible = |b: BlindKind| match b.boss() {
		None => false,
		Some(Boss::Regular) => {
			b.def().min_ante <= clamped_ante && (clamped_ante % win_ante != 0 || ante < 2)
		}
		Some(Boss::Showdown) => ante % win_ante == 0 && ante >= 2,
	};
	let used = &mut blind_select.bosses_used;
	let mut min_use = 100;
	for b in ALL_BLINDS {
		if eligible(b) {
			min_use = min_use.min(used[b]);
		}
	}
	let mut list = [BlindKind::Small; ALL_BLINDS.len()];
	let mut n = 0;
	for b in ALL_BLINDS {
		if eligible(b) && used[b] <= min_use {
			list[n] = b;
			n += 1;
		}
	}
	let k = Key::from("boss");
	let boss = list[rng.element(&k, n)];
	used[boss] += 1;
	boss
}

/// What a voucher is rolled for (`from_tag`): each has its own stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VoucherRoll {
	Ante,
	Tag,
}

/// `get_next_voucher_key(from_tag)`.
pub(crate) fn get_next_voucher(
	areas: &Areas, run: &Run, history: &History, rng: &mut PseudoRandom, roll: VoucherRoll,
) -> Voucher {
	let filter = PoolFilter::new(areas, history);
	let pool = ALL_VOUCHERS.map(|v| {
		let add = !filter.is_used(Center::Voucher(v))
			&& !history.used_vouchers.contains(v)
			&& v.def()
				.requires
				.is_none_or(|r| history.used_vouchers.contains(r))
			&& !areas
				.shop_vouchers
				.iter()
				.any(|s| s.kind == Kind::Voucher(v));
		add.then_some(v)
	});
	let key = match roll {
		VoucherRoll::Tag => Key::from("Voucher_fromtag"),
		VoucherRoll::Ante => run.key_ante("Voucher"),
	};
	sample_or(rng, key, &pool, Voucher::Blank)
}

/// `get_next_tag_key()`.
pub(crate) fn get_next_tag(run: &Run, rng: &mut PseudoRandom) -> TagKind {
	// With a fully discovered profile every `requires` is satisfied.
	let pool = ALL_TAGS.map(|t| t.min_ante().is_none_or(|m| m <= run.ante).then_some(t));
	sample_or(rng, run.key_ante("Tag"), &pool, TagKind::Handy)
}

/// `get_pack('shop_pack')`.
///
/// # Panics
///
/// Never with the generated weights: each booster's range starts exactly
/// where the one before it ends, and the last ends at the total weight,
/// which is at least the poll (where none matches, the game's `get_pack`
/// returns `nil`).
pub(crate) fn get_pack(run: &Run, shop: &mut Shop, rng: &mut PseudoRandom) -> Booster {
	if !shop.first_buffoon {
		shop.first_buffoon = true;
		// `math.random(1, 2)` on the unseeded global generator; both
		// variants are identical apart from art.
		return Booster::BuffoonNormal1;
	}
	let mut cume = 0.0;
	for &b in &ALL_BOOSTERS {
		cume += b.def().weight;
	}
	let k = run.key_ante("shop_pack");
	let poll = rng.random(&k) * cume;
	let mut it = 0.0;
	for &b in &ALL_BOOSTERS {
		it += b.def().weight;
		if it >= poll && it - b.def().weight <= poll {
			return b;
		}
	}
	unreachable!(
		"the boosters' ranges cover 0 to the total weight without gaps, and the poll lies in it"
	)
}
