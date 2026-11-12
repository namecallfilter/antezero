//! Joker contexts of hand scoring: before, individual cards, repetitions,
//! the joker pass (`main` / other jokers), after, and card destruction.

use crate::card::Kind;
use crate::data::{ACE, Enhancement, HandType, Joker, KING, QUEEN, Rank, Suit, rank_nominal};
use crate::ev::Ev;
use crate::game::{Area, CardSet, Game};
use crate::jokers::{CardArea, Ctx, Effect, Scoring, ret};
use crate::rng::Key;
use crate::{cards, flow, jokers, num};

#[expect(
	clippy::too_many_lines,
	reason = "one arm per context; the order of the checks is the Lua if-chain's"
)]
pub(crate) fn calculate(
	g: &mut Game, ji: usize, j: Joker, ctx: &Ctx, sc: Option<&Scoring>, blueprinted: bool,
) -> Option<Effect> {
	use Joker as J;
	let ab = g.jokers[ji].ab;
	let prob = g.probabilities;
	match *ctx {
		Ctx::DestroyingCard { play_idx, full_len } => {
			if blueprinted {
				return None;
			}
			if j == J::SixthSense
				&& full_len == 1
				&& g.play[play_idx].id_is(6)
				&& g.hands_played_round == 0.0
			{
				if jokers::consumable_room(g) {
					g.consumeable_buffer += 1.0;
					g.events.push_before(0.0, Ev::Create {
						set: CardSet::Spectral,
						area: Area::Consumables,
						forced: None,
						append: "sixth",
						rarity: None,
						legendary: false,
					});
					g.events.status_extra();
				}
				ret!(remove: true);
			}
			None
		}
		Ctx::DebuffedHand => {
			if j == J::Matador && g.blind.triggered {
				g.events.push(Ev::EaseDollars(8.0));
				jokers::buffer_dollars(g, 8.0);
				ret!(dollars: Some(8.0));
			}
			None
		}
		Ctx::Individual {
			area: CardArea::Play,
			other,
		} => {
			let sc = sc.expect("individual context without scoring");
			let par = g.pareidolia();
			let smeared = g.smeared();
			let oc = &g.play[other];
			if j == J::Hiker {
				g.play[other].ab.perma_bonus += 5.0;
				ret!(extra: true);
			}
			if j == J::LuckyCat && oc.lucky_trigger && !blueprinted {
				jokers::jab(g, ji).x_mult += 0.25;
				ret!(extra: true);
			}
			if j == J::Wee && oc.id_is(2) && !blueprinted {
				jokers::jab(g, ji).chips += 8.0;
				ret!(extra: true);
			}
			if j == J::Photograph {
				let first_face = sc
					.scoring()
					.iter()
					.map(|&i| i as usize)
					.find(|&i| g.play[i].is_face(false, par));
				if Some(other) == first_face {
					ret!(x_mult: Some(2.0));
				}
			}
			if j == J::J8Ball && jokers::consumable_room(g) && g.play[other].id_is(8) {
				let k = Key::from("8ball");
				if g.rng.random(&k) < prob / 4.0 {
					g.consumeable_buffer += 1.0;
					ret!(extra: true, eight_ball: true);
				}
			}
			let oc = &g.play[other];
			if j == J::Idol
				&& oc.get_id() == Some(g.idol_card.rank)
				&& oc.is_suit(g.idol_card.suit, false, false, smeared)
			{
				ret!(x_mult: Some(2.0));
			}
			if j == J::ScaryFace && oc.is_face(false, par) {
				ret!(chips: Some(30.0));
			}
			if j == J::Smiley && oc.is_face(false, par) {
				ret!(mult: Some(5.0));
			}
			if j == J::Ticket && oc.enhancement == Enhancement::Gold {
				jokers::buffer_dollars(g, 4.0);
				ret!(dollars: Some(4.0));
			}
			if j == J::Scholar && oc.id_is(ACE) {
				ret!(chips: Some(20.0), mult: Some(4.0));
			}
			if j == J::WalkieTalkie && (oc.id_is(10) || oc.id_is(4)) {
				ret!(chips: Some(10.0), mult: Some(4.0));
			}
			if j == J::Business && oc.is_face(false, par) {
				let k = Key::from("business");
				if g.rng.random(&k) < prob / 2.0 {
					jokers::buffer_dollars(g, 2.0);
					ret!(dollars: Some(2.0));
				}
			}
			let oc = &g.play[other];
			if j == J::Fibonacci && matches!(oc.get_id(), Some(2 | 3 | 5 | 8 | ACE)) {
				ret!(mult: Some(8.0));
			}
			if j == J::EvenSteven && matches!(oc.get_id(), Some(r) if r <= 10 && r % 2 == 0) {
				ret!(mult: Some(4.0));
			}
			if j == J::OddTodd
				&& matches!(oc.get_id(), Some(r) if (r <= 10 && r % 2 == 1) || r == ACE)
			{
				ret!(chips: Some(31.0));
			}
			let suit_mult = match j {
				J::GreedyJoker => Some(Suit::Diamonds),
				J::LustyJoker => Some(Suit::Hearts),
				J::WrathfulJoker => Some(Suit::Spades),
				J::GluttenousJoker => Some(Suit::Clubs),
				_ => None,
			};
			if let Some(s) = suit_mult
				&& oc.is_suit(s, false, false, smeared)
			{
				ret!(mult: Some(3.0));
			}
			if j == J::RoughGem && oc.is_suit(Suit::Diamonds, false, false, smeared) {
				jokers::buffer_dollars(g, 1.0);
				ret!(dollars: Some(1.0));
			}
			let oc = &g.play[other];
			if j == J::OnyxAgate && oc.is_suit(Suit::Clubs, false, false, smeared) {
				ret!(mult: Some(7.0));
			}
			if j == J::Arrowhead && oc.is_suit(Suit::Spades, false, false, smeared) {
				ret!(chips: Some(50.0));
			}
			if j == J::Bloodstone && oc.is_suit(Suit::Hearts, false, false, smeared) {
				let k = Key::from("bloodstone");
				if g.rng.random(&k) < prob / 2.0 {
					ret!(x_mult: Some(1.5));
				}
			}
			let oc = &g.play[other];
			if j == J::Ancient
				&& g.ancient_suit
					.is_some_and(|s| oc.is_suit(s, false, false, smeared))
			{
				ret!(x_mult: Some(1.5));
			}
			if j == J::Triboulet && (oc.id_is(QUEEN) || oc.id_is(KING)) {
				ret!(x_mult: Some(2.0));
			}
			None
		}
		Ctx::Individual {
			area: CardArea::Hand,
			other,
		} => {
			let par = g.pareidolia();
			let oc = &g.hand[other];
			if j == J::ShootTheMoon && oc.id_is(QUEEN) {
				if oc.debuff {
					ret!(extra: true);
				}
				ret!(h_mult: Some(13.0));
			}
			if j == J::Baron && oc.id_is(KING) {
				if oc.debuff {
					ret!(extra: true);
				}
				ret!(x_mult: Some(1.5));
			}
			if j == J::ReservedParking && oc.is_face(false, par) {
				let k = Key::from("parking");
				if g.rng.random(&k) < prob / 2.0 {
					if g.hand[other].debuff {
						ret!(extra: true);
					}
					jokers::buffer_dollars(g, 1.0);
					ret!(dollars: Some(1.0));
				}
			}
			if j == J::RaisedFist {
				let mut temp_mult = 15.0;
				let mut temp_id: Rank = 15;
				let mut raised = None;
				for (i, c) in g.hand.iter().enumerate() {
					if temp_id >= c.rank && !c.is_stone() {
						temp_mult = rank_nominal(c.rank);
						temp_id = c.rank;
						raised = Some(i);
					}
				}
				if raised == Some(other) {
					if g.hand[other].debuff {
						ret!(extra: true);
					}
					ret!(h_mult: Some(2.0 * temp_mult));
				}
			}
			None
		}
		Ctx::Repetition {
			area: CardArea::Play,
			other,
			..
		} => {
			let sc = sc.expect("repetition without scoring");
			let par = g.pareidolia();
			let oc = &g.play[other];
			if j == J::SockAndBuskin && oc.is_face(false, par) {
				ret!(repetitions: 1);
			}
			if j == J::HangingChad && sc.scoring().first().map(|&i| i as usize) == Some(other) {
				ret!(repetitions: 2);
			}
			if j == J::Dusk && g.hands_left == 0.0 {
				ret!(repetitions: 1);
			}
			if j == J::Selzer {
				ret!(repetitions: 1);
			}
			if j == J::Hack && matches!(oc.get_id(), Some(2..=5)) {
				ret!(repetitions: 1);
			}
			None
		}
		Ctx::Repetition {
			area: CardArea::Hand,
			effects,
			..
		} => {
			if j == J::Mime && effects {
				ret!(repetitions: 1);
			}
			None
		}
		Ctx::OtherJoker { other } => {
			if let (J::Baseball, Some(other)) = (j, other)
				&& other != ji
				&& let Some(oj) = g.jokers[other].joker()
				&& oj.rarity() == 2
			{
				g.events.push(Ev::Wait);
				ret!(xmult_mod: Some(1.5));
			}
			None
		}
		Ctx::Before => {
			let sc = sc.expect("before without scoring");
			joker_before(g, ji, j, sc, blueprinted)
		}
		Ctx::After => {
			if j == J::IceCream && !blueprinted {
				if ab.chips - 5.0 <= 0.0 {
					jokers::schedule_self_destruct(g, ji);
					ret!(extra: true);
				}
				jokers::jab(g, ji).chips -= 5.0;
				ret!(extra: true);
			}
			if j == J::Selzer && !blueprinted {
				if ab.remaining - 1.0 <= 0.0 {
					jokers::schedule_self_destruct(g, ji);
					ret!(extra: true);
				}
				jokers::jab(g, ji).remaining -= 1.0;
				ret!(extra: true);
			}
			None
		}
		Ctx::Main => {
			let sc = sc.expect("main without scoring");
			joker_main(g, ji, j, sc)
		}
		_ => unreachable!("lifecycle context {ctx:?}"),
	}
}

#[expect(
	clippy::too_many_lines,
	reason = "one check per joker, in the Lua if-chain's order"
)]
fn joker_before(
	g: &mut Game, ji: usize, j: Joker, sc: &Scoring, blueprinted: bool,
) -> Option<Effect> {
	use Joker as J;
	let prob = g.probabilities;
	let ab = g.jokers[ji].ab;
	let hands = sc.hands;
	if j == J::Trousers
		&& (hands.has(HandType::TwoPair) || hands.has(HandType::FullHouse))
		&& !blueprinted
	{
		jokers::jab(g, ji).mult += 2.0;
		ret!(extra: true);
	}
	if j == J::Space {
		let k = Key::from("space");
		if g.rng.random(&k) < prob / 4.0 {
			ret!(level_up: true);
		}
	}
	if j == J::Square && g.play.len() == 4 && !blueprinted {
		jokers::jab(g, ji).chips += 4.0;
		ret!(extra: true);
	}
	if j == J::Runner && hands.has(HandType::Straight) && !blueprinted {
		jokers::jab(g, ji).chips += 15.0;
		ret!(extra: true);
	}
	if j == J::MidasMask && !blueprinted {
		let par = g.pareidolia();
		let mut any = false;
		for &i in sc.scoring() {
			let i = i as usize;
			if g.play[i].is_face(false, par) {
				any = true;
				cards::set_enhancement(g, Area::Play, i, Enhancement::Gold);
				g.events.push(Ev::Wait);
			}
		}
		if any {
			ret!(extra: true);
		}
	}
	if j == J::Vampire && !blueprinted {
		let mut drained = 0;
		for &i in sc.scoring() {
			let i = i as usize;
			let c = &g.play[i];
			if c.enhancement != Enhancement::Base && !c.debuff && !c.vampired {
				drained += 1;
				g.play[i].vampired = true;
				cards::set_enhancement(g, Area::Play, i, Enhancement::Base);
				let id = g.play[i].id;
				g.events.push(Ev::Unvampire { id });
			}
		}
		if drained > 0 {
			jokers::jab(g, ji).x_mult += 0.1 * f64::from(drained);
			ret!(extra: true);
		}
	}
	if j == J::TodoList && Some(sc.name) == ab.to_do {
		g.events.push(Ev::EaseDollars(4.0));
		jokers::buffer_dollars(g, 4.0);
		ret!(dollars: Some(4.0));
	}
	if j == J::Dna && g.hands_played_round == 0.0 && g.play.len() == 1 {
		let mut c = flow::copy_card(g, g.play[0].clone(), false);
		cards::add_to_deck(g, &mut c, false);
		g.hand.push(c);
		g.events.push(Ev::Materialize);
		ret!(playing_cards_created: true, extra: true);
	}
	if j == J::RideTheBus && !blueprinted {
		let par = g.pareidolia();
		let faces = sc
			.scoring()
			.iter()
			.any(|&i| g.play[i as usize].is_face(false, par));
		if faces {
			let last = ab.mult;
			jokers::jab(g, ji).mult = 0.0;
			if last > 0.0 {
				ret!(extra: true);
			}
		} else {
			jokers::jab(g, ji).mult += 1.0;
		}
	}
	if j == J::Obelisk && !blueprinted {
		let mut reset = true;
		let more_than = g.hands[sc.name as usize].played;
		for h in HandType::PAIRS_ORDER {
			let hl = &g.hands[h as usize];
			if h != sc.name && hl.played >= more_than && hl.visible {
				reset = false;
			}
		}
		if reset {
			if ab.x_mult > 1.0 {
				jokers::jab(g, ji).x_mult = 1.0;
				ret!(extra: true);
			}
		} else {
			jokers::jab(g, ji).x_mult += 0.2;
		}
	}
	if j == J::GreenJoker && !blueprinted {
		jokers::jab(g, ji).mult += 1.0;
		ret!(extra: true);
	}
	None
}

#[expect(
	clippy::too_many_lines,
	reason = "one check per joker, in the Lua if-chain's order"
)]
fn joker_main(g: &mut Game, ji: usize, j: Joker, sc: &Scoring) -> Option<Effect> {
	use Joker as J;
	let ab = g.jokers[ji].ab;
	let hands = sc.hands;
	let has_type = |h: Option<HandType>| -> bool {
		match h {
			None => true,
			Some(h) => hands.has(h),
		}
	};
	let jtype = joker_hand_type(j);
	if j == J::LoyaltyCard {
		let every = 5.0;
		let rem = crate::rng::lua_mod(
			every - 1.0 - (g.hands_played - ab.hands_played_at_create),
			every + 1.0,
		);
		jokers::jab(g, ji).loyalty_remaining = rem;
		#[expect(clippy::float_cmp, reason = "exact: both are whole numbers of hands")]
		let due = rem == every;
		if due {
			ret!(xmult_mod: Some(4.0));
		}
	}
	let ab = g.jokers[ji].ab;
	if j != J::SeeingDouble && ab.x_mult > 1.0 && has_type(jtype) {
		ret!(xmult_mod: Some(ab.x_mult));
	}
	let (t_mult, t_chips) = joker_t_values(j);
	if t_mult > 0.0 && has_type(jtype) {
		ret!(mult_mod: Some(t_mult));
	}
	if t_chips > 0.0 && has_type(jtype) {
		ret!(chip_mod: Some(t_chips));
	}
	if j == J::Half && g.play.len() <= 3 {
		ret!(mult_mod: Some(20.0));
	}
	if j == J::Abstract {
		let x = num::float(
			g.jokers
				.iter()
				.filter(|c| matches!(c.kind, Kind::Joker(_)))
				.count(),
		);
		ret!(mult_mod: Some(x * 3.0));
	}
	if j == J::Acrobat && g.hands_left == 0.0 {
		ret!(xmult_mod: Some(3.0));
	}
	if j == J::MysticSummit && g.discards_left == 0.0 {
		ret!(mult_mod: Some(15.0));
	}
	if j == J::Misprint {
		let k = Key::from("misprint");
		let m = g.rng.random_range(&k, 0.0, 23.0);
		ret!(mult_mod: Some(m));
	}
	if j == J::Banner && g.discards_left > 0.0 {
		ret!(chip_mod: Some(g.discards_left * 30.0));
	}
	if j == J::Stuntman {
		ret!(chip_mod: Some(250.0));
	}
	if j == J::Matador && g.blind.triggered {
		g.events.push(Ev::EaseDollars(8.0));
		jokers::buffer_dollars(g, 8.0);
		ret!(dollars: Some(8.0));
	}
	if j == J::Supernova {
		ret!(mult_mod: Some(g.hands[sc.name as usize].played));
	}
	if j == J::Ceremonial && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::Vagabond && jokers::consumable_room(g) && g.dollars <= 4.0 {
		g.consumeable_buffer += 1.0;
		g.events.push_before(0.0, Ev::Create {
			set: CardSet::Tarot,
			area: Area::Consumables,
			forced: None,
			append: "vag",
			rarity: None,
			legendary: false,
		});
		ret!(extra: true);
	}
	if j == J::Superposition && jokers::consumable_room(g) {
		let aces = sc
			.scoring()
			.iter()
			.filter(|&&i| g.play[i as usize].id_is(ACE))
			.count();
		if aces >= 1 && hands.has(HandType::Straight) {
			g.consumeable_buffer += 1.0;
			g.events.push_before(0.0, Ev::Create {
				set: CardSet::Tarot,
				area: Area::Consumables,
				forced: None,
				append: "sup",
				rarity: None,
				legendary: false,
			});
			ret!(extra: true);
		}
	}
	if j == J::Seance && jokers::consumable_room(g) && hands.has(HandType::StraightFlush) {
		g.consumeable_buffer += 1.0;
		g.events.push_before(0.0, Ev::Create {
			set: CardSet::Spectral,
			area: Area::Consumables,
			forced: None,
			append: "sea",
			rarity: None,
			legendary: false,
		});
		ret!(extra: true);
	}
	if j == J::FlowerPot {
		let smeared = g.smeared();
		let mut suits = [0u8; 4]; // H, D, S, C
		let order = [Suit::Hearts, Suit::Diamonds, Suit::Spades, Suit::Clubs];
		for &i in sc.scoring() {
			let c = &g.play[i as usize];
			if c.enhancement != Enhancement::Wild {
				for (si, &s) in order.iter().enumerate() {
					if c.is_suit(s, true, false, smeared) && suits[si] == 0 {
						suits[si] += 1;
						break;
					}
				}
			}
		}
		for &i in sc.scoring() {
			let c = &g.play[i as usize];
			if c.enhancement == Enhancement::Wild {
				for (si, &s) in order.iter().enumerate() {
					if c.is_suit(s, false, false, smeared) && suits[si] == 0 {
						suits[si] += 1;
						break;
					}
				}
			}
		}
		if suits.iter().all(|&x| x > 0) {
			ret!(xmult_mod: Some(3.0));
		}
	}
	if j == J::SeeingDouble {
		let smeared = g.smeared();
		// Hearts, Diamonds, Spades, Clubs
		let mut suits = [0u32; 4];
		for &i in sc.scoring() {
			let c = &g.play[i as usize];
			if c.enhancement != Enhancement::Wild {
				if c.is_suit(Suit::Hearts, false, false, smeared) {
					suits[0] += 1;
				}
				if c.is_suit(Suit::Diamonds, false, false, smeared) {
					suits[1] += 1;
				}
				if c.is_suit(Suit::Spades, false, false, smeared) {
					suits[2] += 1;
				}
				if c.is_suit(Suit::Clubs, false, false, smeared) {
					suits[3] += 1;
				}
			}
		}
		for &i in sc.scoring() {
			let c = &g.play[i as usize];
			if c.enhancement == Enhancement::Wild {
				if c.is_suit(Suit::Clubs, false, false, smeared) && suits[3] == 0 {
					suits[3] += 1;
				} else if c.is_suit(Suit::Diamonds, false, false, smeared) && suits[1] == 0 {
					suits[1] += 1;
				} else if c.is_suit(Suit::Spades, false, false, smeared) && suits[2] == 0 {
					suits[2] += 1;
				} else if c.is_suit(Suit::Hearts, false, false, smeared) && suits[0] == 0 {
					suits[0] += 1;
				}
			}
		}
		if (suits[0] > 0 || suits[1] > 0 || suits[2] > 0) && suits[3] > 0 {
			ret!(xmult_mod: Some(2.0));
		}
	}
	if j == J::Wee {
		ret!(chip_mod: Some(ab.chips));
	}
	if j == J::Castle && ab.chips > 0.0 {
		ret!(chip_mod: Some(ab.chips));
	}
	if j == J::BlueJoker && !g.deck.is_empty() {
		ret!(chip_mod: Some(2.0 * num::float(g.deck.len())));
	}
	if j == J::Erosion && (g.starting_deck_size - num::float(g.playing_card_count())) > 0.0 {
		ret!(mult_mod: Some(4.0 * (g.starting_deck_size - num::float(g.playing_card_count()))));
	}
	if j == J::Square {
		ret!(chip_mod: Some(ab.chips));
	}
	if j == J::Runner {
		ret!(chip_mod: Some(ab.chips));
	}
	if j == J::IceCream {
		ret!(chip_mod: Some(ab.chips));
	}
	if j == J::Stone && ab.stone_tally > 0.0 {
		ret!(chip_mod: Some(25.0 * ab.stone_tally));
	}
	if j == J::SteelJoker && ab.steel_tally > 0.0 {
		ret!(xmult_mod: Some(1.0 + 0.2 * ab.steel_tally));
	}
	if j == J::Bull && (g.dollars + g.dollar_buffer) > 0.0 {
		ret!(chip_mod: Some(2.0 * (g.dollars + g.dollar_buffer).max(0.0)));
	}
	if j == J::DriversLicense && ab.driver_tally >= 16.0 {
		ret!(xmult_mod: Some(3.0));
	}
	if j == J::Blackboard {
		let smeared = g.smeared();
		if g.hand.iter().all(|c| {
			c.is_suit(Suit::Clubs, false, true, smeared)
				|| c.is_suit(Suit::Spades, false, true, smeared)
		}) {
			ret!(xmult_mod: Some(3.0));
		}
	}
	if j == J::Stencil && (g.joker_limit.limit - num::float(g.jokers.len())) > 0.0 {
		ret!(xmult_mod: Some(ab.x_mult));
	}
	if j == J::Swashbuckler && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::Joker {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::Trousers && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::RideTheBus && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::Flash && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::Popcorn && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::GreenJoker && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::FortuneTeller && g.consumable_usage_tarot > 0.0 {
		ret!(mult_mod: Some(g.consumable_usage_tarot));
	}
	if j == J::GrosMichel {
		ret!(mult_mod: Some(15.0));
	}
	if j == J::Cavendish {
		ret!(xmult_mod: Some(3.0));
	}
	if j == J::RedCard && ab.mult > 0.0 {
		ret!(mult_mod: Some(ab.mult));
	}
	if j == J::CardSharp && g.hands[sc.name as usize].played_this_round > 1.0 {
		ret!(xmult_mod: Some(3.0));
	}
	if j == J::Bootstraps && ((g.dollars + g.dollar_buffer) / 5.0).floor() >= 1.0 {
		ret!(mult_mod: Some(2.0 * ((g.dollars + g.dollar_buffer) / 5.0).floor()));
	}
	if j == J::Caino && ab.caino_xmult > 1.0 {
		ret!(xmult_mod: Some(ab.caino_xmult));
	}
	None
}

/// `ability.type` (poker hand requirement) for a joker.
fn joker_hand_type(j: Joker) -> Option<HandType> {
	use Joker as J;
	Some(match j {
		J::Jolly | J::Sly | J::Duo => HandType::Pair,
		J::Zany | J::Wily | J::Trio => HandType::ThreeOfAKind,
		J::Mad | J::Clever => HandType::TwoPair,
		J::Crazy | J::Devious | J::Order => HandType::Straight,
		J::Droll | J::Crafty | J::Tribe => HandType::Flush,
		J::Family => HandType::FourOfAKind,
		_ => return None,
	})
}

/// `(t_mult, t_chips)`.
fn joker_t_values(j: Joker) -> (f64, f64) {
	use Joker as J;
	match j {
		J::Jolly => (8.0, 0.0),
		J::Zany | J::Crazy => (12.0, 0.0),
		J::Mad | J::Droll => (10.0, 0.0),
		J::Sly => (0.0, 50.0),
		J::Wily | J::Devious => (0.0, 100.0),
		J::Clever | J::Crafty => (0.0, 80.0),
		_ => (0.0, 0.0),
	}
}
