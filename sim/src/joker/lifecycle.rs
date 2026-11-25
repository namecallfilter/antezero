//! Joker contexts outside hand scoring: shop, packs, selling, blind setup,
//! discards and the end of the round.
//!
//! A joker reacts to a context in one `match` arm, as in [`super::scoring`].
//! What a joker does when it is gained or lost (`add_to_deck` /
//! `remove_from_deck`) is in `card`.

use super::calculate::calc_jokers;
use super::effect::{
	Discarded, Effect, JokerEvent, LifecycleCtx, queue_consumable, schedule_self_destruct,
};
use crate::areas::Pile;
use crate::card::{Card, Kind, Pricing, SuitTest, YORICK_DISCARDS};
use crate::data::{Edition, Enhancement, Joker, Rank, Seal, TagKind, Tarot};
use crate::event::{Blocks, Draw, EditionTiming, Event, EventQueue, Trigger};
use crate::game::{Area, Areas, CardSet, Game};
use crate::pools::{NewCard, OwnedArea, RarityRoll};
use crate::rng::Key;
use crate::round::{CardDebuff, Round};
use crate::run::TagEvent;
use crate::{areas, card, event, flow, num, pools, round};

/// `Card:calculate_joker` for the contexts outside hand scoring; returns
/// the effect to show, if any. `copier` is the Blueprint or Brainstorm the
/// joker's logic runs for, if it is being copied.
pub(super) fn calculate(
	g: &mut Game, ji: usize, j: Joker, ctx: LifecycleCtx, copier: Option<usize>,
) -> Option<Effect> {
	use Joker as J;
	let blueprinted = copier.is_some();
	match (ctx, j) {
		(LifecycleCtx::OpenBooster, J::Hallucination) if g.areas.consumable_room_after_queued() => {
			let key = g.run.key_ante("halu");
			if g.rng.random(&key) < g.run.probabilities / 2.0 {
				queue_consumable(&mut g.areas, &mut g.events, CardSet::Tarot, "hal");
				g.events.status_extra();
			}
		}
		(LifecycleCtx::SellingSelf, _) => selling_self(g, ji, j, blueprinted),
		(LifecycleCtx::SellingCard, J::Campfire) if !blueprinted => {
			g.areas.jokers[ji].ab.x_mult += 0.25;
			g.events.push(Event::StatusLater(0.75));
		}
		(LifecycleCtx::RerollShop, J::Flash) if !blueprinted => {
			g.areas.jokers[ji].ab.mult += 2.0;
			g.events.push(Event::StatusLater(0.75));
		}
		(LifecycleCtx::EndingShop, J::Perkeo) if !g.areas.consumables.is_empty() => {
			g.events.push(Event::Joker(JokerEvent::PerkeoCopy));
			g.events.status_extra();
		}
		(LifecycleCtx::SkipBlind, J::Throwback) if !blueprinted => {
			g.events.push(Event::StatusLater(0.75));
		}
		(LifecycleCtx::SkippingBooster, J::RedCard) if !blueprinted => {
			g.areas.jokers[ji].ab.mult += 3.0;
			g.events.push(Event::StatusLater(0.45));
		}
		(LifecycleCtx::PlayingCardAdded { n }, J::Hologram)
			if !g.areas.jokers[ji].getting_sliced && !blueprinted && n > 0 =>
		{
			g.areas.jokers[ji].ab.x_mult += num::float(n) * 0.25;
			g.events.status_extra();
		}
		(LifecycleCtx::FirstHandDrawn, J::Certificate) => {
			g.events.push(Event::Joker(JokerEvent::CertificateCard));
			calc_jokers(g, LifecycleCtx::PlayingCardAdded { n: 1 });
		}
		(LifecycleCtx::SettingBlind { boss }, _) if !g.areas.jokers[ji].getting_sliced => {
			setting_blind(g, ji, j, boss, copier);
		}
		(LifecycleCtx::RemovePlayingCards { faces, .. }, J::Caino) if !blueprinted && faces > 0 => {
			g.areas.jokers[ji].ab.caino_xmult += num::float(faces) * 1.0;
			g.events.push(Event::StatusLater(0.75));
		}
		(LifecycleCtx::RemovePlayingCards { glass, .. }, J::Glass) if !blueprinted && glass > 0 => {
			g.events.push(Event::Joker(JokerEvent::GlassJokerOuter {
				id: g.areas.jokers[ji].core.id,
				n: num::index32(glass),
			}));
		}
		(LifecycleCtx::UsingConsumeable { kind, hanged_glass }, _) if !blueprinted => {
			using_consumable(
				&mut g.areas.jokers,
				&mut g.events,
				ji,
				j,
				kind,
				hanged_glass,
			);
		}
		(
			LifecycleCtx::PreDiscard {
				hook: false,
				burnt_hand: Some(hand),
			},
			J::Burnt,
		) if g.round.discards_used <= 0.0 => {
			g.events.status_extra();
			g.events.hand_text(0.3);
			event::level_up_hand(&mut g.hands, &mut g.events, hand, 1.0);
			g.events.hand_text(0.0);
		}
		(LifecycleCtx::Discard(d), _) => {
			return discard(&mut g.areas, &g.round, &mut g.events, ji, j, blueprinted, d);
		}
		(LifecycleCtx::EndOfRound { game_over }, _) if !blueprinted => {
			return end_of_round(g, ji, j, game_over);
		}
		(LifecycleCtx::EndOfRoundRepetition { effects: true }, J::Mime) => {
			return Some(Effect {
				repetitions: 1,
				..Effect::NONE
			});
		}
		// No other joker reacts in these contexts.
		_ => {}
	}
	None
}

/// `context.selling_self`.
fn selling_self(g: &mut Game, ji: usize, j: Joker, blueprinted: bool) {
	match j {
		Joker::Luchador if !g.blind.disabled && g.blind.is_boss() => {
			g.events.status_extra();
			round::disable_blind(g);
		}
		Joker::DietCola => g.events.push(Event::Tag(TagEvent::Add {
			kind: TagKind::Double,
			orbital: None,
		})),
		Joker::Invisible if g.areas.jokers[ji].ab.invis_rounds >= 2.0 && !blueprinted => {
			g.events.status_extra();
			let others = |i, _: &Card| i != ji;
			if num::float(g.areas.jokers.len()) <= g.areas.limits.jokers.limit
				&& let Some(pick) =
					areas::pick_card(&mut g.rng, "invisible", &g.areas.jokers, others)
			{
				let strip = g.areas.jokers[pick].core.edition == Some(Edition::Negative);
				let original = g.areas.jokers[pick].clone();
				let mut c = flow::copy_card(
					&mut g.areas,
					&mut g.history,
					g.shop.discount_percent,
					&g.hands,
					&mut g.rng,
					original,
					strip,
				);
				if c.is_joker(Joker::Invisible) {
					c.ab.invis_rounds = 0.0;
				}
				card::add_to_deck(g, &mut c);
				g.areas.jokers.push(c);
			}
		}
		_ => {}
	}
}

/// `context.setting_blind` for a joker that is not being sliced. Most
/// effects also need the acting joker (`copier`, else the joker itself) to
/// be intact.
fn setting_blind(g: &mut Game, ji: usize, j: Joker, boss: bool, copier: Option<usize>) {
	use Joker as J;
	let blueprinted = copier.is_some();
	let actor_intact = !g.areas.jokers[copier.unwrap_or(ji)].getting_sliced;
	match j {
		J::Chicot if !blueprinted && boss => g.events.push(Event::Joker(JokerEvent::ChicotOuter)),
		J::Madness if !blueprinted && !boss => {
			g.areas.jokers[ji].ab.x_mult += 0.5;
			let victims = |i, c: &Card| i != ji && !c.ab.eternal() && !c.getting_sliced;
			// The victim is drawn even when the Madness is being sliced itself.
			if let Some(pick) = areas::pick_card(&mut g.rng, "madness", &g.areas.jokers, victims)
				&& actor_intact
			{
				g.areas.jokers[pick].getting_sliced = true;
				let id = g.areas.jokers[pick].core.id;
				g.events.push(Event::Joker(JokerEvent::DissolveJoker {
					id,
					time_factor: 1.6,
				}));
			}
			if actor_intact {
				g.events.status_extra();
			}
		}
		J::Burglar if actor_intact => g.events.push(Event::Joker(JokerEvent::Burglar)),
		J::RiffRaff
			if actor_intact
				&& (num::float(g.areas.jokers.len()) + g.areas.joker_buffer)
					< g.areas.limits.jokers.limit =>
		{
			let n = 2f64.min(
				g.areas.limits.jokers.limit
					- (num::float(g.areas.jokers.len()) + g.areas.joker_buffer),
			);
			g.areas.joker_buffer += n;
			g.events.push(Event::Joker(JokerEvent::RiffRaff {
				n: num::index32(num::trunc(n)),
			}));
			g.events.status_extra();
		}
		J::Cartomancer if actor_intact && g.areas.consumable_room_after_queued() => {
			g.areas.consumable_buffer += 1.0;
			g.events.push(Event::Joker(JokerEvent::CartomancerOuter));
		}
		J::Ceremonial
			if !blueprinted
				&& ji + 1 < g.areas.jokers.len()
				&& !g.areas.jokers[ji + 1].ab.eternal()
				&& !g.areas.jokers[ji + 1].getting_sliced =>
		{
			g.areas.jokers[ji + 1].getting_sliced = true;
			g.areas.joker_buffer -= 1.0;
			let me = g.areas.jokers[ji].core.id;
			let victim = g.areas.jokers[ji + 1].core.id;
			g.events
				.push(Event::Joker(JokerEvent::DaggerSlice { dagger: me, victim }));
			g.events.status_extra();
		}
		J::Marble if actor_intact => {
			g.events.push(Event::Joker(JokerEvent::MarbleStone));
			g.events.status_extra();
			// G.deck.config.card_limit + 1 (no game effect here)
			g.events.push(Event::Wait);
			g.events.draw(Draw::next(Pile::Play, Pile::Deck), 0.1);
			calc_jokers(g, LifecycleCtx::PlayingCardAdded { n: 1 });
		}
		_ => {}
	}
}

/// `context.using_consumeable` for a joker acting for itself.
fn using_consumable(
	jokers: &mut [Card], events: &mut EventQueue, ji: usize, j: Joker, kind: Kind,
	hanged_glass: usize,
) {
	match j {
		Joker::Glass if kind == Kind::Tarot(Tarot::HangedMan) && hanged_glass > 0 => {
			jokers[ji].ab.x_mult += 0.75 * num::float(hanged_glass);
			events.push(Event::StatusLater(0.75));
		}
		Joker::FortuneTeller if matches!(kind, Kind::Tarot(_)) => {
			events.push(Event::StatusLater(0.75));
		}
		Joker::Constellation if matches!(kind, Kind::Planet(_)) => {
			jokers[ji].ab.x_mult += 0.1;
			events.push(Event::StatusLater(0.75));
		}
		_ => {}
	}
}

/// `context.discard`, once per discarded card.
fn discard(
	areas: &mut Areas, round: &Round, events: &mut EventQueue, ji: usize, j: Joker,
	blueprinted: bool, d: Discarded,
) -> Option<Effect> {
	use Joker as J;
	let ab = areas.jokers[ji].ab;
	let oc = &areas.hand.cards[d.other];
	match j {
		J::Ramen if !blueprinted => {
			if ab.x_mult - 0.01 <= 1.0 {
				schedule_self_destruct(events, areas.jokers[ji].core.id);
				Some(Effect::EXTRA)
			} else {
				areas.jokers[ji].ab.x_mult -= 0.01;
				Some(Effect {
					extra: true,
					delay: Some(0.2),
					..Effect::NONE
				})
			}
		}
		J::Yorick if !blueprinted => {
			let yorick = &mut areas.jokers[ji].ab;
			if yorick.yorick_discards <= 1.0 {
				yorick.yorick_discards = YORICK_DISCARDS;
				yorick.x_mult += 1.0;
				Some(Effect {
					extra: true,
					delay: Some(0.2),
					..Effect::NONE
				})
			} else {
				yorick.yorick_discards -= 1.0;
				None
			}
		}
		J::Trading if !blueprinted && round.discards_used <= 0.0 && d.n_full == 1 => {
			events.push(Event::EaseDollars(3.0));
			Some(Effect {
				remove: true,
				delay: Some(0.45),
				..Effect::NONE
			})
		}
		J::Castle
			if !oc.core.debuff
				&& oc.is_suit(round.castle_suit, SuitTest::Scoring, areas.smeared())
				&& !blueprinted =>
		{
			areas.jokers[ji].ab.chips += 3.0;
			Some(Effect::EXTRA)
		}
		J::Mail if !oc.core.debuff && oc.effective_rank() == Some(round.mail_rank.rank) => {
			// It eases its dollars itself: the table it returns pays none.
			events.push(Event::EaseDollars(5.0));
			Some(Effect::NONE)
		}
		J::HitTheRoad if !oc.core.debuff && oc.has_rank(Rank::Jack) && !blueprinted => {
			areas.jokers[ji].ab.x_mult += 0.5;
			Some(Effect {
				extra: true,
				delay: Some(0.45),
				..Effect::NONE
			})
		}
		J::GreenJoker if !blueprinted && d.other == d.last => {
			let m = (ab.mult - 1.0).max(0.0);
			areas.jokers[ji].ab.mult = m;
			(m != ab.mult).then_some(Effect::EXTRA)
		}
		J::Faceless if d.other == d.last && d.full_faces >= 3 => {
			events.push(Event::DollarsStatus {
				dollars: 5.0,
				delay: 0.45,
			});
			None
		}
		_ => None,
	}
}

/// `context.end_of_round` for a joker acting for itself.
fn end_of_round(g: &mut Game, ji: usize, j: Joker, game_over: bool) -> Option<Effect> {
	use Joker as J;
	let ab = g.areas.jokers[ji].ab;
	let boss = g.blind.is_boss();
	match j {
		J::Campfire if boss && ab.x_mult > 1.0 => g.areas.jokers[ji].ab.x_mult = 1.0,
		J::Rocket if boss => g.areas.jokers[ji].ab.dollars += 2.0,
		J::TurtleBean => {
			if ab.extra - 1.0 <= 0.0 {
				schedule_self_destruct(&mut g.events, g.areas.jokers[ji].core.id);
			} else {
				g.areas.jokers[ji].ab.extra -= 1.0;
				flow::change_hand_size(&mut g.events, -1.0);
			}
		}
		J::Invisible => g.areas.jokers[ji].ab.invis_rounds += 1.0,
		J::Popcorn => {
			if ab.mult - 4.0 <= 0.0 {
				schedule_self_destruct(&mut g.events, g.areas.jokers[ji].core.id);
			} else {
				g.areas.jokers[ji].ab.mult -= 4.0;
			}
		}
		J::TodoList => {
			let hand = g.hands.random_visible(&mut g.rng, "to_do", ab.to_do);
			g.areas.jokers[ji].ab.to_do = Some(hand);
		}
		J::Egg => {
			g.areas.jokers[ji].ab.extra_value += 3.0;
			Pricing::new(g.shop.discount_percent, &g.areas.jokers)
				.set_cost(&mut g.areas.jokers[ji], Some(Area::Jokers));
		}
		J::Gift => {
			let pricing = Pricing::new(g.shop.discount_percent, &g.areas.jokers);
			for (cards, area) in [
				(&mut g.areas.jokers, Area::Jokers),
				(&mut g.areas.consumables, Area::Consumables),
			] {
				for c in cards {
					c.ab.extra_value += 1.0;
					pricing.set_cost(c, Some(area));
				}
			}
		}
		J::HitTheRoad if ab.x_mult > 1.0 => g.areas.jokers[ji].ab.x_mult = 1.0,
		J::GrosMichel | J::Cavendish => {
			let (key, odds) = if j == J::Cavendish {
				("cavendish", 1000.0)
			} else {
				("gros_michel", 6.0)
			};
			let key = Key::from(key);
			if g.rng.random(&key) < g.run.probabilities / odds {
				schedule_self_destruct(&mut g.events, g.areas.jokers[ji].core.id);
				if j == J::GrosMichel {
					g.history.gros_michel_extinct = true;
				}
			}
		}
		J::MrBones if game_over && g.round.chips / g.blind.chips >= 0.25 => {
			let id = g.areas.jokers[ji].core.id;
			g.events.push(Event::Joker(JokerEvent::DissolveJoker {
				id,
				time_factor: 1.0,
			}));
			return Some(Effect {
				saved: true,
				..Effect::NONE
			});
		}
		_ => return None,
	}
	Some(Effect::EXTRA)
}

pub(crate) fn run_event(g: &mut Game, e: JokerEvent) {
	use JokerEvent as E;
	match e {
		E::ChicotOuter => {
			g.events.push(Event::Joker(E::ChicotDisable));
			g.events.status_extra();
		}
		E::ChicotDisable => {
			round::disable_blind(g);
			g.events.delay(0.4);
		}
		E::Burglar => {
			let d = g.round.discards_left;
			g.events.push(Event::EaseDiscard(-d));
			g.events.push(Event::EaseHands(3.0));
			g.events.status_extra();
		}
		E::RiffRaff { n } => {
			for _ in 0..n {
				pools::create_owned(
					g,
					NewCard::joker(RarityRoll::Fixed(0.0), OwnedArea::Jokers, "rif"),
				);
				g.events.materialize(1.0);
				g.areas.joker_buffer = 0.0;
			}
		}
		E::CartomancerOuter => {
			g.events.push(Event::Create(NewCard::new(
				CardSet::Tarot,
				OwnedArea::Consumables,
				"car",
			)));
			g.events.status_extra();
		}
		E::DaggerSlice { dagger, victim } => {
			g.areas.joker_buffer = 0.0;
			let value = g
				.areas
				.jokers
				.iter()
				.find(|c| c.core.id == victim)
				.expect("the sliced joker stays until this event dissolves it")
				.core
				.sell_cost;
			if let Some(d) = g.areas.jokers.iter_mut().find(|c| c.core.id == dagger) {
				d.ab.mult += value * 2.0;
			}
			g.events.start_dissolve_scaled(victim, 1.6);
		}
		E::MarbleStone => {
			let (suit, rank) = pools::random_front(&mut g.rng, &Key::from("marb_fr"));
			let c = card::make_playing_card(
				&mut g.areas,
				g.run.hands_played,
				g.shop.discount_percent,
				suit,
				rank,
				Enhancement::Stone,
			);
			g.events.materialize(1.0);
			g.areas.play.push(c);
		}
		E::CertificateCard => certificate_card(g),
		E::GlassJokerOuter { id, n } => g.events.push(Event::Joker(E::GlassJokerInner { id, n })),
		E::GlassJokerInner { id, n } => {
			if let Some(c) = g.areas.jokers.iter_mut().find(|c| c.core.id == id) {
				c.ab.x_mult += 0.75 * f64::from(n);
			}
		}
		E::PerkeoCopy => perkeo_copy(g),
		E::EatenOuter { id } => g.events.push_event(
			Event::Joker(E::RemoveJokerLater { id }),
			Trigger::After(0.3),
			Blocks::Barrier,
		),
		E::RemoveJokerLater { id } => {
			if let Some(i) = areas::position_of(&g.areas.jokers, id) {
				let c = g.areas.jokers.remove(i);
				card::remove_owned(g, c);
			}
		}
		E::DissolveJoker { id, time_factor } => g.events.start_dissolve_scaled(id, time_factor),
		E::Unvampire { id } => {
			if let Some(c) = g.areas.playing_cards_mut().find(|c| c.core.id == id) {
				c.vampired = false;
			}
		}
	}
}

/// Body of Perkeo's ending-shop event.
fn perkeo_copy(g: &mut Game) {
	let Some(pick) = areas::pick_card(&mut g.rng, "perkeo", &g.areas.consumables, |_, _| true)
	else {
		return;
	};
	let original = g.areas.consumables[pick].clone();
	let mut c = flow::copy_card(
		&mut g.areas,
		&mut g.history,
		g.shop.discount_percent,
		&g.hands,
		&mut g.rng,
		original,
		// strip_edition=
		false,
	);
	card::set_edition(
		&mut c,
		Some(Edition::Negative),
		None,
		Pricing::new(g.shop.discount_percent, &g.areas.jokers),
		&mut g.areas.limits,
	);
	g.events
		.edition_fx(Some(Edition::Negative), EditionTiming::Immediate);
	card::add_to_deck(g, &mut c);
	g.areas.consumables.push(c);
}

/// Certificate: a random playing card with a random seal, into the hand.
fn certificate_card(g: &mut Game) {
	let (suit, rank) = pools::random_front(&mut g.rng, &Key::from("cert_fr"));
	let mut c = card::make_playing_card(
		&mut g.areas,
		g.run.hands_played,
		g.shop.discount_percent,
		suit,
		rank,
		Enhancement::Base,
	);
	let key = Key::from("certsl");
	let st = g.rng.random(&key);
	c.seal = Some(if st > 0.75 {
		Seal::Red
	} else if st > 0.5 {
		Seal::Blue
	} else if st > 0.25 {
		Seal::Gold
	} else {
		Seal::Purple
	});
	Pricing::new(g.shop.discount_percent, &g.areas.jokers).set_cost(&mut c, None);
	g.events.materialize(1.0);
	CardDebuff::new(&g.blind, &g.areas.jokers).apply(&mut c);
	g.areas.hand.cards.push(c);
	g.areas.hand.sort();
}
