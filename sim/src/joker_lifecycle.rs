//! Joker contexts outside hand scoring: shop, packs, selling, blind setup,
//! discards and the end of the round.

use crate::card::Kind;
use crate::data::{Edition, JACK, Joker, TagKind, Tarot};
use crate::ev::Ev;
use crate::game::{Area, CardSet, Game};
use crate::jokers::{Ctx, Effect, ret};
use crate::rng::Key;
use crate::{blind, cards, events, flow, jokers, num};

#[expect(
	clippy::too_many_lines,
	reason = "one arm per context; the order of the checks is the Lua if-chain's"
)]
pub(crate) fn calculate(
	g: &mut Game, ji: usize, j: Joker, ctx: &Ctx, blueprinted: bool, actor: usize,
) -> Option<Effect> {
	use Joker as J;
	let ab = g.jokers[ji].ab;
	let prob = g.probabilities;
	match *ctx {
		Ctx::OpenBooster => {
			if j == J::Hallucination && jokers::consumable_room(g) {
				let key = g.key_ante("halu");
				if g.rng.random(&key) < prob / 2.0 {
					g.consumeable_buffer += 1.0;
					g.events.push_before(0.0, Ev::Create {
						set: CardSet::Tarot,
						area: Area::Consumables,
						forced: None,
						append: "hal",
						rarity: None,
						legendary: false,
					});
					g.events.status_extra();
				}
			}
			None
		}
		Ctx::SellingSelf => {
			if j == J::Luchador
				&& g.blind.kind.is_some()
				&& !g.blind.disabled
				&& g.blind.slot() == Some(crate::game::BlindSlot::Boss)
			{
				g.events.status_extra();
				blind::disable_blind(g);
			}
			if j == J::DietCola {
				g.events.push(Ev::AddTag {
					kind: TagKind::Double,
					orbital: None,
				});
			}
			if j == J::Invisible && ab.invis_rounds >= 2.0 && !blueprinted {
				let mut sorted: Vec<usize> = (0..g.jokers.len()).filter(|&i| i != ji).collect();
				g.events.status_extra();
				if !sorted.is_empty() && num::float(g.jokers.len()) <= g.joker_limit.limit {
					// pseudorandom_element(jokers) sorts by sort_id.
					sorted.sort_by_key(|&i| g.jokers[i].id);
					let key = Key::from("invisible");
					let pick = sorted[g.rng.element(&key, sorted.len())];
					let strip = g.jokers[pick].edition == Edition::Negative;
					let mut c = flow::copy_card(g, g.jokers[pick].clone(), strip);
					if c.is_joker(J::Invisible) {
						c.ab.invis_rounds = 0.0;
					}
					cards::add_to_deck(g, &mut c, false);
					g.jokers.push(c);
				}
			}
			None
		}
		Ctx::SellingCard => {
			if j == J::Campfire && !blueprinted {
				jokers::jab(g, ji).x_mult += 0.25;
				g.events.push(Ev::StatusLater(0.75));
			}
			None
		}
		Ctx::RerollShop => {
			if j == J::Flash && !blueprinted {
				jokers::jab(g, ji).mult += 2.0;
				g.events.push(Ev::StatusLater(0.75));
			}
			None
		}
		Ctx::EndingShop => {
			if j == J::Perkeo && !g.consumables.is_empty() {
				g.events.push(Ev::PerkeoCopy);
				g.events.status_extra();
			}
			None
		}
		Ctx::SkipBlind => {
			if j == J::Throwback && !blueprinted {
				g.events.push(Ev::StatusLater(0.75));
			}
			None
		}
		Ctx::SkippingBooster => {
			if j == J::RedCard && !blueprinted {
				jokers::jab(g, ji).mult += 3.0;
				g.events.push(Ev::StatusLater(0.45));
			}
			None
		}
		Ctx::PlayingCardAdded { n } => {
			if g.jokers[ji].getting_sliced {
				return None;
			}
			if j == J::Hologram && !blueprinted && n > 0 {
				jokers::jab(g, ji).x_mult += num::float(n) * 0.25;
				g.events.status_extra();
			}
			None
		}
		Ctx::FirstHandDrawn => {
			if j == J::Certificate {
				g.events.push(Ev::CertificateCard);
				jokers::playing_card_joker_effects(g, 1);
			}
			None
		}
		Ctx::SettingBlind { boss } => {
			if g.jokers[ji].getting_sliced {
				return None;
			}
			if j == J::Chicot && !blueprinted && boss {
				g.events.push(Ev::ChicotOuter);
			}
			if j == J::Madness && !blueprinted && !boss {
				jokers::jab(g, ji).x_mult += 0.5;
				let mut dest: Vec<usize> = (0..g.jokers.len())
					.filter(|&i| i != ji && !g.jokers[i].ab.eternal && !g.jokers[i].getting_sliced)
					.collect();
				if !dest.is_empty() {
					dest.sort_by_key(|&i| g.jokers[i].id);
					let key = Key::from("madness");
					let pick = dest[g.rng.element(&key, dest.len())];
					if !g.jokers[actor].getting_sliced {
						g.jokers[pick].getting_sliced = true;
						let id = g.jokers[pick].id;
						g.events.push(Ev::DissolveJoker { id, fac: 1.6 });
					}
				}
				if !g.jokers[actor].getting_sliced {
					g.events.status_extra();
				}
			}
			if j == J::Burglar && !g.jokers[actor].getting_sliced {
				g.events.push(Ev::Burglar);
			}
			if j == J::RiffRaff
				&& !g.jokers[actor].getting_sliced
				&& (num::float(g.jokers.len()) + g.joker_buffer) < g.joker_limit.limit
			{
				let n =
					2f64.min(g.joker_limit.limit - (num::float(g.jokers.len()) + g.joker_buffer));
				g.joker_buffer += n;
				#[expect(
					clippy::cast_possible_truncation,
					clippy::cast_sign_loss,
					reason = "`n` is 1 or 2 free joker slots"
				)]
				let n = n as u32;
				g.events.push(Ev::RiffRaff { n });
				g.events.status_extra();
			}
			if j == J::Cartomancer && !g.jokers[actor].getting_sliced && jokers::consumable_room(g)
			{
				g.consumeable_buffer += 1.0;
				g.events.push(Ev::CartomancerOuter);
			}
			if j == J::Ceremonial
				&& !blueprinted
				&& ji + 1 < g.jokers.len()
				&& !g.jokers[ji + 1].ab.eternal
				&& !g.jokers[ji + 1].getting_sliced
			{
				g.jokers[ji + 1].getting_sliced = true;
				g.joker_buffer -= 1.0;
				let me = g.jokers[ji].id;
				let victim = g.jokers[ji + 1].id;
				g.events.push(Ev::DaggerSlice { dagger: me, victim });
				g.events.status_extra();
			}
			if j == J::Marble && !g.jokers[actor].getting_sliced {
				g.events.push(Ev::MarbleStone);
				g.events.status_extra();
				// G.deck.config.card_limit + 1 (no game effect here)
				g.events.push(Ev::Wait);
				g.events.draw_ev(
					crate::ev::Pile::Play,
					crate::ev::Pile::Deck,
					None,
					false,
					false,
					0.1,
				);
				jokers::playing_card_joker_effects(g, 1);
			}
			None
		}
		Ctx::RemovePlayingCards { faces, glass } => {
			if j == J::Caino && !blueprinted {
				if faces > 0 {
					jokers::jab(g, ji).caino_xmult += num::float(faces) * 1.0;
					g.events.push(Ev::StatusLater(0.75));
				}
				return None;
			}
			if j == J::Glass && !blueprinted {
				if glass > 0 {
					g.events.push(Ev::GlassJokerOuter {
						id: g.jokers[ji].id,
						n: num::index32(glass),
					});
				}
				return None;
			}
			None
		}
		Ctx::UsingConsumeable { kind, hanged_glass } => {
			if j == J::Glass && !blueprinted && kind == Kind::Tarot(Tarot::HangedMan) {
				if hanged_glass > 0 {
					jokers::jab(g, ji).x_mult += 0.75 * num::float(hanged_glass);
					g.events.push(Ev::StatusLater(0.75));
				}
				return None;
			}
			if j == J::FortuneTeller && !blueprinted && matches!(kind, Kind::Tarot(_)) {
				g.events.push(Ev::StatusLater(0.75));
			}
			if j == J::Constellation && !blueprinted && matches!(kind, Kind::Planet(_)) {
				jokers::jab(g, ji).x_mult += 0.1;
				g.events.push(Ev::StatusLater(0.75));
				return None;
			}
			None
		}
		Ctx::PreDiscard { hook } => {
			if j == J::Burnt
				&& g.discards_used <= 0.0
				&& !hook
				&& let Some(h) = g.burnt_hand
			{
				g.events.status_extra();
				g.events.hand_text(0.3);
				events::level_up_hand_ev(g, h, 1.0, false);
				g.events.hand_text(0.0);
			}
			None
		}
		Ctx::Discard {
			other,
			last,
			n_full,
			full_faces,
		} => {
			let oc = &g.hand[other];
			if j == J::Ramen && !blueprinted {
				if ab.x_mult - 0.01 <= 1.0 {
					jokers::schedule_self_destruct(g, ji);
					ret!(extra: true);
				}
				jokers::jab(g, ji).x_mult -= 0.01;
				ret!(extra: true, delay: Some(0.2));
			}
			if j == J::Yorick && !blueprinted {
				if ab.yorick_discards <= 1.0 {
					jokers::jab(g, ji).yorick_discards = 23.0;
					jokers::jab(g, ji).x_mult += 1.0;
					ret!(extra: true, delay: Some(0.2));
				}
				jokers::jab(g, ji).yorick_discards -= 1.0;
				return None;
			}
			if j == J::Trading && !blueprinted && g.discards_used <= 0.0 && n_full == 1 {
				g.events.push(Ev::EaseDollars(3.0));
				ret!(remove: true, delay: Some(0.45));
			}
			let smeared = g.smeared();
			if j == J::Castle
				&& !oc.debuff
				&& oc.is_suit(g.castle_suit, false, false, smeared)
				&& !blueprinted
			{
				jokers::jab(g, ji).chips += 3.0;
				ret!(extra: true);
			}
			let oc = &g.hand[other];
			if j == J::Mail && !oc.debuff && oc.get_id() == Some(g.mail_rank.rank) {
				g.events.push(Ev::EaseDollars(5.0));
				ret!(dollars: None);
			}
			let oc = &g.hand[other];
			if j == J::HitTheRoad && !oc.debuff && oc.id_is(JACK) && !blueprinted {
				jokers::jab(g, ji).x_mult += 0.5;
				ret!(extra: true, delay: Some(0.45));
			}
			if j == J::GreenJoker && !blueprinted && other == last {
				let prev = ab.mult;
				let m = (ab.mult - 1.0).max(0.0);
				jokers::jab(g, ji).mult = m;
				#[expect(
					clippy::float_cmp,
					reason = "exact: whether the floor at 0 left the value unchanged"
				)]
				let changed = m != prev;
				if changed {
					ret!(extra: true);
				}
			}
			if j == J::Faceless && other == last && full_faces >= 3 {
				g.events.push(Ev::DollarsStatus {
					d: 5.0,
					delay: 0.45,
				});
				return None;
			}
			None
		}
		Ctx::EndOfRound { game_over } => {
			if blueprinted {
				return None;
			}
			let boss = g.blind.boss;
			if j == J::Campfire && boss && ab.x_mult > 1.0 {
				jokers::jab(g, ji).x_mult = 1.0;
				ret!(extra: true);
			}
			if j == J::Rocket && boss {
				jokers::jab(g, ji).dollars += 2.0;
				ret!(extra: true);
			}
			if j == J::TurtleBean {
				if ab.remaining - 1.0 <= 0.0 {
					jokers::schedule_self_destruct(g, ji);
					ret!(extra: true);
				}
				jokers::jab(g, ji).remaining -= 1.0;
				cards::change_hand_size(g, -1.0);
				ret!(extra: true);
			}
			if j == J::Invisible {
				jokers::jab(g, ji).invis_rounds += 1.0;
				ret!(extra: true);
			}
			if j == J::Popcorn {
				if ab.mult - 4.0 <= 0.0 {
					jokers::schedule_self_destruct(g, ji);
					ret!(extra: true);
				}
				jokers::jab(g, ji).mult -= 4.0;
				ret!(extra: true);
			}
			if j == J::TodoList {
				let (hands, n) = g.visible_hands_pairs(ab.to_do);
				let key = Key::from("to_do");
				let i = g.rng.element(&key, n);
				jokers::jab(g, ji).to_do = Some(hands[i]);
				ret!(extra: true);
			}
			if j == J::Egg {
				jokers::jab(g, ji).extra_value += 3.0;
				cards::reprice(g, Area::Jokers, ji);
				ret!(extra: true);
			}
			if j == J::Gift {
				for area in [Area::Jokers, Area::Consumables] {
					for i in 0..g.area(area).len() {
						g.area_mut(area)[i].ab.extra_value += 1.0;
						cards::reprice(g, area, i);
					}
				}
				ret!(extra: true);
			}
			if j == J::HitTheRoad && ab.x_mult > 1.0 {
				jokers::jab(g, ji).x_mult = 1.0;
				ret!(extra: true);
			}
			if j == J::GrosMichel || j == J::Cavendish {
				let (key, odds) = if j == J::Cavendish {
					("cavendish", 1000.0)
				} else {
					("gros_michel", 6.0)
				};
				let key = Key::from(key);
				if g.rng.random(&key) < prob / odds {
					jokers::schedule_self_destruct(g, ji);
					if j == J::GrosMichel {
						g.gros_michel_extinct = true;
					}
				}
				ret!(extra: true);
			}
			if j == J::MrBones && game_over && g.chips / g.blind.chips >= 0.25 {
				let id = g.jokers[ji].id;
				g.events.push(Ev::DissolveJoker { id, fac: 1.0 });
				ret!(saved: true);
			}
			None
		}
		Ctx::EndOfRoundRepetition { effects } => {
			if j == J::Mime && effects {
				ret!(repetitions: 1);
			}
			None
		}
		_ => unreachable!("scoring context {ctx:?}"),
	}
}
