//! `Game:start_run` / `init_game_object` / `Back:apply_to_run`.

use crate::card::Kind;
use crate::consumables::CEv;
use crate::data::{
	ACE, Back, BlindKind, CenterSet, Enhancement, HandType, JACK, KING, P_CARDS_SORTED, QUEEN,
	Rank, Spectral, Suit, Tarot, Voucher, rank_char,
};
use crate::ev::Ev;
use crate::events::{EventQueue, St};
use crate::game::{
	Area, Blind, BlindSlot, BlindStatus, CardSet, Game, HandLevel, IdolCard, Limit, MailRank,
	Modifiers, PackSlot, Phase, StartingParams,
};
use crate::rng::{Key, PseudoRandom};
use crate::{blind, cards, events, flow, num, pools, round_end, shop};

/// Longest supported seed (the game's own seeds have 8 characters; stream
/// keys are the stream name followed by the seed).
pub const MAX_SEED_LEN: usize = 32;

/// See [`Game::new_run`].
pub(crate) fn new_run(seed: &str, back: Back, stake: u8) -> Game {
	assert!(
		seed.len() <= MAX_SEED_LEN,
		"seed longer than {MAX_SEED_LEN} bytes"
	);
	assert!((1..=8).contains(&stake), "stake {stake} is not 1..=8");
	let mut g = blank(seed, back, stake);
	events::begin_action(&mut g);
	start_run(&mut g);
	events::run_queue(&mut g);
	g
}

#[expect(
	clippy::too_many_lines,
	reason = "one line per field of the game state"
)]
fn blank(seed: &str, back: Back, stake: u8) -> Game {
	let mut hands = [HandLevel {
		level: 1.0,
		chips: 0.0,
		mult: 0.0,
		played: 0.0,
		played_this_round: 0.0,
		visible: false,
	}; 12];
	for h in HandType::ALL {
		let base = h.base();
		hands[h as usize] = HandLevel {
			level: 1.0,
			chips: base.chips,
			mult: base.mult,
			played: 0.0,
			played_this_round: 0.0,
			visible: base.visible,
		};
	}
	Game {
		rng: PseudoRandom::new(seed),
		back,
		stake,
		modifiers: Modifiers::default(),
		starting_params: StartingParams::default(),
		phase: Phase::BlindSelect,
		won: false,
		endless: false,
		blinds_cleared: 0,
		win_notified: false,
		pause_rest: false,
		scored: false,
		hand_text_cleared: false,
		next_sort_id: 0,
		deck: Vec::with_capacity(64),
		hand: Vec::with_capacity(16),
		play: Vec::with_capacity(8),
		discard: Vec::with_capacity(64),
		jokers: Vec::with_capacity(8),
		consumables: Vec::with_capacity(4),
		shop_jokers: Vec::with_capacity(4),
		shop_vouchers: Vec::with_capacity(2),
		shop_boosters: Vec::with_capacity(2),
		pack_cards: Vec::with_capacity(5),
		joker_limit: Limit::new(5.0),
		consumable_limit: Limit::new(2.0),
		hand_limit: Limit::new(8.0),
		hand_sort_by_suit: false,
		pack: None,
		ante: 1,
		blind_ante: 1,
		win_ante: 8,
		round: 0,
		dollars: 0.0,
		chips: 0.0,
		hands_left: 0.0,
		discards_left: 0.0,
		hands_played_round: 0.0,
		discards_used: 0.0,
		hands_played: 0.0,
		unused_discards: 0.0,
		skips: 0.0,
		reroll_cost: 5.0,
		reroll_cost_increase: 0.0,
		free_rerolls: 0.0,
		used_packs: [PackSlot::Unrolled; 2],
		current_voucher: None,
		voucher_seen: false,
		rearranges: 0,
		allow_skip: true,
		steamodded: false,
		idol_card: IdolCard {
			rank: ACE,
			suit: Suit::Spades,
			id: None,
		},
		mail_rank: MailRank {
			rank: ACE,
			id: None,
		},
		ancient_suit: Some(Suit::Spades),
		castle_suit: Suit::Spades,
		most_played_hand: HandType::HighCard,
		round_dollars: 0.0,
		rr_hands: 1.0,
		rr_discards: 1.0,
		rr_reroll_cost: 1.0,
		temp_reroll_cost: None,
		temp_handsize: None,
		blind_states: [
			BlindStatus::Select,
			BlindStatus::Upcoming,
			BlindStatus::Upcoming,
		],
		boss_choice: BlindKind::Small,
		blind_tags: [None, None],
		boss_rerolled: false,
		selected_blind: None,
		blind_on_deck: None,
		round_bonus_next_hands: 0.0,
		round_bonus_discards: 0.0,
		hands,
		probabilities: 1.0,
		bosses_used: [0; 30],
		used_jokers: CenterSet::default(),
		used_vouchers: 0,
		banned: CenterSet::default(),
		banned_tags: 0,
		banned_blinds: 0,
		gros_michel_extinct: false,
		planets_used: 0,
		consumable_usage_tarot: 0.0,
		consumable_usage_planet: 0.0,
		consumable_usage_spectral: 0.0,
		last_tarot_planet: None,
		last_hand_played: None,
		tags: Vec::new(),
		next_tag_id: 0,
		shop_joker_max: 2.0,
		edition_rate: 1.0,
		joker_rate: 20.0,
		tarot_rate: 4.0,
		planet_rate: 4.0,
		spectral_rate: 0.0,
		playing_card_rate: 0.0,
		consumeable_buffer: 0.0,
		joker_buffer: 0.0,
		dollar_buffer: 0.0,
		discount_percent: 0.0,
		interest_cap: 25.0,
		interest_amount: 1.0,
		bankrupt_at: 0.0,
		perishable_rounds: 5.0,
		rental_rate: 3.0,
		ecto_minus: 1.0,
		pack_size: 2,
		pack_choices: 0,
		first_shop_buffoon: false,
		orbital_choices: Vec::new(),
		shop_free: false,
		shop_d6ed: false,
		starting_deck_size: 52.0,
		last_blind_boss: false,
		facing_blind: false,
		blind: Blind::empty(),
		state: St::BlindSelect,
		state_complete: false,
		events: EventQueue::new(),
		consumed: Vec::new(),
		in_flight: None,
		pending_highlight: Vec::new(),
		pack_interrupt: None,
		end_round_pending: false,
		burnt_hand: None,
		shop_open: false,
		shop_exists_at_open: false,
		ankh_deletable: Vec::new(),
		last_scoring: Vec::new(),
		last_hand_score: 0.0,
	}
}

fn start_run(g: &mut Game) {
	let stake = g.stake;
	if stake >= 2 {
		g.modifiers.no_blind_reward[BlindSlot::Small as usize] = true;
	}
	if stake >= 3 {
		g.modifiers.scaling = 2;
	}
	if stake >= 4 {
		g.modifiers.enable_eternals_in_shop = true;
	}
	if stake >= 5 {
		g.starting_params.discards -= 1.0;
	}
	if stake >= 6 {
		g.modifiers.scaling = 3;
	}
	if stake >= 7 {
		g.modifiers.enable_perishables_in_shop = true;
	}
	if stake >= 8 {
		g.modifiers.enable_rentals_in_shop = true;
	}
	apply_back(g);

	let sp = g.starting_params;
	g.rr_hands = sp.hands;
	g.rr_discards = sp.discards;
	g.dollars = sp.dollars;
	g.rr_reroll_cost = sp.reroll_cost;
	g.reroll_cost = sp.reroll_cost;

	g.boss_choice = pools::get_new_boss(g);
	g.current_voucher = Some(pools::get_next_voucher(g, false));
	g.blind_tags[0] = Some(pools::get_next_tag(g, ""));
	g.blind_tags[1] = Some(pools::get_next_tag(g, ""));

	g.consumable_limit = Limit::new(sp.consumable_slots);
	g.joker_limit = Limit::new(sp.joker_slots);
	g.hand_limit = Limit::new(sp.hand_size);

	// Deck: card prototypes sorted by `P_CARDS` key ("C_2" < ... < "S_T").
	let mut protos: Vec<(Suit, Rank)> = Vec::with_capacity(52);
	for &(s, r) in &P_CARDS_SORTED {
		let (s, r) = if sp.erratic_suits_and_ranks {
			let i = g.rng.element_str("erratic", 52);
			P_CARDS_SORTED[i]
		} else {
			(s, r)
		};
		if sp.no_faces && matches!(r, JACK | QUEEN | KING) {
			continue;
		}
		protos.push((s, r));
	}
	protos.sort_by_key(|&(s, r)| (s.letter(), rank_char(r)));
	for (s, r) in protos {
		let mut c = cards::make_playing_card(g, s, r, Enhancement::Base);
		c.face_down = true;
		g.deck.insert(0, c);
	}
	g.starting_deck_size = num::float(g.playing_card_count());
	g.events.delay(0.5);

	g.discards_left = g.rr_discards;
	g.hands_left = g.rr_hands;
	let k = Key::from("shuffle");
	flow::shuffle_deck(g, &k);
	pools::reset_idol_card(g);
	pools::reset_mail_rank(g);
	g.ancient_suit = None;
	pools::reset_ancient_card(g);
	pools::reset_castle_card(g);
	blind::set_blind(g, None);
	round_end::reset_blinds(g);
	// unlock_notify
	g.events.push(Ev::Wait);

	g.state = St::BlindSelect;
	g.state_complete = false;
}

/// `Back:apply_to_run()`.
fn apply_back(g: &mut Game) {
	use Back as B;
	let b = g.back;
	let voucher = |g: &mut Game, v: Voucher| {
		g.set_voucher_used(v);
		shop::apply_voucher(g, v, true);
	};
	match b {
		B::Magic => voucher(g, Voucher::CrystalBall),
		B::Nebula => voucher(g, Voucher::Telescope),
		_ => {}
	}
	if b == B::Blue {
		g.starting_params.hands += 1.0;
	}
	if b == B::Black {
		g.starting_params.hands += -1.0;
	}
	match b {
		B::Magic => {
			for _ in 0..2 {
				g.events.push(Ev::Create {
					set: CardSet::Tarot,
					area: Area::Consumables,
					forced: Some(Kind::Tarot(Tarot::Fool)),
					append: "deck",
					rarity: None,
					legendary: false,
				});
			}
		}
		B::Ghost => g.events.push(Ev::Create {
			set: CardSet::Tarot,
			area: Area::Consumables,
			forced: Some(Kind::Spectral(Spectral::Hex)),
			append: "deck",
			rarity: None,
			legendary: false,
		}),
		_ => {}
	}
	if b == B::Yellow {
		g.starting_params.dollars += 10.0;
	}
	if b == B::Abandoned {
		g.starting_params.no_faces = true;
	}
	if b == B::Ghost {
		g.spectral_rate = 2.0;
	}
	if b == B::Red {
		g.starting_params.discards += 1.0;
	}
	if b == B::Zodiac {
		for v in [
			Voucher::TarotMerchant,
			Voucher::PlanetMerchant,
			Voucher::OverstockNorm,
		] {
			voucher(g, v);
		}
	}
	if b == B::Checkered {
		g.events.push(Ev::Consumable(CEv::Checkered));
	}
	if b == B::Erratic {
		g.starting_params.erratic_suits_and_ranks = true;
	}
	match b {
		B::Black => g.starting_params.joker_slots += 1.0,
		B::Painted => {
			g.starting_params.joker_slots += -1.0;
			g.starting_params.hand_size += 2.0;
		}
		B::Plasma => g.starting_params.ante_scaling = 2.0,
		B::Nebula => g.starting_params.consumable_slots += -1.0,
		B::Green => {
			g.modifiers.no_interest = true;
			g.modifiers.money_per_hand = Some(2.0);
			g.modifiers.money_per_discard = Some(1.0);
		}
		_ => {}
	}
}
