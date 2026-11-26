//! Per-interval hand, joker, consumable, and best-score statistics for `VecEnv`.

use std::collections::HashMap;
use std::iter;
use std::ops::{Index, IndexMut};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sim::card::{AnyCard, Kind};
use sim::data::{ALL_JOKERS, Back, HandType, JOKER_COUNT};
use sim::game::Game as SimGame;
use sim::{Action, num};

const HAND_COUNT: usize = HandType::ALL.len();
const JOKER_WORDS: usize = JOKER_COUNT.div_ceil(u64::BITS as usize);

#[derive(Clone, Copy, Debug)]
pub(crate) enum ConsEvent {
	Used = 0,
	BuyUse = 1,
	PackPick = 2,
	Bought = 3,
	Sold = 4,
}

impl ConsEvent {
	const COUNT: usize = 5;
}

#[derive(Clone, Copy, Default)]
struct ConsCounts([u64; ConsEvent::COUNT]);

impl Index<ConsEvent> for ConsCounts {
	type Output = u64;

	fn index(&self, ev: ConsEvent) -> &u64 {
		&self.0[ev as usize]
	}
}

impl IndexMut<ConsEvent> for ConsCounts {
	fn index_mut(&mut self, ev: ConsEvent) -> &mut u64 {
		&mut self.0[ev as usize]
	}
}

#[derive(Clone, Copy, Default)]
pub(crate) struct HandCounts<T>([T; HAND_COUNT]);

impl<T> Index<HandType> for HandCounts<T> {
	type Output = T;

	fn index(&self, ht: HandType) -> &T {
		&self.0[ht as usize]
	}
}

impl<T> IndexMut<HandType> for HandCounts<T> {
	fn index_mut(&mut self, ht: HandType) -> &mut T {
		&mut self.0[ht as usize]
	}
}

/// Bitset of jokers held at any point during an episode.
#[derive(Clone, Copy, Default)]
pub(crate) struct JokerSet([u64; JOKER_WORDS]);

impl JokerSet {
	pub(crate) fn mark(&mut self, g: &SimGame) {
		for c in &g.areas.jokers {
			if let Kind::Joker(j) = c.kind {
				let idx = j as usize;
				self.0[idx / 64] |= 1u64 << (idx % 64);
			}
		}
	}

	fn indices(self) -> impl Iterator<Item = usize> {
		self.0.into_iter().enumerate().flat_map(|(w_idx, mut w)| {
			iter::from_fn(move || {
				if w == 0 {
					return None;
				}
				let bit = num::index(w.trailing_zeros());
				w &= w - 1;
				Some(w_idx * 64 + bit)
			})
		})
	}
}

pub(crate) struct EpisodeStatsOut {
	pub(crate) back: Back,
	pub(crate) stake: u8,
	pub(crate) ante: i32,
	pub(crate) won: bool,
	pub(crate) truncated: bool,
	pub(crate) blinds_cleared: u32,
	pub(crate) jokers_seen: JokerSet,
	pub(crate) hand_plays: HandCounts<u32>,
	pub(crate) best_hand_score: f64,
}

pub(crate) struct IntervalStats {
	hand_plays: HandCounts<u64>,
	hand_wins: HandCounts<u64>,
	// `(card_key, set_name) -> [used, buy_use, pack_pick, bought, sold]`.
	consumables: HashMap<(&'static str, &'static str), ConsCounts>,
	// `joker_idx -> [runs, wins, losses, ante_sum, blinds]`.
	joker_runs: [u64; JOKER_COUNT],
	joker_wins: [u64; JOKER_COUNT],
	joker_losses: [u64; JOKER_COUNT],
	joker_ante_sum: [i64; JOKER_COUNT],
	joker_blinds: [u64; JOKER_COUNT],
	// `back_key -> (best_stake, best_score)`.
	best: HashMap<&'static str, (u8, f64)>,
}

impl Default for IntervalStats {
	fn default() -> Self {
		Self {
			hand_plays: HandCounts::default(),
			hand_wins: HandCounts::default(),
			consumables: HashMap::new(),
			joker_runs: [0; JOKER_COUNT],
			joker_wins: [0; JOKER_COUNT],
			joker_losses: [0; JOKER_COUNT],
			joker_ante_sum: [0; JOKER_COUNT],
			joker_blinds: [0; JOKER_COUNT],
			best: HashMap::new(),
		}
	}
}

impl IntervalStats {
	pub(crate) fn record(
		&mut self, cons_event: Option<((&'static str, &'static str), ConsEvent)>,
		ep_stats: Option<EpisodeStatsOut>,
	) {
		if let Some((key, ev)) = cons_event {
			self.consumables.entry(key).or_default()[ev] += 1;
		}
		let Some(st) = ep_stats else {
			return;
		};
		for ht in HandType::ALL {
			let plays = u64::from(st.hand_plays[ht]);
			self.hand_plays[ht] += plays;
			if st.won {
				self.hand_wins[ht] += plays;
			}
		}
		for j_idx in st.jokers_seen.indices() {
			self.joker_runs[j_idx] += 1;
			if st.won {
				self.joker_wins[j_idx] += 1;
			} else if !st.truncated {
				self.joker_losses[j_idx] += 1;
			}
			self.joker_ante_sum[j_idx] += i64::from(st.ante);
			self.joker_blinds[j_idx] += u64::from(st.blinds_cleared);
		}
		let b_entry = self.best.entry(st.back.key()).or_insert((0, 0.0));
		if st.won && st.stake > b_entry.0 {
			b_entry.0 = st.stake;
		}
		if st.best_hand_score > b_entry.1 {
			b_entry.1 = st.best_hand_score;
		}
	}

	pub(crate) fn into_py_dict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
		let d = PyDict::new(py);
		let hands: Vec<(&'static str, u64, u64)> = HandType::ALL
			.iter()
			.map(|&ht| (ht.name(), self.hand_plays[ht], self.hand_wins[ht]))
			.collect();
		d.set_item("hands", hands)?;
		let jokers = PyDict::new(py);
		for (idx, j) in ALL_JOKERS.iter().enumerate() {
			let runs = self.joker_runs[idx];
			if runs > 0 {
				jokers.set_item(
					j.key(),
					(
						runs,
						self.joker_wins[idx],
						self.joker_losses[idx],
						self.joker_ante_sum[idx],
						self.joker_blinds[idx],
					),
				)?;
			}
		}
		d.set_item("jokers", jokers)?;
		let cons = PyDict::new(py);
		for ((card, set_name), counts) in self.consumables {
			cons.set_item((card, set_name), counts.0)?;
		}
		d.set_item("consumables", cons)?;
		let best = PyDict::new(py);
		for (deck, vals) in self.best {
			best.set_item(deck, vals)?;
		}
		d.set_item("best", best)?;
		Ok(d)
	}
}

fn kind_consumable_info(kind: Kind) -> Option<(&'static str, &'static str)> {
	match kind {
		Kind::Tarot(t) => Some((t.key(), "tarot")),
		Kind::Planet(p) => Some((p.key(), "planet")),
		Kind::Spectral(s) => Some((s.key(), "spectral")),
		_ => None,
	}
}

fn any_card_consumable_info(card: &AnyCard) -> Option<(&'static str, &'static str)> {
	match card {
		AnyCard::Card(c) => kind_consumable_info(c.kind),
		AnyCard::Playing(_) => None,
	}
}

pub(crate) fn consumable_event(
	g: &SimGame, act: Action,
) -> Option<((&'static str, &'static str), ConsEvent)> {
	match act {
		Action::UseConsumable { index, .. } => g
			.areas
			.consumables
			.get(usize::from(index))
			.and_then(|c| kind_consumable_info(c.kind))
			.map(|info| (info, ConsEvent::Used)),
		Action::BuyAndUse { index, .. } => g
			.areas
			.shop_jokers
			.get(usize::from(index))
			.and_then(any_card_consumable_info)
			.map(|info| (info, ConsEvent::BuyUse)),
		Action::PickPackCard { index, .. } => g
			.areas
			.pack_cards
			.get(usize::from(index))
			.and_then(any_card_consumable_info)
			.map(|info| (info, ConsEvent::PackPick)),
		Action::BuyShopCard(index) => g
			.areas
			.shop_jokers
			.get(usize::from(index))
			.and_then(any_card_consumable_info)
			.map(|info| (info, ConsEvent::Bought)),
		Action::SellConsumable(index) => g
			.areas
			.consumables
			.get(usize::from(index))
			.and_then(|c| kind_consumable_info(c.kind))
			.map(|info| (info, ConsEvent::Sold)),
		_ => None,
	}
}
