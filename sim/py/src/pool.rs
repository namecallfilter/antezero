//! Replay `.bal` recordings into `SimGame` start states for `VecEnv`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::SplitWhitespace;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sim::game::{Game as SimGame, HandSort};
use sim::{Action, CardMask, Stake, json, num};

use crate::game::parse_back;

fn resolve_pool_dir(path: &str) -> PyResult<PathBuf> {
	let p = Path::new(path);
	if p.is_dir() {
		return Ok(p.to_path_buf());
	}
	let agent_p = Path::new("agent").join(path);
	if agent_p.is_dir() {
		return Ok(agent_p);
	}
	Err(PyValueError::new_err(format!(
		"start_pool directory not found: {path:?}"
	)))
}

fn mask_from_tokens<'a>(tokens: impl Iterator<Item = &'a str>) -> Option<CardMask> {
	let mut mask = CardMask::EMPTY;
	for tok in tokens {
		let idx: usize = tok.parse().ok()?;
		if idx >= CardMask::CAPACITY {
			return None;
		}
		mask.insert(idx);
	}
	Some(mask)
}

fn parse_bal_action(line: &str) -> Option<Action> {
	let mut it = line.split_whitespace();
	let op = it.next()?;
	let next_u8 = |it: &mut SplitWhitespace<'_>| -> Option<u8> { it.next()?.parse().ok() };
	match op {
		"play" => Some(Action::Play(mask_from_tokens(it)?)),
		"discard" => Some(Action::Discard(mask_from_tokens(it)?)),
		"use" => Some(Action::UseConsumable {
			index: next_u8(&mut it)?,
			targets: mask_from_tokens(it)?,
		}),
		"buy_use" => Some(Action::BuyAndUse {
			index: next_u8(&mut it)?,
			targets: mask_from_tokens(it)?,
		}),
		"pick" => Some(Action::PickPackCard {
			index: next_u8(&mut it)?,
			targets: mask_from_tokens(it)?,
		}),
		"sell" => Some(Action::SellJoker(next_u8(&mut it)?)),
		"sell_cons" => Some(Action::SellConsumable(next_u8(&mut it)?)),
		"sort_rank" => Some(Action::SortHand(HandSort::Rank)),
		"sort_suit" => Some(Action::SortHand(HandSort::Suit)),
		_ => line.parse().ok(),
	}
}

fn untargeted_fallback(act: Action) -> Option<Action> {
	match act {
		Action::UseConsumable { index, targets } if targets != CardMask::EMPTY => {
			Some(Action::UseConsumable {
				index,
				targets: CardMask::EMPTY,
			})
		}
		Action::BuyAndUse { index, targets } if targets != CardMask::EMPTY => {
			Some(Action::BuyAndUse {
				index,
				targets: CardMask::EMPTY,
			})
		}
		Action::PickPackCard { index, targets } if targets != CardMask::EMPTY => {
			Some(Action::PickPackCard {
				index,
				targets: CardMask::EMPTY,
			})
		}
		_ => None,
	}
}

fn step_recording_action(g: &mut SimGame, act: Action) -> bool {
	if g.step(act).is_ok() {
		return true;
	}
	untargeted_fallback(act).is_some_and(|fallback| g.step(fallback).is_ok())
}

fn parse_header(line: &str) -> Option<(&str, &str, u8)> {
	let parts: Vec<&str> = line.split_whitespace().collect();
	if parts.len() != 6 || parts[0] != "seed" || parts[2] != "deck" || parts[4] != "stake" {
		return None;
	}
	let stake: u8 = parts[5].parse().ok()?;
	Some((parts[1], parts[3], stake))
}

fn push_snapshot(games: &mut Vec<SimGame>, g: &SimGame) {
	if !g.is_terminal() {
		let mut snap = g.clone();
		snap.flow.rearranges = 0;
		games.push(snap);
	}
}

/// Loads `SimGame` start states from a pool directory (`index.jsonl` +
/// `recordings/*.bal`). Unreadable or unsupported recording entries (such as
/// unsimulated challenge runs) are skipped and the loaded count is logged.
pub(crate) fn load_start_pool(path: &str) -> PyResult<Vec<SimGame>> {
	let dir = resolve_pool_dir(path)?;
	let index_path = dir.join("index.jsonl");
	let index_text = fs::read_to_string(&index_path)
		.map_err(|e| PyValueError::new_err(format!("cannot read {}: {e}", index_path.display())))?;

	let mut by_rec: BTreeMap<String, Vec<usize>> = BTreeMap::new();
	let mut total = 0usize;
	for raw in index_text.lines() {
		let line = raw.trim();
		if line.is_empty() {
			continue;
		}
		let Ok(doc) = json::parse(line) else {
			continue;
		};
		let Some(rec) = doc.get("rec").as_str() else {
			continue;
		};
		let Some(k) = doc
			.get("k")
			.as_f64()
			.and_then(num::whole)
			.and_then(|v| usize::try_from(v).ok())
		else {
			continue;
		};
		by_rec.entry(rec.to_string()).or_default().push(k);
		total += 1;
	}

	let rec_dir = dir.join("recordings");
	let mut games = Vec::with_capacity(total);
	for (rec, mut ks) in by_rec {
		ks.sort_unstable();
		let max_k = *ks.last().unwrap_or(&0);
		let rec_path = rec_dir.join(&rec);
		let Ok(text) = fs::read_to_string(&rec_path) else {
			continue;
		};
		let mut lines = text.lines();
		let Some(header) = lines.next().map(str::trim) else {
			continue;
		};
		let Some((seed, deck, stake_u8)) = parse_header(header) else {
			continue;
		};
		let Ok(back) = parse_back(deck) else {
			continue;
		};
		let Ok(stake) = Stake::try_from(stake_u8) else {
			continue;
		};
		let mut g = SimGame::new_run(seed, back, stake);
		g.set_tracing(false);

		let mut ki = 0usize;
		while ki < ks.len() && ks[ki] == 0 {
			push_snapshot(&mut games, &g);
			ki += 1;
		}
		let mut step_num = 0usize;
		for raw in lines {
			let line = raw.split('#').next().unwrap_or("").trim();
			if line.is_empty() {
				continue;
			}
			step_num += 1;
			if step_num > max_k || ki >= ks.len() {
				break;
			}
			let optional = line.starts_with('?');
			let act_line = if optional {
				line.trim_start_matches('?').trim_start()
			} else {
				line
			};
			let Some(act) = parse_bal_action(act_line) else {
				if optional {
					continue;
				}
				break;
			};
			if !step_recording_action(&mut g, act) && !optional {
				break;
			}
			while ki < ks.len() && ks[ki] == step_num {
				push_snapshot(&mut games, &g);
				ki += 1;
			}
		}
	}
	eprintln!(
		"[vecenv] start_pool: loaded {}/{total} states from {}",
		games.len(),
		dir.display()
	);
	Ok(games)
}
