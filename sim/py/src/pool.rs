//! Reads a start-pool directory (`index.jsonl` and `recordings/*.bal`) into
//! the start states of `VecEnv`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sim::balatro::load_starts;
use sim::game::Game as SimGame;
use sim::{json, num};

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

/// The start states of the pool directory at `path`: for each line
/// `{"rec": name, "k": step}` of its `index.jsonl`, the game after `step`
/// action lines of `recordings/<name>` (see [`load_starts`]), with its
/// rearrangement count cleared.
///
/// A recording that cannot be read or does not replay through its last
/// requested step gives no state, and the reason is logged; so is the number
/// of states loaded.
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
	for (rec, mut steps) in by_rec {
		steps.sort_unstable();
		let loaded = fs::read_to_string(rec_dir.join(&rec))
			.map_err(|e| e.to_string())
			.and_then(|text| load_starts(&text, &steps).map_err(|e| e.to_string()));
		match loaded {
			Ok(points) => games.extend(points.into_iter().flatten().map(|point| {
				let mut game = point.game;
				game.flow.rearranges = 0;
				game
			})),
			Err(reason) => eprintln!("[vecenv] start_pool: {rec} skipped: {reason}"),
		}
	}
	eprintln!(
		"[vecenv] start_pool: loaded {}/{total} states from {}",
		games.len(),
		dir.display()
	);
	Ok(games)
}
