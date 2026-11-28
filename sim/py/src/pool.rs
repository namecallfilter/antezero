//! Reads a start-pool directory (`index.jsonl` and `recordings/*.bal`) into
//! the start states of `VecEnv`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use sim::balatro::load_starts;
use sim::game::Game as SimGame;
use sim::num;

/// The start states of the pool directory `path`: for each line
/// `{"rec": name, "k": step}` of its `index.jsonl`, the game after `step`
/// action lines of `recordings/<name>` (see [`load_starts`]), with its
/// rearrangement count cleared.
///
/// # Errors
///
/// Returns a `ValueError` that names the file and what is wrong with it if
/// the index or a recording cannot be read, an index line is not such an
/// object, or a recording does not replay to a start state for each of its
/// steps.
pub(crate) fn load_start_pool(path: &str) -> PyResult<Vec<SimGame>> {
	use sim::json::ValueExt;

	let dir = Path::new(path);
	let index_path = dir.join("index.jsonl");
	let cannot_read =
		|file: &Path, e| PyValueError::new_err(format!("cannot read {}: {e}", file.display()));
	let index = fs::read_to_string(&index_path).map_err(|e| cannot_read(&index_path, e))?;

	let mut by_rec: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
	let entries: Vec<Value> = index
		.lines()
		.enumerate()
		.map(|(i, line)| {
			serde_json::from_str(line).map_err(|e| {
				PyValueError::new_err(format!("{} line {}: {e}", index_path.display(), i + 1))
			})
		})
		.collect::<PyResult<_>>()?;
	for (i, entry) in entries.iter().enumerate() {
		let lacks = |what| {
			PyValueError::new_err(format!(
				"{} line {} has no {what}",
				index_path.display(),
				i + 1
			))
		};
		let rec = entry["rec"]
			.as_str()
			.ok_or_else(|| lacks("recording name \"rec\""))?;
		let step = entry["k"]
			.number()
			.and_then(num::whole)
			.and_then(|k| usize::try_from(k).ok())
			.ok_or_else(|| lacks("step \"k\", a whole number from 0"))?;
		by_rec.entry(rec).or_default().push(step);
	}

	let mut games = Vec::with_capacity(entries.len());
	for (rec, mut steps) in by_rec {
		steps.sort_unstable();
		let rec_path = dir.join("recordings").join(rec);
		let script = fs::read_to_string(&rec_path).map_err(|e| cannot_read(&rec_path, e))?;
		let points = load_starts(&script, &steps)
			.map_err(|e| PyValueError::new_err(format!("{}: {e}", rec_path.display())))?;
		games.extend(points.into_iter().map(|point| {
			let mut game = point.game;
			game.flow.rearranges = 0;
			game
		}));
	}
	Ok(games)
}
