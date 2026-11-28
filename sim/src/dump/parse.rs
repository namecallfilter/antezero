//! JSON field extractors, name lookups, and [`DumpError`] for reading state
//! dumps.

use std::error::Error;
use std::fmt;

use serde_json::Value;

use crate::card::Kind;
use crate::data::{ALL_BLINDS, BlindKind, Booster, Joker, Planet, Rank, Spectral, Tarot, Voucher};
use crate::game::State;
use crate::num;

/// Why a state dump does not make a game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpError(pub(super) String);

impl fmt::Display for DumpError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "state dump: {}", self.0)
	}
}

impl Error for DumpError {}

pub(super) fn bad(key: &str, v: &Value) -> DumpError {
	DumpError(format!("bad {key}: {v:?}"))
}

/// The value of `key`, which the dump always writes (as `null` where the
/// game has `nil`).
pub(super) fn at<'j>(j: &'j Value, key: &str) -> Result<&'j Value, DumpError> {
	j.get(key).ok_or_else(|| DumpError(format!("no {key}")))
}

pub(super) fn number(j: &Value, key: &str) -> Result<f64, DumpError> {
	use crate::json::ValueExt;

	let v = at(j, key)?;
	v.number().ok_or_else(|| bad(key, v))
}

/// A number the game leaves `nil` until it is set.
pub(super) fn opt_number(j: &Value, key: &str) -> Result<Option<f64>, DumpError> {
	use crate::json::ValueExt;

	match at(j, key)? {
		Value::Null => Ok(None),
		v => v.number().map(Some).ok_or_else(|| bad(key, v)),
	}
}

pub(super) fn whole<T: TryFrom<i64>>(j: &Value, key: &str, x: f64) -> Result<T, DumpError> {
	num::whole(x)
		.and_then(|n| T::try_from(n).ok())
		.ok_or_else(|| bad(key, &j[key]))
}

pub(super) fn int<T: TryFrom<i64>>(j: &Value, key: &str) -> Result<T, DumpError> {
	whole(j, key, number(j, key)?)
}

pub(super) fn opt_int<T: TryFrom<i64>>(j: &Value, key: &str) -> Result<Option<T>, DumpError> {
	opt_number(j, key)?.map(|x| whole(j, key, x)).transpose()
}

pub(super) fn flag(j: &Value, key: &str) -> Result<bool, DumpError> {
	match at(j, key)? {
		Value::Bool(b) => Ok(*b),
		v => Err(bad(key, v)),
	}
}

/// A flag the game leaves `nil` while it is false.
pub(super) fn opt_flag(j: &Value, key: &str) -> Result<bool, DumpError> {
	match at(j, key)? {
		Value::Null => Ok(false),
		_ => flag(j, key),
	}
}

/// A flag of a table the dump copies from the game as it is (a card's
/// `ability`), where the game leaves it `nil` while it is false: the key is
/// then left out.
pub(super) fn raw_flag(j: &Value, key: &str) -> Result<bool, DumpError> {
	match &j[key] {
		Value::Null => Ok(false),
		Value::Bool(b) => Ok(*b),
		v => Err(bad(key, v)),
	}
}

pub(super) fn object<'j>(j: &'j Value, key: &str) -> Result<&'j Value, DumpError> {
	let v = at(j, key)?;
	v.as_object().map(|_| v).ok_or_else(|| bad(key, v))
}

pub(super) fn text<'j>(j: &'j Value, key: &str) -> Result<&'j str, DumpError> {
	opt_text(j, key)?.ok_or_else(|| bad(key, &j[key]))
}

pub(super) fn opt_text<'j>(j: &'j Value, key: &str) -> Result<Option<&'j str>, DumpError> {
	raw_text(key, at(j, key)?)
}

fn raw_text<'j>(key: &str, v: &'j Value) -> Result<Option<&'j str>, DumpError> {
	match v {
		Value::Null => Ok(None),
		Value::String(s) => Ok(Some(s)),
		v => Err(bad(key, v)),
	}
}

/// The value named by the text at `key`, as `find` looks it up.
pub(super) fn named<T>(
	j: &Value, key: &str, find: impl Fn(&str) -> Option<T>,
) -> Result<T, DumpError> {
	opt_named(j, key, find)?.ok_or_else(|| bad(key, &j[key]))
}

/// Like [`named`], `None` where the game has `nil`.
pub(super) fn opt_named<T>(
	j: &Value, key: &str, find: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, DumpError> {
	named_value(key, at(j, key)?, find)
}

/// Like [`opt_named`] in a table the dump copies from the game as it is
/// (see [`raw_flag`]).
pub(super) fn raw_named<T>(
	j: &Value, key: &str, find: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, DumpError> {
	named_value(key, &j[key], find)
}

fn named_value<T>(
	key: &str, v: &Value, find: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, DumpError> {
	raw_text(key, v)?
		.map(|s| find(s).ok_or_else(|| bad(key, v)))
		.transpose()
}

pub(super) fn list<'j>(j: &'j Value, key: &str) -> Result<&'j [Value], DumpError> {
	match at(j, key)? {
		Value::Array(a) => Ok(a),
		v => Err(bad(key, v)),
	}
}

/// A table of the game's, which Lua writes as `[]` when it is empty.
pub(super) fn entries<'j>(j: &'j Value, key: &str) -> Result<Vec<(&'j str, &'j Value)>, DumpError> {
	match at(j, key)? {
		Value::Object(m) => Ok(m.iter().map(|(k, v)| (k.as_str(), v)).collect()),
		Value::Array(a) if a.is_empty() => Ok(Vec::new()),
		v => Err(bad(key, v)),
	}
}

pub(super) fn by_name<T: Copy>(all: &[T], name: fn(T) -> &'static str, s: &str) -> Option<T> {
	all.iter().copied().find(|&x| name(x) == s)
}

/// The card that is not a playing card whose center has the key `k`.
pub(super) fn kind_of(k: &str) -> Option<Kind> {
	Joker::from_key(k)
		.map(Kind::Joker)
		.or_else(|| Tarot::from_key(k).map(Kind::Tarot))
		.or_else(|| Planet::from_key(k).map(Kind::Planet))
		.or_else(|| Spectral::from_key(k).map(Kind::Spectral))
		.or_else(|| Voucher::from_key(k).map(Kind::Voucher))
		.or_else(|| Booster::from_key(k).map(Kind::Booster))
}

pub(super) fn rank_of_value(s: &str) -> Option<Rank> {
	by_name(&Rank::ALL, Rank::value, s)
}

pub(super) fn blind_of_name(s: &str) -> Option<BlindKind> {
	by_name(&ALL_BLINDS, BlindKind::name, s)
}

pub(super) fn state_of_name(s: &str) -> Option<State> {
	by_name(&State::ALL, State::name, s)
}
