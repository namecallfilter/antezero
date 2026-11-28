//! The state dumps' conventions on top of JSON: `null` is Lua's `nil`, and
//! the numbers JSON cannot hold are strings (`"inf"`, `"-inf"`, `"nan"`).

use serde_json::Value;

use crate::num;

/// `x` as the harness writes a number (`%.17g`): a whole number below 10^17
/// as an integer and any other as a float, an infinity or NaN as a string,
/// and `None` as `null`. JSON values compare numbers by kind (`1 != 1.0`),
/// so a written dump equals the game's only if the kinds match too.
#[must_use]
pub fn number(x: impl Into<Option<f64>>) -> Value {
	let Some(x) = x.into() else {
		return Value::Null;
	};
	if x.is_nan() {
		"nan".into()
	} else if x.is_infinite() {
		if x > 0.0 { "inf" } else { "-inf" }.into()
	} else {
		match num::whole(x) {
			// `%.17g` gives larger numbers an exponent, and `-0` reads back
			// as a float.
			Some(n) if x.abs() < 1e17 && (n != 0 || x.is_sign_positive()) => n.into(),
			_ => x.into(),
		}
	}
}

/// A dump's values read with its conventions.
pub trait ValueExt {
	/// The number, also one written as `"inf"`, `"-inf"` or `"nan"`.
	#[must_use]
	fn number(&self) -> Option<f64>;

	/// A flag: `null` (Lua's `nil`) is false.
	///
	/// # Panics
	///
	/// If the value is neither a boolean nor `null`.
	#[must_use]
	fn flag(&self) -> bool;

	/// A list: `null` (Lua's `nil`) is empty.
	///
	/// # Panics
	///
	/// If the value is neither an array nor `null`.
	#[must_use]
	fn list(&self) -> &[Value];
}

impl ValueExt for Value {
	fn number(&self) -> Option<f64> {
		match self {
			Value::Number(n) => n.as_f64(),
			Value::String(s) => match s.as_str() {
				"inf" => Some(f64::INFINITY),
				"-inf" => Some(f64::NEG_INFINITY),
				"nan" => Some(f64::NAN),
				_ => None,
			},
			_ => None,
		}
	}

	fn flag(&self) -> bool {
		match self {
			Value::Bool(b) => *b,
			Value::Null => false,
			other => panic!("expected a flag (a boolean or null), got {other:?}"),
		}
	}

	fn list(&self) -> &[Value] {
		match self {
			Value::Array(a) => a,
			Value::Null => &[],
			other => panic!("expected an array or null, got {other:?}"),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::rng::splitmix64;

	#[test]
	fn numbers_read_back_as_written() {
		let cases = [
			(1.0, "1"),
			(-2500.0, "-2500"),
			(0.5, "0.5"),
			(-0.0, "-0.0"),
			(99_999_999_999_999_984.0, "99999999999999984"),
			(1e17, "1e+17"),
			(1e300, "1e+300"),
			(f64::INFINITY, r#""inf""#),
			(f64::NEG_INFINITY, r#""-inf""#),
			(f64::NAN, r#""nan""#),
		];
		for (x, text) in cases {
			assert_eq!(number(x).to_string(), text);
			let back: Value = serde_json::from_str(text).unwrap();
			assert_eq!(back.number().unwrap().to_bits(), x.to_bits(), "{text}");
		}
		assert_eq!(number(None), Value::Null);
	}

	#[test]
	fn floats_read_back_bit_for_bit() {
		// The harness writes `%.17g`; serde_json reads some of those an ulp
		// off unless its `float_roundtrip` feature is on.
		for i in 0..10_000 {
			let x = f64::from_bits(splitmix64(i));
			if !x.is_finite() {
				continue;
			}
			let text = format!("{x:.16e}");
			let back: Value = serde_json::from_str(&text).unwrap();
			assert_eq!(back.number().unwrap().to_bits(), x.to_bits(), "{text}");
		}
	}

	#[test]
	fn null_is_false_and_empty() {
		assert!(!Value::Null.flag());
		assert!(Value::Bool(true).flag());
		assert_eq!(Value::Null.list(), &[] as &[Value]);
		assert_eq!(Value::from(vec![1, 2]).list().len(), 2);
	}
}
