//! A small strict JSON parser for the harness' state dumps (numbers the
//! harness cannot write as JSON are strings: `"inf"`, `"-inf"`, `"nan"`).

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
	Null,
	Bool(bool),
	Num(f64),
	Str(String),
	Arr(Vec<Json>),
	Obj(BTreeMap<String, Json>),
}

impl Json {
	/// Member `k` of an object (`Null` if absent or not an object).
	#[must_use]
	pub fn get(&self, k: &str) -> &Json {
		static NULL: Json = Json::Null;
		match self {
			Json::Obj(m) => m.get(k).unwrap_or(&NULL),
			_ => &NULL,
		}
	}

	#[must_use]
	pub fn num(&self) -> Option<f64> {
		match self {
			Json::Num(n) => Some(*n),
			Json::Str(s) if s == "inf" => Some(f64::INFINITY),
			Json::Str(s) if s == "-inf" => Some(f64::NEG_INFINITY),
			Json::Str(s) if s == "nan" => Some(f64::NAN),
			_ => None,
		}
	}

	#[must_use]
	pub fn str(&self) -> Option<&str> {
		if let Json::Str(s) = self {
			Some(s)
		} else {
			None
		}
	}

	#[must_use]
	pub fn bool(&self) -> bool {
		matches!(self, Json::Bool(true))
	}

	#[must_use]
	pub fn arr(&self) -> &[Json] {
		if let Json::Arr(a) = self { a } else { &[] }
	}

	#[must_use]
	pub fn obj(&self) -> Option<&BTreeMap<String, Json>> {
		if let Json::Obj(m) = self {
			Some(m)
		} else {
			None
		}
	}
}

/// Parses one JSON document (nothing but whitespace may follow it).
///
/// # Errors
///
/// Returns what is malformed and at which byte.
pub fn parse(s: &str) -> Result<Json, String> {
	let mut p = Parser {
		b: s.as_bytes(),
		i: 0,
	};
	let v = p.value()?;
	p.ws();
	if p.i != p.b.len() {
		return Err(p.err("trailing characters"));
	}
	Ok(v)
}

struct Parser<'a> {
	b: &'a [u8],
	i: usize,
}

impl Parser<'_> {
	fn err(&self, what: &str) -> String {
		format!("{what} at byte {}", self.i)
	}

	fn ws(&mut self) {
		while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
			self.i += 1;
		}
	}

	fn peek(&self) -> Result<u8, String> {
		self.b
			.get(self.i)
			.copied()
			.ok_or_else(|| self.err("unexpected end"))
	}

	fn expect(&mut self, c: u8) -> Result<(), String> {
		if self.peek()? != c {
			return Err(self.err(&format!("expected {:?}", c as char)));
		}
		self.i += 1;
		Ok(())
	}

	fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
		if !self.b[self.i..].starts_with(word.as_bytes()) {
			return Err(self.err("bad literal"));
		}
		self.i += word.len();
		Ok(v)
	}

	fn value(&mut self) -> Result<Json, String> {
		self.ws();
		match self.peek()? {
			b'n' => self.literal("null", Json::Null),
			b't' => self.literal("true", Json::Bool(true)),
			b'f' => self.literal("false", Json::Bool(false)),
			b'"' => Ok(Json::Str(self.string()?)),
			b'[' => {
				self.i += 1;
				let mut v = Vec::new();
				self.ws();
				if self.peek()? == b']' {
					self.i += 1;
					return Ok(Json::Arr(v));
				}
				loop {
					v.push(self.value()?);
					self.ws();
					match self.peek()? {
						b',' => self.i += 1,
						b']' => {
							self.i += 1;
							return Ok(Json::Arr(v));
						}
						_ => return Err(self.err("expected ',' or ']'")),
					}
				}
			}
			b'{' => {
				self.i += 1;
				let mut m = BTreeMap::new();
				self.ws();
				if self.peek()? == b'}' {
					self.i += 1;
					return Ok(Json::Obj(m));
				}
				loop {
					self.ws();
					let k = self.string()?;
					self.ws();
					self.expect(b':')?;
					let v = self.value()?;
					m.insert(k, v);
					self.ws();
					match self.peek()? {
						b',' => self.i += 1,
						b'}' => {
							self.i += 1;
							return Ok(Json::Obj(m));
						}
						_ => return Err(self.err("expected ',' or '}'")),
					}
				}
			}
			b'-' | b'0'..=b'9' => {
				let st = self.i;
				while self.i < self.b.len()
					&& matches!(
						self.b[self.i],
						b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
					) {
					self.i += 1;
				}
				let t = std::str::from_utf8(&self.b[st..self.i])
					.expect("the loop accepted ASCII bytes only");
				t.parse::<f64>()
					.map(Json::Num)
					.map_err(|e| format!("bad number {t:?} at byte {st}: {e}"))
			}
			_ => Err(self.err("unexpected character")),
		}
	}

	fn hex4(&mut self) -> Result<u32, String> {
		let h = self
			.b
			.get(self.i..self.i + 4)
			.and_then(|h| std::str::from_utf8(h).ok())
			.ok_or_else(|| self.err("short \\u escape"))?;
		let v = u32::from_str_radix(h, 16).map_err(|_| self.err("bad \\u escape"))?;
		self.i += 4;
		Ok(v)
	}

	fn string(&mut self) -> Result<String, String> {
		self.expect(b'"')?;
		let mut out = Vec::new();
		loop {
			let c = self.peek().map_err(|_| self.err("unterminated string"))?;
			self.i += 1;
			match c {
				b'"' => {
					return String::from_utf8(out).map_err(|_| self.err("invalid UTF-8 in string"));
				}
				b'\\' => {
					let e = self.peek()?;
					self.i += 1;
					let ch = match e {
						b'"' => '"',
						b'\\' => '\\',
						b'/' => '/',
						b'b' => '\u{8}',
						b'f' => '\u{c}',
						b'n' => '\n',
						b'r' => '\r',
						b't' => '\t',
						b'u' => {
							let hi = self.hex4()?;
							let code = if (0xd800..0xdc00).contains(&hi) {
								if !self.b[self.i..].starts_with(b"\\u") {
									return Err(self.err("unpaired surrogate"));
								}
								self.i += 2;
								let lo = self.hex4()?;
								if !(0xdc00..0xe000).contains(&lo) {
									return Err(self.err("unpaired surrogate"));
								}
								0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
							} else {
								hi
							};
							char::from_u32(code).ok_or_else(|| self.err("bad \\u escape"))?
						}
						_ => return Err(self.err("bad escape")),
					};
					out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes());
				}
				0..0x20 => return Err(self.err("control character in string")),
				_ => out.push(c),
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_harness_output() {
		let j = parse(r#" {"a":[1,-2.5e3,"inf"],"b":{"c":true,"d":null},"e":"x\"\\\n\u00e9\u0001\ud83d\ude00","f":[]} "#).unwrap();
		assert_eq!(j.get("a").arr().len(), 3);
		assert_eq!(j.get("a").arr()[1].num(), Some(-2500.0));
		assert_eq!(j.get("a").arr()[2].num(), Some(f64::INFINITY));
		assert!(j.get("b").get("c").bool());
		assert_eq!(j.get("b").get("d"), &Json::Null);
		assert_eq!(j.get("e").str(), Some("x\"\\\né\u{1}😀"));
		assert_eq!(parse("\"\\t\\r\\/é\"").unwrap(), Json::Str("\t\r/é".into()));
		assert_eq!(parse("{}").unwrap(), Json::Obj(BTreeMap::new()));
	}

	#[test]
	fn rejects_malformed_input() {
		for bad in [
			"",
			"n",
			"nul",
			"true garbage",
			"[1 2]",
			"[",
			"[1,",
			"{\"a\" 1}",
			"{\"a\":1,}",
			"[1,]",
			"\"abc",
			"\"\\x\"",
			"\"\\u12\"",
			"\"\\ud800\"",
			"\"a\nb\"",
			"01x",
			"{1:2}",
			"-",
		] {
			assert!(parse(bad).is_err(), "{bad:?} parsed");
		}
	}
}
