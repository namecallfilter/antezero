//! The oracle: a live `LuaJIT` state, optionally running the Balatro harness,
//! and where its DLL and scripts are found.

use std::path::{Path, PathBuf};

#[must_use]
pub fn default_dll() -> PathBuf {
	if let Ok(p) = std::env::var("BALATRO_LUA51") {
		return PathBuf::from(p);
	}
	PathBuf::from(r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\lua51.dll")
}

#[must_use]
pub fn default_src() -> PathBuf {
	if let Ok(p) = std::env::var("BALATRO_SRC") {
		return PathBuf::from(p);
	}
	Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/balatro")
}

fn lua_dir() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("lua")
}

/// Whether the bare `LuaJIT` state can be created on this machine.
#[must_use]
pub fn dll_available() -> bool {
	cfg!(windows) && default_dll().exists()
}

/// Whether the full game harness can run on this machine.
#[must_use]
pub fn available() -> bool {
	dll_available() && default_src().join("game.lua").exists()
}

/// `s` as a Lua 5.1 string literal.
#[must_use]
pub fn lua_str(s: &str) -> String {
	use std::fmt::Write;

	let mut out = String::with_capacity(s.len() + 2);
	out.push('"');
	for &b in s.as_bytes() {
		match b {
			b'"' => out.push_str("\\\""),
			b'\\' => out.push_str("\\\\"),
			0x20..=0x7e => out.push(b as char),
			// Control and non-ASCII bytes as decimal escapes (byte exact).
			_ => write!(out, "\\{b:03}").expect("writing to a String cannot fail"),
		}
	}
	out.push('"');
	out
}

/// A live `LuaJIT` state, optionally running the Balatro harness.
pub struct Oracle {
	#[cfg(windows)]
	lua: crate::lua::State,
}

impl Oracle {
	/// Loads `lua51.dll` and creates a bare `LuaJIT` state (no harness).
	///
	/// # Errors
	///
	/// Returns why the DLL could not be loaded or the state not created.
	pub fn raw() -> Result<Self, String> {
		#[cfg(windows)]
		{
			Ok(Oracle {
				lua: crate::lua::State::new(&default_dll())?,
			})
		}
		#[cfg(not(windows))]
		{
			Err("the oracle needs Windows (it runs the game's lua51.dll)".into())
		}
	}

	/// Loads the full headless game harness.
	///
	/// # Errors
	///
	/// Returns why the DLL or the harness could not be loaded.
	pub fn new() -> Result<Self, String> {
		let o = Self::raw()?;
		let dir = |p: PathBuf| lua_str(&p.display().to_string().replace('\\', "/"));
		let boot = format!(
			"HARNESS_SRC = {}\nHARNESS_DIR = {}\npackage.path = HARNESS_DIR..'/?.lua;'..HARNESS_SRC..'/?.lua;'..package.path\nrequire('harness')",
			dir(default_src()),
			dir(lua_dir())
		);
		o.exec(&boot)?;
		Ok(o)
	}

	/// Runs a chunk of Lua code.
	///
	/// # Errors
	///
	/// Returns the Lua error message.
	pub fn exec(&self, code: &str) -> Result<(), String> {
		#[cfg(windows)]
		{
			self.lua.exec(code)
		}
		#[cfg(not(windows))]
		{
			let _ = code;
			unreachable!("no oracle off Windows")
		}
	}

	/// Evaluates a Lua chunk that returns a string (or a number; `nil`
	/// gives an empty string).
	///
	/// # Errors
	///
	/// Returns the Lua error message, or what the chunk returned instead.
	pub fn eval(&self, code: &str) -> Result<String, String> {
		#[cfg(windows)]
		{
			self.lua.eval(code)
		}
		#[cfg(not(windows))]
		{
			let _ = code;
			unreachable!("no oracle off Windows")
		}
	}

	/// Calls the global function `name(arg)` and returns its string result
	/// (as [`eval`](Self::eval)).
	///
	/// # Errors
	///
	/// Returns the Lua error message, or what the function returned instead.
	pub fn call(&self, name: &str, arg: &str) -> Result<String, String> {
		#[cfg(windows)]
		{
			self.lua.call(name, arg)
		}
		#[cfg(not(windows))]
		{
			let _ = (name, arg);
			unreachable!("no oracle off Windows")
		}
	}
}
