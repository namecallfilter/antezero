//! Ground-truth oracle: the real Balatro Lua code, executed headlessly inside
//! the exact LuaJIT build that ships with the game (`lua51.dll`).
//!
//! The LÖVE engine is replaced by stubs (`lua/harness.lua`) and the game's
//! own scripts are loaded from `reference/balatro`. Commands and results are
//! exchanged as strings so that the oracle can drive full runs and dump
//! comparable state snapshots.
//!
//! The harness instruments the game without changing its logic: it traces
//! `pseudoseed` calls, card draws and removals, rebases `G.TIMERS.TOTAL` to 0
//! at the start of every action, replaces `draw_card` with a copy whose only
//! change is marking its event (for draw tracing), disables GC time-slicing,
//! and models `Card.unique_val` (which the game derives from a UI node
//! counter) exactly as the simulator does, so this oracle cannot check that
//! tie-break on its own.
//!
//! Windows-only (the DLL is a Windows binary). Locations can be overridden
//! with `BALATRO_LUA51` (path to `lua51.dll`) and `BALATRO_SRC` (directory
//! with the extracted game scripts).

use std::path::{Path, PathBuf};

pub mod canon;
pub mod diff;
pub mod json;

#[cfg(windows)]
mod lua;

pub fn default_dll() -> PathBuf {
    if let Ok(p) = std::env::var("BALATRO_LUA51") {
        return PathBuf::from(p);
    }
    PathBuf::from(r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\lua51.dll")
}

pub fn default_src() -> PathBuf {
    if let Ok(p) = std::env::var("BALATRO_SRC") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/balatro")
}

fn lua_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("lua")
}

/// Whether the bare LuaJIT state can be created on this machine.
pub fn dll_available() -> bool {
    cfg!(windows) && default_dll().exists()
}

/// Whether the full game harness can run on this machine.
pub fn available() -> bool {
    dll_available() && default_src().join("game.lua").exists()
}

/// `s` as a Lua 5.1 string literal.
pub fn lua_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for &b in s.as_bytes() {
        match b {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            0x20..=0x7e => out.push(b as char),
            // Control and non-ASCII bytes as decimal escapes (byte exact).
            _ => out.push_str(&format!("\\{b:03}")),
        }
    }
    out.push('"');
    out
}

/// A live LuaJIT state, optionally running the Balatro harness.
pub struct Oracle {
    #[cfg(windows)]
    lua: lua::State,
}

impl Oracle {
    /// Loads `lua51.dll` and creates a bare LuaJIT state (no harness).
    pub fn raw() -> Result<Self, String> {
        #[cfg(windows)]
        {
            Ok(Oracle { lua: lua::State::new(&default_dll())? })
        }
        #[cfg(not(windows))]
        {
            Err("the oracle needs Windows (it runs the game's lua51.dll)".into())
        }
    }

    /// Loads the full headless game harness.
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
