//! Ground-truth oracle: the real Balatro Lua code, executed headlessly inside
//! the exact `LuaJIT` build that ships with the game (`lua51.dll`).
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

pub mod canon;
pub mod diff;
pub mod json;
#[cfg(windows)]
mod lua;
mod oracle;

pub use oracle::{Oracle, available, default_dll, default_src, dll_available, lua_str};
