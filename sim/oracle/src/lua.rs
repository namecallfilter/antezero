//! The Lua 5.1 C API of the game's `lua51.dll`, loaded at runtime.

use std::ffi::{CString, c_char, c_int, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::Path;

type LuaState = c_void;

const LUA_GLOBALSINDEX: c_int = -10002;
const LUA_TNIL: c_int = 0;
const LUA_TNUMBER: c_int = 3;
const LUA_TSTRING: c_int = 4;
/// Name of the chunks passed to `luaL_loadbuffer` (shown in errors).
const CHUNK: &std::ffi::CStr = c"=oracle";

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> c_int;
    fn GetLastError() -> u32;
}
const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;

struct Api {
    new_state: unsafe extern "C" fn() -> *mut LuaState,
    open_libs: unsafe extern "C" fn(*mut LuaState),
    load_buffer: unsafe extern "C" fn(*mut LuaState, *const c_char, usize, *const c_char) -> c_int,
    pcall: unsafe extern "C" fn(*mut LuaState, c_int, c_int, c_int) -> c_int,
    to_lstring: unsafe extern "C" fn(*mut LuaState, c_int, *mut usize) -> *const c_char,
    lua_type: unsafe extern "C" fn(*mut LuaState, c_int) -> c_int,
    set_top: unsafe extern "C" fn(*mut LuaState, c_int),
    get_field: unsafe extern "C" fn(*mut LuaState, c_int, *const c_char),
    push_lstring: unsafe extern "C" fn(*mut LuaState, *const c_char, usize),
    close: unsafe extern "C" fn(*mut LuaState),
}

/// The loaded DLL (freed on drop).
struct Module(*mut c_void);

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { FreeLibrary(self.0) };
    }
}

/// A Lua state of the game's LuaJIT. Fields drop in order: the state is
/// closed before the DLL is released.
pub struct State {
    api: Api,
    l: *mut LuaState,
    _module: Module,
}

// The state is only ever touched from the owning thread.
unsafe impl Send for State {}

impl Drop for State {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.l) };
    }
}

impl State {
    pub fn new(dll: &Path) -> Result<State, String> {
        let wide: Vec<u16> = dll.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let handle = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
        if handle.is_null() {
            return Err(format!("could not load {} (error {})", dll.display(), unsafe { GetLastError() }));
        }
        let module = Module(handle);
        macro_rules! sym {
            ($name:literal: $ty:ty) => {{
                let p = unsafe { GetProcAddress(module.0, concat!($name, "\0").as_ptr() as *const c_char) };
                if p.is_null() {
                    return Err(format!("missing symbol {}", $name));
                }
                // SAFETY: the symbol is the Lua 5.1 C API function of this signature.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }
        let api = Api {
            new_state: sym!("luaL_newstate": unsafe extern "C" fn() -> *mut LuaState),
            open_libs: sym!("luaL_openlibs": unsafe extern "C" fn(*mut LuaState)),
            load_buffer: sym!("luaL_loadbuffer": unsafe extern "C" fn(*mut LuaState, *const c_char, usize, *const c_char) -> c_int),
            pcall: sym!("lua_pcall": unsafe extern "C" fn(*mut LuaState, c_int, c_int, c_int) -> c_int),
            to_lstring: sym!("lua_tolstring": unsafe extern "C" fn(*mut LuaState, c_int, *mut usize) -> *const c_char),
            lua_type: sym!("lua_type": unsafe extern "C" fn(*mut LuaState, c_int) -> c_int),
            set_top: sym!("lua_settop": unsafe extern "C" fn(*mut LuaState, c_int)),
            get_field: sym!("lua_getfield": unsafe extern "C" fn(*mut LuaState, c_int, *const c_char)),
            push_lstring: sym!("lua_pushlstring": unsafe extern "C" fn(*mut LuaState, *const c_char, usize)),
            close: sym!("lua_close": unsafe extern "C" fn(*mut LuaState)),
        };
        let l = unsafe { (api.new_state)() };
        if l.is_null() {
            return Err("luaL_newstate failed".into());
        }
        unsafe { (api.open_libs)(l) };
        Ok(State { api, l, _module: module })
    }

    pub fn exec(&self, code: &str) -> Result<(), String> {
        unsafe {
            if (self.api.load_buffer)(self.l, code.as_ptr() as *const c_char, code.len(), CHUNK.as_ptr()) != 0 {
                return Err(self.pop_error());
            }
            if (self.api.pcall)(self.l, 0, 0, 0) != 0 {
                return Err(self.pop_error());
            }
        }
        Ok(())
    }

    pub fn eval(&self, code: &str) -> Result<String, String> {
        unsafe {
            if (self.api.load_buffer)(self.l, code.as_ptr() as *const c_char, code.len(), CHUNK.as_ptr()) != 0 {
                return Err(self.pop_error());
            }
            if (self.api.pcall)(self.l, 0, 1, 0) != 0 {
                return Err(self.pop_error());
            }
            self.pop_result()
        }
    }

    pub fn call(&self, name: &str, arg: &str) -> Result<String, String> {
        let cname = CString::new(name).map_err(|_| format!("bad function name {name:?}"))?;
        unsafe {
            (self.api.get_field)(self.l, LUA_GLOBALSINDEX, cname.as_ptr());
            (self.api.push_lstring)(self.l, arg.as_ptr() as *const c_char, arg.len());
            if (self.api.pcall)(self.l, 1, 1, 0) != 0 {
                return Err(self.pop_error());
            }
            self.pop_result()
        }
    }

    /// Pops the single result of a call: a string, a number (as text) or
    /// `nil` (an empty string).
    unsafe fn pop_result(&self) -> Result<String, String> {
        let t = unsafe { (self.api.lua_type)(self.l, -1) };
        let r = match t {
            LUA_TNIL => Ok(String::new()),
            LUA_TNUMBER | LUA_TSTRING => Ok(unsafe { self.top_string() }),
            _ => Err(format!("expected a string result, got Lua type {t}")),
        };
        unsafe { (self.api.set_top)(self.l, 0) };
        r
    }

    unsafe fn top_string(&self) -> String {
        let mut len = 0usize;
        let p = unsafe { (self.api.to_lstring)(self.l, -1, &mut len) };
        if p.is_null() {
            return String::new();
        }
        let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        String::from_utf8_lossy(bytes).into_owned()
    }

    fn pop_error(&self) -> String {
        unsafe {
            let s = self.top_string();
            (self.api.set_top)(self.l, 0);
            s
        }
    }
}
