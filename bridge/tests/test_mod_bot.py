"""The mod's bot (mod/src/bot.lua, core.lua, act.lua, endpoints.lua) on the
player's profile, in the game's own LuaJIT with the installed game's and
Steamodded's functions it works with: the run's options menu and its
toggle, holding R, the run select's Play buttons, the save thread (writing
to an in-memory disk), Steamodded's per-stake usage tables and deck wins.
Around them stands a small model of the game (starting a run, its save
step, its clock). Skipped where the game, its Lovely dump or Steamodded is not
installed."""

from __future__ import annotations

import ctypes
import faulthandler
import os
import zipfile
from pathlib import Path

import pytest
from antezero_bridge.launch import GAME_DIR, MODS_DIR

ROOT = Path(__file__).resolve().parents[2]
LUA51 = Path(os.environ.get("BALATRO_LUA51", GAME_DIR / "lua51.dll"))
PATCHED = MODS_DIR / "lovely" / "dump"  # the game's files Lovely patched, as they run
SMODS_SRC = MODS_DIR / "smods" / "src"


def _game_file(rel: str) -> str:
    if (PATCHED / rel).exists():
        return (PATCHED / rel).read_text(encoding="utf-8")
    with zipfile.ZipFile(GAME_DIR / "Balatro.exe") as z:
        return z.read(rel).decode("utf-8")


def _definition(text: str, header: str, until: str | None = None) -> str:
    """The top-level definition starting at the line ``header`` begins: to
    the first line that is ``end``, or with ``until``, to the line before
    the next one that begins with it."""
    lines = text.splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(header))
    if until is None:
        j = next(j for j in range(i + 1, len(lines)) if lines[j] == "end") + 1
    else:
        j = next(j for j in range(i + 1, len(lines)) if lines[j].startswith(until))
    return "\n".join(lines[i:j])


def _sources() -> str:
    misc = _game_file("functions/misc_functions.lua")
    events = _game_file("functions/common_events.lua")
    buttons = _game_file("functions/button_callbacks.lua")
    ui = _game_file("functions/UI_definitions.lua")
    game = _game_file("game.lua")
    utils = (SMODS_SRC / "utils.lua").read_text(encoding="utf-8")
    overrides = (SMODS_SRC / "overrides.lua").read_text(encoding="utf-8")
    select = (SMODS_SRC / "utils" / "run_select.lua").read_text(encoding="utf-8")
    return "\n".join(
        [
            _game_file("engine/string_packer.lua"),
            *(_definition(misc, h) for h in ("function EMPTY(", "function copy_table(", "function remove_save(")),
            _definition(events, "function unlock_notify("),
            *(_definition(buttons, h) for h in ("G.FUNCS.start_setup_run =", "G.FUNCS.buy_from_shop =", "function G.FUNCS.toggle_button(")),
            *(
                _definition(ui, h)
                for h in ("function create_toggle(", "function create_UIBox_options(", "function create_UIBox_generic_options(", "function UIBox_button(")
            ),
            _definition(_game_file("engine/controller.lua"), "function Controller:key_press("),
            _definition(_game_file("engine/controller.lua"), "function Controller:key_hold_update(", until="function "),
            *(_definition(game, h) for h in ("function Game:save_settings(", "function Game:save_progress(")),
            "save_thread = assert(loadstring(" + _long_string(_game_file("engine/save_manager.lua")) + "))",
            *(_definition(utils, h) for h in ("function SMODS.stake_from_index(", "function convert_usage_entry(", "function convert_save_data(")),
            _definition(overrides, "function set_deck_win("),
            *(
                _definition(select, h)
                for h in (
                    "G.FUNCS.run_select_start_run =",
                    "G.FUNCS.run_select_quick_start =",
                    "function SMODS.RunSelect.Functions.start_run(",
                    "local start_run = Game.start_run",  # Steamodded's Game:start_run
                )
            ),
        ]
    )


def _long_string(s: str) -> str:
    return "[==[" + s + "]==]"


WORLD = r"""
local function noop() end
clock, disk, shown, spawned = 0, {}, {}, {}
local queue = {}
require = noop -- the save thread's
os.execute = function(cmd) spawned[#spawned+1] = cmd return 0 end
love = {
  timer = {getTime = function() return clock end},
  update = function(dt) G:update(dt) end,
  filesystem = {
    getInfo = function(f) if disk[f] then return {} end end,
    read = function(f) return disk[f] end,
    write = function(f, s) disk[f] = s return true end,
    append = function(f, s) disk[f] = (disk[f] or '') .. s return true end,
    remove = function(f) disk[f] = nil return true end,
    createDirectory = noop,
  },
  data = {compress = function(_, _, s) return s end, decompress = function(_, _, s) return s end},
  system = {getOS = function() return 'Windows' end},
  thread = {getChannel = function() return {demand = function() return table.remove(queue, 1) end, push = noop} end},
}
disk['antezero.json'] = 'agent'
JSON = {decode = function(s) assert(s == 'agent', 'no antezero.json') return {command = 'agent'} end}
Game = {}
function Game:delete_run() end
function Game:main_menu() G.STAGE = G.STAGES.MAIN_MENU end
-- The game's clock in a run: real time by dt, game time by dt times the
-- Game Speed, the event queues by real time.
function Game:update(dt)
  G.TIMERS.REAL = G.TIMERS.REAL + dt
  G.TIMERS.TOTAL = G.TIMERS.TOTAL + dt * G.SETTINGS.GAMESPEED
  G.E_MANAGER:update(dt)
end
-- The game's own start of a run, as far as the profile and saves go.
function Game:start_run(args)
  G.SAVED_GAME = nil
  G.STAGE = G.STAGES.RUN
  local back = G.GAME.viewed_back or G.P_CENTERS.b_red
  for _, c in pairs(G.P_CENTERS) do if args.deck_choice and c.name == args.deck_choice.name then back = c end end
  G.GAME = args.savetext and args.savetext.GAME or
    {won = false, seeded = args.seed ~= nil, stake = 1, round = 0, round_resets = {ante = 1},
     pseudorandom = {seed = args.seed or 'RUN'}, selected_back = {effect = {center = back}}}
  G:save_settings()
  unlock_notify()
end
Card = {init = noop, apply_to_run = noop, open = noop, set_ability = noop, remove = noop}
Controller = {queue_L_cursor_press = noop, queue_R_cursor_press = noop, key_press = noop, button_press = noop}
sendInfoMessage, sendErrorMessage, get_current_pool = noop, noop, noop
set_discover_tallies, set_profile_progress, set_challenge_unlock, unlock_achievement = noop, noop, noop, noop
function create_unlock_overlay(key) shown[#shown+1] = key end
function localize(k) return k end
function Event(e) return e end
function Sprite() return {states = {drag = {}, visible = true}} end
function Moveable() return {} end
SMODS = {current_mod = {config = {count = false, speed = 1, search = false}}, Atlas = function() return {key = 'bot'} end,
         RunSelect = {Functions = {clean_up = noop}, Setup = {choices = {}}, Pages = {}}, RunSelectPage = {obj_buffer = {}},
         build_stake_chain = function() return {} end, Keybinds = {}}
AZB = {VERSION = 'test', rpc = {BAD_REQUEST = 'BAD_REQUEST', NOT_ALLOWED = 'NOT_ALLOWED', INVALID_STATE = 'INVALID_STATE', INTERNAL_ERROR = 'INTERNAL_ERROR'}}
function AZB.rpc.register(r) AZB.rpc[r.name] = r.execute end
local white = {key = 'stake_white', order = 1, vanilla_index = 1}
local red = {key = 'b_red', name = 'Red Deck', order = 1, unlocked = true, discovered = true}
local blue = {key = 'b_blue', name = 'Blue Deck', order = 2, unlocked = true, discovered = true}
G = setmetatable({
  FUNCS = {start_run = function(e, args) G:delete_run() G:start_run(args) end},
  ARGS = {}, UIDEF = {run_setup = function() return {} end}, C = setmetatable({UI = setmetatable({}, {__index = function() return {} end})}, {__index = function() return {} end}),
  UIT = {T = 1, B = 2, C = 3, R = 4, O = 5, ROOT = 7}, ROOM = {T = {w = 20, h = 11}},
  E_MANAGER = {add_event = noop, clear_queue = noop, update = noop, queues = {base = {}}}, CONTROLLER = {locks = {}},
  STAGES = {MAIN_MENU = 1, RUN = 2}, STAGE = 1, TIMERS = {TOTAL = 0, REAL = 0}, ASSET_ATLAS = {}, I = {CARD = {}},
  SETTINGS = {profile = 1, tutorial_complete = true, GAMESPEED = 4, current_setup = 'New Run', paused = false},
  PROFILES = {{last_choices = {deck_choice = 'b_red', stake_choice = 'stake_white'}, MEMORY = {stake = 1, deck = 'Red Deck'},
               high_scores = {current_streak = {amt = 4}}, career_stats = {c_wins = 2},
               deck_usage = {b_blue = {count = 1, order = 2, wins = {[1] = 2}, losses = {}}}}},
  P_CENTERS = {b_red = red, b_blue = blue,
               j_joker = {key = 'j_joker', name = 'Joker', order = 1, unlocked = true, discovered = true},
               j_blueprint = {key = 'j_blueprint', name = 'Blueprint', order = 2, unlocked = false, discovered = false}},
  P_BLINDS = {}, P_TAGS = {}, P_SEALS = {}, P_STAKES = {stake_white = white},
  P_CENTER_POOLS = {Back = {red, blue}, Stake = {white}},
  SAVE_MANAGER = {channel = {push = function(_, msg) queue[#queue+1] = STR_UNPACK(STR_PACK(msg)) end}}, -- a copy, as LOVE sends it
  GAME = {won = false},
  STATES = {SELECTING_HAND = 1, SHOP = 5, GAME_OVER = 4, ROUND_EVAL = 8},
}, {__index = Game})
G.P_LOCKED = {G.P_CENTERS.j_blueprint}
G.FUNCS.exit_overlay_menu = function() G.OVERLAY_MENU = nil end
G.FUNCS.go_to_menu = function() G:main_menu() end -- after its screen wipe
"""

MOD = r"""
convert_save_data(G.PROFILES[1], true) -- as Steamodded loads a profile
AZB.json = dofile(MOD .. 'src/json.lua')
for _, f in ipairs({'core', 'act', 'bot', 'endpoints'}) do dofile(MOD .. 'src/' .. f .. '.lua') end
AZB.state = function() return '{}' end
AZB.bot.make_hud = function() end

-- The game's save step (in Game:update), then the save thread's writes.
function flush()
  local f = G.FILE_HANDLER
  if f and f.update_queued then
    if f.progress then G.SAVE_MANAGER.channel:push({type = 'save_progress', save_progress = G.ARGS.save_progress})
    elseif f.settings then
      G.SAVE_MANAGER.channel:push({type = 'save_settings', save_settings = G.ARGS.save_settings,
                                   profile_num = G.SETTINGS.profile, save_profile = G.PROFILES[G.SETTINGS.profile]})
    end
    if f.run then
      G.SAVE_MANAGER.channel:push({type = 'save_run', save_table = G.ARGS.save_run, profile_num = G.SETTINGS.profile})
      G.SAVED_GAME = nil
    end
    G.FILE_HANDLER = {}
  end
  G.SAVE_MANAGER.channel:push({type = 'kill'})
  save_thread()
end

-- misc_functions' save_run, as far as saving goes: it saves G.GAME whole
function save_run()
  if G.F_NO_SAVING == true then return end
  G.ARGS.save_run = {GAME = G.GAME}
  G.FILE_HANDLER = G.FILE_HANDLER or {}
  G.FILE_HANDLER.run, G.FILE_HANDLER.update_queued = true, true
end

function play(deck) -- the run select's Play button with `deck` picked
  SMODS.RunSelect.Setup.choices = {deck_choice = deck, stake_choice = 'stake_white'}
  G.FUNCS.run_select_start_run()
end

MENU = {get_UIE_by_ID = function() end} -- an open menu
-- The bot toggle of the run's options menu, as the game builds it; with
-- `press`, the menu is opened, the toggle pressed and the menu closed.
function bot_toggle(press)
  local function find(n)
    if type(n) ~= 'table' then return end
    if n.config and n.config.button == 'toggle_button' and n.config.ref_table.label == 'Bot plays this run' then return n end
    for _, c in pairs(n.nodes or {}) do
      local r = find(c)
      if r then return r end
    end
  end
  local n = find(create_UIBox_options())
  if n and press then
    G.OVERLAY_MENU = MENU
    G.FUNCS.toggle_button({config = n.config})
    AZB.bot.tick()
    G.FUNCS.exit_overlay_menu()
  end
  return n
end

function poll() -- the bot's own poll
  local reply
  AZB.rpc.az_bot({run = AZB.bot.run.id}, function(r) reply = r end)
  return reply
end

function profile() return G.PROFILES[1] end
function on_disk(file) return disk[file] and STR_UNPACK(disk[file]) end
function deck_wins(entry) return convert_usage_entry(STR_UNPACK(STR_PACK(entry))).wins[1] end -- as saved and loaded
"""


class _Lua:
    def __init__(self) -> None:
        lib = ctypes.CDLL(str(LUA51))
        lib.luaL_newstate.restype = ctypes.c_void_p
        lib.luaL_openlibs.argtypes = [ctypes.c_void_p]
        lib.luaL_loadbuffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
        lib.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.lua_tolstring.restype = ctypes.c_char_p
        lib.lua_settop.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.lua_close.argtypes = [ctypes.c_void_p]
        self.lib, self.state = lib, lib.luaL_newstate()
        lib.luaL_openlibs(self.state)

    def run(self, code: str, name: str = "test") -> str | None:
        lib, L = self.lib, self.state
        b = code.encode()
        # LuaJIT raises Lua errors as Windows exceptions, which faulthandler
        # reports as fatal even when a pcall catches them.
        reporting = faulthandler.is_enabled()
        faulthandler.disable()
        try:
            rc = lib.luaL_loadbuffer(L, b, len(b), name.encode()) or lib.lua_pcall(L, 0, 1, 0)
        finally:
            if reporting:
                faulthandler.enable()
        out = lib.lua_tolstring(L, -1, None)
        lib.lua_settop(L, 0)
        if rc:
            raise AssertionError((out or b"?").decode())
        return out.decode() if out is not None else None

    def close(self) -> None:
        self.lib.lua_close(self.state)


@pytest.fixture
def game():
    if not (LUA51.exists() and PATCHED.exists() and SMODS_SRC.exists() and (GAME_DIR / "Balatro.exe").exists()):
        pytest.skip("needs the game's lua51.dll, its Lovely dump and Steamodded")
    lua = _Lua()
    mod = (ROOT / "mod").as_posix() + "/"
    lua.run(WORLD, "world")
    lua.run(_sources(), "game")
    lua.run(f"MOD = {mod!r}\n" + MOD, "mod")
    yield lua
    lua.close()


def test_the_bot_takes_over_from_the_runs_options_menu_and_hands_it_back(game):
    game.run("""
    G:main_menu()
    assert(not bot_toggle(), 'no toggle at the main menu')
    play('b_red')
    assert(not AZB.bot.active and not AZB.bot.protected, "the player's run")
    bot_toggle(true)
    local first = AZB.bot.run.id
    assert(AZB.bot.active and AZB.bot.protected and #spawned == 1 and spawned[1]:find('agent', 1, true), 'taken over')
    bot_toggle(true)
    AZB.bot.tick()
    assert(not AZB.bot.active and AZB.bot.line:find('handed back') and AZB.bot.line:find('does not count'), AZB.bot.line)
    bot_toggle(true)
    assert(AZB.bot.active and AZB.bot.run.id ~= first, 'taken over again, a new turn for the agent')
    """)


def test_the_run_stops_counting_when_the_bot_takes_over_with_count_off(game):
    game.run("""
    SMODS.current_mod.config.count = true
    play('b_red')
    bot_toggle(true)
    assert(AZB.bot.active and not AZB.bot.protected, 'a turn that counts')
    profile().career_stats.c_wins = 3 -- what it did
    bot_toggle(true)
    SMODS.current_mod.config.count = false
    bot_toggle(true)
    assert(AZB.bot.protected, 'from here it does not count')
    profile().career_stats.c_wins = 4
    SMODS.current_mod.config.count = true -- too late for this run
    bot_toggle(true)
    bot_toggle(true)
    assert(AZB.bot.protected)
    G.FUNCS.go_to_menu()
    assert(not AZB.bot.protected and profile().career_stats.c_wins == 3, 'the part that counted is kept')
    """)


def test_the_toggle_is_off_at_the_start_of_every_run(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    G.FUNCS.start_setup_run({config = {}}) -- New Run from the run's menu, with the vanilla screen's Play
    assert(not AZB.bot.active and not AZB.bot.protected)
    bot_toggle(true)
    G.FUNCS.start_run(nil, {savetext = {GAME = G.GAME}}) -- Continue
    assert(not AZB.bot.active and not AZB.bot.protected)
    """)


def test_a_challenge_run_is_not_taken_over(game):
    game.run("""
    play('b_red')
    G.GAME.challenge = 'c_omelette_1'
    bot_toggle(true)
    assert(not AZB.bot.active and not AZB.bot.protected)
    """)


def test_a_win_in_a_run_that_does_not_count_stays_out_of_the_profile(game):
    game.run("""
    play('b_blue')
    bot_toggle(true)
    set_deck_win()
    G:main_menu()
    local e = profile().deck_usage.b_blue
    assert(e.wins_by_key.stake_white == 2, 'the per-stake view after the run: ' .. tostring(e.wins_by_key.stake_white))
    play('b_blue')
    set_deck_win() -- the player's own win
    flush()
    local saved = on_disk('1/profile.jkr').deck_usage.b_blue
    assert(deck_wins(saved) == 3, 'wins saved: ' .. deck_wins(saved))
    """)


def test_a_run_started_from_the_screen_of_one_that_does_not_count_keeps_its_choices(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    play('b_blue')
    assert(not AZB.bot.protected)
    flush()
    assert(profile().last_choices.deck_choice == 'b_blue', 'in memory')
    assert(on_disk('1/profile.jkr').last_choices.deck_choice == 'b_blue', 'saved')
    """)


def test_the_players_choices_for_the_next_run_are_kept_while_a_run_does_not_count(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    profile().MEMORY.deck = 'Blue Deck' -- the New Run screen, opened from the run's menu, and left with Back
    G:save_settings()
    flush()
    assert(on_disk('1/profile.jkr').MEMORY.deck == 'Blue Deck', 'saved')
    G:main_menu()
    assert(profile().MEMORY.deck == 'Blue Deck', 'kept')
    """)


def test_the_new_run_screen_of_a_run_that_does_not_count_offers_what_the_player_has(game):
    game.run("""
    local blue = G.P_CENTERS.b_blue
    blue.unlocked = false
    play('b_red')
    bot_toggle(true)
    blue.unlocked = true -- the bot's run unlocks Blue Deck, and wins
    profile().career_stats.c_wins = 3
    G.UIDEF.run_setup() -- New Run, from the run's menu or its end screen
    assert(not blue.unlocked and profile().career_stats.c_wins == 2, 'the screen offers what the player has')
    assert(AZB.bot.protected, 'the run still does not count')
    blue.unlocked = true -- Back: the run goes on, and unlocks it again
    profile().career_stats.c_wins = 5
    G:main_menu()
    assert(not blue.unlocked and profile().career_stats.c_wins == 2, 'the set-aside profile is as it was')
    """)


def test_the_players_unlock_notes_wait_for_their_next_run(game):
    game.run("""
    play('b_red')
    disk['1/unlock_notify.jkr'] = 'j_blueprint' .. string.char(10) -- from before the bot took over
    bot_toggle(true)
    unlock_notify() -- the end screen's New Run
    assert(disk['1/unlock_notify.jkr'] and #shown == 0, 'kept while the run does not count')
    play('b_blue')
    assert(not disk['1/unlock_notify.jkr'] and shown[1] == 'j_blueprint', "shown at the player's next run")
    """)


def test_a_bot_win_does_not_spare_the_run_the_player_left_unfinished(game):
    game.run("""
    play('b_red')
    G.SAVED_GAME = {GAME = {won = false}} -- the player's run, saved as it was handed over
    bot_toggle(true)
    G.GAME.won = true -- the bot wins it
    play('b_blue') -- and the player starts another
    flush()
    assert(profile().high_scores.current_streak.amt == 0, 'the unfinished run ends the streak')
    assert(on_disk('1/profile.jkr').high_scores.current_streak.amt == 0, 'saved')
    """)


def test_the_skipped_tutorial_is_never_saved(game):
    game.run("""
    SMODS.current_mod.config.count = true
    G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress = false, {section = 'small_blind', completed_parts = {}}
    play('b_red')
    G.OVERLAY_TUTORIAL = {remove = function() end}
    bot_toggle(true)
    assert(G.SETTINGS.tutorial_complete and not G.OVERLAY_TUTORIAL, 'skipped while the bot plays')
    G:save_settings()
    flush()
    local s = on_disk('settings.jkr')
    assert(s.tutorial_complete == false and s.tutorial_progress.section == 'small_blind', 'a settings save during the run saved the skip')
    G:save_progress()
    flush()
    assert(on_disk('settings.jkr').tutorial_complete == false, 'a progress save during the run saved the skip')
    G:main_menu()
    assert(G.SETTINGS.tutorial_complete == false and G.SETTINGS.tutorial_progress.section == 'small_blind')
    """)


def test_a_run_that_does_not_count_saves_only_the_players_own_state(game):
    game.run("""
    play('b_red')
    save_run()          -- the player's run queues a checkpoint ...
    G.SETTINGS.GAMESPEED = 2
    G:save_settings()   -- ... and a settings change
    bot_toggle(true)    -- and the bot takes over before the game sent them
    flush()
    assert(on_disk('1/save.jkr').GAME.selected_back.effect.center.key == 'b_red', "the player's run checkpoint")
    assert(on_disk('settings.jkr').GAMESPEED == 2, "the player's settings")

    -- the bot's play unlocks and discovers a card, wins, loses its streak,
    -- saves; the player changes a setting meanwhile
    local bp = G.P_CENTERS.j_blueprint
    bp.unlocked, bp.discovered = true, true
    profile().career_stats.c_wins = 3
    profile().high_scores.current_streak.amt = 0
    G:save_progress()
    G.SETTINGS.GAMESPEED = 3
    G:save_settings()
    G.GAME.round = 5
    save_run()
    remove_save()
    flush()
    assert(on_disk('settings.jkr').GAMESPEED == 3, "the player's setting is saved")
    local p = on_disk('1/profile.jkr')
    assert(p.career_stats.c_wins == 2 and p.high_scores.current_streak.amt == 4, 'the run saved its stats')
    assert(not (on_disk('1/meta.jkr') or {unlocked = {}}).unlocked.j_blueprint, 'the run saved its unlock')
    assert(on_disk('1/save.jkr').GAME.round == 0, "the run was saved, or deleted the player's")

    G:main_menu()
    assert(not bp.unlocked and profile().career_stats.c_wins == 2 and profile().high_scores.current_streak.amt == 4)
    assert(G.SETTINGS.GAMESPEED == 3)
    """)


def test_the_developer_tools_save_override_does_not_reach_a_bot_run(game):
    game.run("""
    AZB.config.save_guard = false -- AZB_ALLOW_SAVES=1
    play('b_blue')
    disk['1/save.jkr'] = STR_PACK({GAME = {round = 1}})
    bot_toggle(true)
    G.GAME.round = 3
    save_run()
    remove_save()
    flush()
    assert(on_disk('1/save.jkr').GAME.round == 1)
    """)


def test_holding_r_in_a_run_that_does_not_count_restarts_it_as_the_players(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    bot_toggle(true) -- handed back; the run still does not count
    local c = setmetatable({held_key_times = {r = 0.8}, held_keys = {r = true}, locks = {}}, {__index = Controller})
    c:key_hold_update('r', 0.1)
    assert(not AZB.bot.protected and G.GAME.round == 0, 'a new run')
    flush()
    assert(profile().high_scores.current_streak.amt == 0, 'abandoning the run ends the streak')
    assert(on_disk('1/profile.jkr').high_scores.current_streak.amt == 0, 'saved')
    """)


def test_holding_r_while_the_bot_plays_restarts_the_run(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    G.GAME.round = 2
    local c = setmetatable({held_key_times = {}, held_keys = {}, pressed_keys = {}, locks = {}}, {__index = Controller})
    c:key_press('x')
    assert(not c.pressed_keys.x, 'other keys stay locked while the bot plays')
    c:key_press('r')
    assert(c.pressed_keys.r)
    c.held_key_times.r = 0 -- as the game's key_press_update starts the hold
    for _ = 1, 10 do c:key_hold_update('r', 0.1) end
    assert(not AZB.bot.protected and G.GAME.round == 0, 'a new run')
    """)


def test_a_run_handed_back_says_it_does_not_count(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    AZB.rpc.az_bot({run = AZB.bot.run.id, stop = 'stopped: test'}, function() end)
    AZB.bot.tick()
    assert(AZB.bot.line:find('does not count'), AZB.bot.line)
    """)


def test_the_bot_hears_of_a_menu_opened_since_its_last_poll(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    AZB.bot.tick()
    assert(poll().interrupted, 'the menu it was taken over in')
    assert(not poll().interrupted, 'reported once')
    G.OVERLAY_MENU = MENU
    AZB.bot.tick()
    local r = poll()
    assert(r.paused and r.interrupted)
    G.OVERLAY_MENU = nil
    AZB.bot.tick()
    assert(not poll().paused)
    G.OVERLAY_MENU = MENU -- opened and closed between two polls
    AZB.bot.tick()
    G.OVERLAY_MENU = nil
    AZB.bot.tick()
    r = poll()
    assert(not r.paused and r.interrupted, 'closed since')
    G.STATE, G.OVERLAY_MENU = G.STATES.GAME_OVER, MENU -- the game over screen
    AZB.bot.tick()
    r = poll()
    assert(not r.paused and not r.interrupted, 'the end of the run is no menu')
    """)


def test_an_action_of_a_turn_that_is_over_is_refused_before_it_touches_the_hand(game):
    replies = game.run("""
    play('b_red')
    bot_toggle(true)
    local cards = {{}, {highlighted = true}} -- what the player selected
    G.hand = {cards = cards,
              unhighlight_all = function(self) for _, c in ipairs(self.cards) do c.highlighted = false end end,
              add_to_highlighted = function(self, c) c.highlighted = true end}
    G.STATE, G.STATE_COMPLETE, G.buttons = G.STATES.SELECTING_HAND, true, {}
    poll()
    local replies, run = {}, AZB.bot.run.id
    local function act() AZB.rpc.az_act({action = {type = 'play', cards = {1}}, run = run}, function(r) replies[#replies+1] = r end) end
    G.OVERLAY_MENU = MENU -- opened after the bot's poll
    AZB.bot.tick()
    act()
    G.OVERLAY_MENU = nil
    poll()
    G.CONTROLLER.locks.test = true -- the game is busy when the next action comes ...
    act()
    bot_toggle(true) -- ... and the player takes the run back meanwhile
    G.CONTROLLER.locks.test = nil
    for _ = 1, AZB.config.quiet_frames + 2 do AZB.tick() end
    assert(not cards[1].highlighted and cards[2].highlighted, 'the hand as the player left it')
    return AZB.json.encode(replies)
    """)
    assert '"name":"INVALID_STATE"' in replies and '"name":"NOT_ALLOWED"' in replies


def test_the_bot_waits_on_the_win_screen_and_plays_on_in_endless_mode(game):
    game.run("""
    play('b_red')
    bot_toggle(true)
    G.STATE, G.STATE_COMPLETE, G.shop, G.shop_jokers = G.STATES.SHOP, true, {}, {cards = {}}
    poll()
    local reply
    AZB.rpc.az_state({}, function(r) reply = r end) -- the bot's wait for the game after its last hand
    G.GAME.won = true
    G.OVERLAY_MENU = {get_UIE_by_ID = function(_, id) return id == 'you_win_UI' and {} or nil end}
    for _ = 1, 10 do AZB.tick() end
    assert(not reply, 'answered under the win screen')
    local r = poll()
    assert(r.paused and r.interrupted and AZB.bot.line:find('Endless Mode'), AZB.bot.line)
    G.FUNCS.exit_overlay_menu() -- Endless Mode
    G.E_MANAGER.queues.base = {{blocking = true}} -- the round's end, still running
    for _ = 1, AZB.config.quiet_frames + 2 do AZB.tick() end
    assert(not reply, 'answered before the round had ended')
    G.E_MANAGER.queues.base = {}
    for _ = 1, AZB.config.quiet_frames + 2 do AZB.tick() end
    assert(reply and reply.settled, 'the game after the win screen')
    assert(AZB.bot.active and not poll().paused)
    G.OVERLAY_MENU = MENU -- a menu in Endless Mode
    AZB.bot.tick()
    assert(poll().paused and AZB.bot.line:find('paused'), AZB.bot.line)
    """)


def test_raising_the_game_speed_never_slows_the_bot(game):
    game.run(_game_file("engine/object.lua") + "\n" + _game_file("engine/event.lua"), "events")
    game.run("""
    G.E_MANAGER = EventManager()
    play('b_red')
    bot_toggle(true)
    SMODS.current_mod.config.speed = 16
    -- frames until the game is through a hand's events: steps one after another, then a wait
    local function frames(gamespeed)
      G.SETTINGS.GAMESPEED = gamespeed
      local done, n = false, 0
      for _ = 1, 20 do G.E_MANAGER:add_event(Event({})) end
      G.E_MANAGER:add_event(Event({trigger = 'after', delay = 1, func = function() done = true return true end}))
      repeat love.update(1 / 60); n = n + 1 until done or n == 100
      return n
    end
    local slow, fast = frames(1), frames(4)
    assert(fast <= slow, fast .. ' frames at Game Speed 4, ' .. slow .. ' at 1')
    """)


def test_the_cards_of_a_continued_run_have_numbers_of_their_own_while_the_bot_plays(game):
    game.run("""
    play('b_red')
    -- continued after a restart: the game's counter started over below the run's numbers
    local seven, forty = {sort_id = 7}, {sort_id = 40}
    G.I.CARD = {seven, forty, {sort_id = 12}, {sort_id = 40}, {sort_id = 7}}
    G.sort_id = 12
    bot_toggle(true)
    local seen = {}
    for _, c in ipairs(G.I.CARD) do
      assert(not seen[c.sort_id], 'two cards numbered ' .. c.sort_id)
      assert(c.sort_id <= G.sort_id, 'a new card would be numbered ' .. G.sort_id + 1)
      seen[c.sort_id] = true
    end
    assert(seven.sort_id == 7 and forty.sort_id == 40, 'the older cards keep their numbers')
    """)


def test_a_buy_the_game_refuses_is_refused(game):
    reply = game.run("""
    G.STATE, G.STATE_COMPLETE = G.STATES.SHOP, true
    G.shop, G.E_MANAGER, G.CONTROLLER, G.TIMERS = {}, {queues = {base = {}}}, {locks = {}}, {TOTAL = 0}
    G.shop_jokers = {cards = {{is = function() return true end}}}
    G.FUNCS.check_for_buy_space = function() return false end -- a full joker row
    AZB.quiet = AZB.config.quiet_frames
    local reply
    AZB.rpc.az_act({action = {type = 'buy', index = 1}}, function(r) reply = r end)
    for _ = 1, AZB.config.quiet_frames + 1 do AZB.tick() end
    return AZB.json.encode(reply)
    """)
    assert '"name":"BAD_REQUEST"' in reply and "no room" in reply


def test_a_run_holds_nothing_of_the_mods(game):
    game.run("""
    play('b_red')
    G.FUNCS.start_run(nil, {savetext = {GAME = {azb_used_jokers = {j_joker = true}, azb_pack_size = 3}}}) -- saved by an earlier version
    local card = {ability = {set = 'Booster', name = 'Arcana Pack', extra = 3}}
    Card.open(card)
    Card.set_ability(card)
    Card.remove(card)
    assert(AZB.pack_size == 3, 'the pack size for the dump')
    for k in pairs(G.GAME) do assert(not tostring(k):find('^azb_'), k) end
    """)


def test_several_pending_vouchers_are_written_for_the_importer_to_refuse(game):
    game.run("""
    dofile(MOD .. 'src/state.lua')
    assert(AZB.vanilla_voucher({'v_grabber', spawn = {v_grabber = true}}) == 'v_grabber')
    assert(AZB.vanilla_voucher({'v_grabber', 'v_wasteful', spawn = {v_grabber = false, v_wasteful = true}}) == 'v_wasteful')
    local several = AZB.vanilla_voucher({'v_grabber', 'v_wasteful', spawn = {v_grabber = true, v_wasteful = true}})
    assert(type(several) == 'table' and #several == 2, 'a second voucher of another mod')
    """)
