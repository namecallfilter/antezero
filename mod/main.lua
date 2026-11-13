-- AnteZero: the in-game side of the antezero agent.
--
-- For players: a bot toggle on the New Run screen's Play button (src/bot.lua).
--
-- For the agent and the developer tools, JSON-RPC methods (src/server.lua;
-- on BalatroBot's dispatcher when the tools start the game with it):
--   az_info     version, config, settle diagnostics, blocked save writes
--   az_state    the oracle-shaped state dump (optionally after settling)
--   az_start    start a seeded run exactly like the oracle's H.start
--   az_act      perform one action exactly like the oracle's H.act, then settle
--   az_inject   scenario injection exactly like the oracle's H.inject, then settle
--   az_legal    legal sim action codes at the current decision point
--   az_bot      the bot run to play, and its status line
--   az_eval     evaluate a Lua expression (only with AZB_ALLOW_EVAL=1)
--
-- "Settle" mirrors H.settle in sim/oracle/lua/harness.lua: wait until there
-- are no blocking events in the base queue, no controller locks, the state
-- is a stable decision point and STOP_USE == 0, for 3 consecutive frames.
-- Actions follow the harness timing protocol (select -> settle -> rebase
-- G.TIMERS.TOTAL -> press -> settle), see src/act.lua.
--
-- State dumps are byte-compatible with the harness's H.state JSON. Where
-- Steamodded represents vanilla state differently the dump reports the
-- vanilla value (pack states, current_round.voucher, used_jokers, pack_size,
-- SMODS-only RNG streams moved to `az`), and src/core.lua carries a few
-- result-neutral or vanilla-restoring shims (AZB_SMODS_COMPAT=0 disables):
-- legendary rarity roll, voucher display-card sort_ids, played_this_ante.
-- While the agent or the tools drive the game (a bot run, or a session the
-- tools started) it also blocks all profile/settings/run save writes
-- (AZB_ALLOW_SAVES=1 re-enables) and unlocks everything in memory like the
-- harness; otherwise it leaves the game alone.

AZB = AZB or {}
AZB.VERSION = "0.2.0"

local function load(path)
  return assert(SMODS.load_file(path))()
end

AZB.json = load("src/json.lua")
load("src/core.lua")
load("src/server.lua")
load("src/state.lua")
load("src/act.lua")
load("src/legal.lua")
load("src/bot.lua")
load("src/endpoints.lua")

sendInfoMessage("AnteZero " .. AZB.VERSION .. " loaded", "AnteZero")
