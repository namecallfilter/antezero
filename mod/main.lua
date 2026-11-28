-- "Settle" mirrors H.settle in sim/oracle/lua/harness.lua: wait until there
-- are no blocking events in the base queue, no controller locks, the state
-- is a stable decision point and STOP_USE == 0, for 3 consecutive frames.
-- Actions follow the harness timing protocol (select -> settle -> rebase
-- G.TIMERS.TOTAL -> press -> settle), see src/act.lua.
--
-- State dumps are byte-compatible with the harness's H.state JSON. Where
-- Steamodded represents vanilla state differently the dump reports the
-- vanilla value. In a session the developer tools started, src/core.lua
-- carries a few result-neutral or vanilla-restoring shims
-- (AZB_SMODS_COMPAT=0 disables them), blocks all profile/settings/run save
-- writes (AZB_ALLOW_SAVES=1 re-enables) and unlocks everything in memory
-- like the harness; a bot run saves only the player's own state (see
-- src/bot.lua); otherwise the mod leaves the game alone. The az_eval method
-- (arbitrary Lua) is refused unless AZB_ALLOW_EVAL=1.

AZB = AZB or {}
AZB.VERSION = "0.3.0"

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
