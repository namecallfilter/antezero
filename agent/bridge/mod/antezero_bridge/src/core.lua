-- Config, profile/save guard, oracle-parity hooks and the settle monitor.

local function env_flag(name, default)
  local v = os.getenv(name)
  if v == nil or v == "" then return default end
  v = v:lower()
  return v == "1" or v == "true" or v == "yes"
end

AZB.config = {
  -- Block every write of settings/profile/progress/run saves (the bot
  -- otherwise overwrites the player's save.jkr, settings and unlocks).
  save_guard = not env_flag("AZB_ALLOW_SAVES", false),
  -- Unlock + discover everything in memory, like the oracle harness
  -- (pools depend on `unlocked`, tag pools on `discovered`).
  unlock_all = env_flag("AZB_UNLOCK_ALL", true),
  -- unique_val = 1 - sort_id*50/1603301 like the harness (vanilla derives it
  -- from the UI node counter; it breaks sort ties between identical cards).
  stable_unique_val = env_flag("AZB_STABLE_UNIQUE_VAL", true),
  -- Rebase G.TIMERS.TOTAL to 0 at the start of every action (H.rebase).
  rebase = env_flag("AZB_REBASE", true),
  -- Shims for Steamodded deviations from vanilla that are result-neutral
  -- (RNG/sort_id bookkeeping) or restore vanilla behaviour (played_this_ante);
  -- see the `smods_compat` blocks below.
  smods_compat = env_flag("AZB_SMODS_COMPAT", true),
  allow_eval = env_flag("AZB_ALLOW_EVAL", false),
  quiet_frames = 3,
}

----------------------------------------------------------------------------
-- Save guard
----------------------------------------------------------------------------
AZB.blocked_saves = {}

function AZB.install_save_guard()
  if not AZB.config.save_guard or AZB._guard_installed then return end
  AZB._guard_installed = true
  G.F_NO_SAVING = true -- save_run() returns early
  local real = G.SAVE_MANAGER and G.SAVE_MANAGER.channel
  if real and not AZB._real_save_channel then
    AZB._real_save_channel = real
    -- Proxy: drop every save_* request, forward everything else ('kill').
    G.SAVE_MANAGER.channel = setmetatable({
      push = function(_, msg)
        local t = type(msg) == "table" and tostring(msg.type) or tostring(msg)
        if t:sub(1, 5) == "save_" then
          AZB.blocked_saves[t] = (AZB.blocked_saves[t] or 0) + 1
          return true
        end
        return real:push(msg)
      end,
    }, {
      __index = function(_, k)
        local v = real[k]
        if type(v) == "function" then
          return function(_, ...) return v(real, ...) end
        end
        return v
      end,
    })
  end
  -- remove_save() deletes <profile>/save.jkr directly.
  remove_save = function() AZB.blocked_saves.remove_save = (AZB.blocked_saves.remove_save or 0) + 1 end
end

----------------------------------------------------------------------------
-- Oracle-parity hooks
----------------------------------------------------------------------------
function AZB.unlock_all()
  if not AZB.config.unlock_all then return end
  for _, t in ipairs({G.P_CENTERS, G.P_BLINDS, G.P_TAGS, G.P_SEALS}) do
    for _, v in pairs(t or {}) do
      if type(v) == "table" then
        v.unlocked = true; v.discovered = true; v.alerted = true
      end
    end
  end
  G.P_LOCKED = {}
end

if not AZB._card_init_hooked then
  AZB._card_init_hooked = true
  local card_init = Card.init
  function Card:init(...)
    card_init(self, ...)
    if AZB.config.stable_unique_val and self.sort_id then
      self.unique_val = 1 - (self.sort_id * 50) / 1603301
    end
    if AZB.capture then AZB.capture[#AZB.capture + 1] = self end
  end
end

AZB.install_save_guard()
AZB.unlock_all()

-- Vanilla `used_jokers` bookkeeping, tracked alongside SMODS's. Vanilla
-- marks/clears every center sharing the card's *name* (e.g. all four
-- "Standard Pack" art variants); SMODS tracks the exact key. For vanilla
-- content only booster names are shared and boosters never consult
-- used_jokers, so this is representation-only; the dump reports the vanilla
-- set so it compares with the oracle.
local function vanilla_used()
  if not G.GAME then return nil end
  local u = G.GAME.azb_used_jokers
  if not u then u = {}; G.GAME.azb_used_jokers = u end
  return u
end
AZB.vanilla_used = vanilla_used

local function vanilla_find(name)
  for _, area in ipairs({G.jokers, G.consumeables}) do
    for _, v in pairs(area and area.cards or {}) do
      if type(v) == 'table' and v.ability and v.ability.name == name then return true end
    end
  end
  return false
end

-- RNG streams that only SMODS creates (its own soul roll for modded
-- "soul" cards; SMODS 26.x small/big blind pool rolls); excluded from the
-- dump's `rng` (reported under az). Result-neutral for vanilla content.
AZB.SMODS_ONLY_RNG = {'^soul_smods_', '^small$', '^big$'}
function AZB.smods_only_rng(k)
  for _, pat in ipairs(AZB.SMODS_ONLY_RNG) do
    if tostring(k):find(pat) then return true end
  end
  return false
end

-- Vanilla get_current_pool rolls 'rarity'..ante..append even for legendary
-- jokers (then ignores it); SMODS skips the roll. Burn it like vanilla so
-- the stream (and math.random's reseed) stay in step. The value is unused.
if AZB.config.smods_compat and not AZB._gcp_hooked then
  AZB._gcp_hooked = true
  local gcp = get_current_pool
  function get_current_pool(_type, _rarity, _legendary, _append, ...)
    if _type == 'Joker' and _legendary and _rarity == nil and G.GAME and G.GAME.round_resets then
      pseudorandom('rarity' .. G.GAME.round_resets.ante .. (_append or ''))
    end
    return gcp(_type, _rarity, _legendary, _append, ...)
  end
end

-- SMODS's Card:apply_to_run also creates a display copy of the voucher in
-- G.vouchers (vanilla creates no card), which consumes a sort_id and shifts
-- every later card id. Give such display cards negative ids and roll the
-- counter back, so ids keep matching vanilla.
if AZB.config.smods_compat and not AZB._apply_hooked then
  AZB._apply_hooked = true
  local apply_to_run = Card.apply_to_run
  AZB.display_id = 0
  function Card.apply_to_run(self, ...)
    local before = G.sort_id
    local outer = AZB.capture
    AZB.capture = {}
    -- vanilla apply_to_run creates no card, so it never marks used_jokers
    local used = G.GAME and G.GAME.azb_used_jokers
    local used_snap = used and {}
    if used then for k, v in pairs(used) do used_snap[k] = v end end
    local ok, r = pcall(apply_to_run, self, ...)
    if G.GAME then
      if used_snap then
        if G.GAME.azb_used_jokers == used then G.GAME.azb_used_jokers = used_snap end
      else
        G.GAME.azb_used_jokers = nil
      end
    end
    local made = AZB.capture
    AZB.capture = outer
    if outer then for _, c in ipairs(made) do outer[#outer + 1] = c end end
    if not ok then error(r, 0) end
    local all_display = #made > 0 and before ~= nil
    for _, c in ipairs(made) do
      if c.area ~= G.vouchers then all_display = false end
    end
    if all_display and G.sort_id == before + #made then
      for _, c in ipairs(made) do
        AZB.display_id = AZB.display_id - 1
        c.sort_id = AZB.display_id
      end
      G.sort_id = before
    end
    return r
  end
end

-- played_this_ante (The Pillar): vanilla sets it on the highlighted cards
-- when the hand is played (before scoring) and Card:set_ability rebuilds the
-- whole ability table, dropping it (Vampire / Midas Mask during scoring,
-- enhancing tarots later). SMODS sets it in an event after evaluation, and
-- its set_ability merges into the existing table, so the flag survives.
-- Restore vanilla: mark at play time, clear it in set_ability (see the
-- set_ability wrapper below), and after SMODS's post-evaluation events clear
-- it again on cards that went through set_ability since they were played.
if AZB.config.smods_compat and not AZB._pta_hooked then
  AZB._pta_hooked = true
  local played = {}
  local play = G.FUNCS.play_cards_from_highlighted
  G.FUNCS.play_cards_from_highlighted = function(...)
    played = {}
    for _, c in ipairs(G.hand and G.hand.highlighted or {}) do
      if c.ability then
        c.ability.played_this_ante = true
        played[#played + 1] = {c, c.azb_set_ability or 0}
      end
    end
    return play(...)
  end
  local evaluate = G.FUNCS.evaluate_play
  G.FUNCS.evaluate_play = function(...)
    local r = evaluate(...)
    local mine = played
    played = {}
    G.E_MANAGER:add_event(Event({func = function()
      for _, p in ipairs(mine) do
        local c, n = p[1], p[2]
        if (c.azb_set_ability or 0) ~= n and c.ability then c.ability.played_this_ante = nil end
      end
      return true
    end}))
    return r
  end
end

-- Vanilla Card:open sets G.GAME.pack_size = ability.extra; SMODS does not.
if not AZB._open_hooked then
  AZB._open_hooked = true
  local open = Card.open
  function Card:open(...)
    if self.ability and self.ability.set == 'Booster' and G.GAME then
      G.GAME.azb_pack_size = self.ability.extra
    end
    return open(self, ...)
  end
end

if not AZB._used_hooked then
  AZB._used_hooked = true
  local set_ability = Card.set_ability
  function Card:set_ability(...)
    local r = set_ability(self, ...)
    if AZB.config.smods_compat and self.ability then
      self.ability.played_this_ante = nil -- vanilla rebuilds the table
      self.azb_set_ability = (self.azb_set_ability or 0) + 1
    end
    if not G.OVERLAY_MENU and self.ability and self.ability.name then
      local u = vanilla_used()
      if u then
        for k, v in pairs(G.P_CENTERS) do
          if v.name == self.ability.name then u[k] = true end
        end
      end
    end
    return r
  end
  local remove = Card.remove
  function Card:remove(...)
    local r = remove(self, ...)
    if not G.OVERLAY_MENU and self.ability and self.ability.name then
      local u = vanilla_used()
      if u and not vanilla_find(self.ability.name) then
        for k, v in pairs(G.P_CENTERS) do
          if v.name == self.ability.name then u[k] = nil end
        end
      end
    end
    return r
  end
end

----------------------------------------------------------------------------
-- State names
----------------------------------------------------------------------------
local state_names
function AZB.state_name(s)
  if not state_names then
    state_names = {}
    for k, v in pairs(G.STATES) do state_names[v] = k end
  end
  return state_names[s] or tostring(s)
end

-- SMODS replaces the vanilla *_PACK states by SMODS_BOOSTER_OPENED; report
-- the vanilla name (what the oracle and the sim use) from the booster kind.
local KIND_STATE = {
  Arcana = "TAROT_PACK", Celestial = "PLANET_PACK", Spectral = "SPECTRAL_PACK",
  Standard = "STANDARD_PACK", Buffoon = "BUFFOON_PACK",
}
function AZB.opened_booster_kind()
  local b = SMODS and SMODS.OPENED_BOOSTER
  local c = b and b.config and b.config.center
  return c and c.kind or nil
end
function AZB.vanilla_state_name()
  if G.STATES.SMODS_BOOSTER_OPENED and G.STATE == G.STATES.SMODS_BOOSTER_OPENED then
    return KIND_STATE[AZB.opened_booster_kind() or ""] or "SMODS_BOOSTER_OPENED"
  end
  return AZB.state_name(G.STATE)
end

function AZB.is_pack_state(s)
  s = s or G.STATE
  local S = G.STATES
  return s == S.TAROT_PACK or s == S.PLANET_PACK or s == S.SPECTRAL_PACK or s == S.STANDARD_PACK
    or s == S.BUFFOON_PACK or (S.SMODS_BOOSTER_OPENED ~= nil and s == S.SMODS_BOOSTER_OPENED)
end

----------------------------------------------------------------------------
-- Settle predicate (port of harness.lua blocking_pending/stable_state)
----------------------------------------------------------------------------
function AZB.blocking_pending()
  local n = 0
  for _, ev in ipairs(G.E_MANAGER.queues.base) do
    if ev.blocking then n = n + 1 end
  end
  return n
end

-- Port of H.rebase: every action starts at G.TIMERS.TOTAL = 0 (pending
-- TOTAL-timed events are shifted), so timer arithmetic matches the oracle
-- and the sim bit for bit.
function AZB.rebase()
  if not AZB.config.rebase then return end
  local T = G.TIMERS.TOTAL
  for _, q in pairs(G.E_MANAGER.queues) do
    for _, ev in ipairs(q) do
      if ev.timer == 'TOTAL' then
        ev.time = ev.time - T
        if ev.ease and ev.ease.start_time then
          ev.ease.start_time = ev.ease.start_time - T
          ev.ease.end_time = ev.ease.end_time - T
        end
      end
    end
  end
  G.TIMERS.TOTAL = 0
end

function AZB.stable_state()
  local s = G.STATE
  local S = G.STATES
  if s == S.SELECTING_HAND then return G.STATE_COMPLETE and G.buttons ~= nil end
  if s == S.SHOP then return G.STATE_COMPLETE and G.shop ~= nil and G.shop_jokers ~= nil end
  if s == S.BLIND_SELECT then return G.STATE_COMPLETE and G.blind_select ~= nil end
  if s == S.ROUND_EVAL then return G.STATE_COMPLETE and G.round_eval ~= nil and G.GAME.current_round.dollars ~= nil end
  if AZB.is_pack_state(s) then
    return G.STATE_COMPLETE and G.pack_cards ~= nil and G.pack_cards.cards ~= nil and #G.pack_cards.cards > 0
  end
  if s == S.GAME_OVER then return true end
  return false
end

function AZB.locks()
  local out = {}
  for k, v in pairs(G.CONTROLLER.locks) do if v then out[#out+1] = tostring(k) end end
  table.sort(out)
  return out
end

function AZB.quiet_now()
  if not (G and G.E_MANAGER and G.CONTROLLER and G.GAME) then return false end
  local locked = false
  for _, v in pairs(G.CONTROLLER.locks) do if v then locked = true end end
  return AZB.blocking_pending() == 0 and not locked and AZB.stable_state() and (G.GAME.STOP_USE or 0) == 0
end

function AZB.diagnostics()
  return {
    state = AZB.state_name(G.STATE),
    vanilla_state = AZB.vanilla_state_name(),
    state_complete = G.STATE_COMPLETE and true or false,
    blocking = AZB.blocking_pending(),
    queue_len = #G.E_MANAGER.queues.base,
    locks = AZB.locks(),
    controller_locked = G.CONTROLLER.locked and true or false,
    stop_use = G.GAME and G.GAME.STOP_USE or nil,
    stable_state = AZB.stable_state() and true or false,
    paused = G.SETTINGS.paused and true or false,
    overlay = G.OVERLAY_MENU ~= nil,
    quiet = AZB.quiet,
    frame = AZB.frame,
    gamespeed = G.SETTINGS.GAMESPEED,
    timers_total = G.TIMERS.TOTAL,
    real_dt = G.real_dt,
  }
end

----------------------------------------------------------------------------
-- Frame monitor: AZB.quiet counts consecutive quiet frames since the last
-- action. Waiters (deferred RPC responses) fire once quiet >= 3.
----------------------------------------------------------------------------
AZB.frame = 0
AZB.quiet = 0
AZB.skip_tick = false
AZB.waiters = {}

-- Called right after anything mutates the game (from inside a dispatch).
function AZB.mark_action()
  AZB.quiet = 0
  -- From a dispatch, the next tick runs before any game update: ignore it.
  -- From inside a tick (deferred action), the next tick follows an update.
  if not AZB.in_tick then AZB.skip_tick = true end
end

function AZB.is_settled()
  if G.STATE == G.STATES.GAME_OVER then return true end
  return AZB.quiet >= AZB.config.quiet_frames
end

local function now() return love.timer.getTime() end

-- Queue `fn(ok, reason)` to run once the game settles (or times out).
function AZB.when_settled(fn, timeout)
  AZB.waiters[#AZB.waiters + 1] = {fn = fn, deadline = now() + (timeout or 120), frames = 0}
end

local function run_waiters()
  if #AZB.waiters == 0 then return end
  -- Winning pauses the game behind an overlay; dismissing it is "endless
  -- mode" (the harness does the same while settling).
  if G.SETTINGS.paused and G.OVERLAY_MENU and G.STATE ~= G.STATES.GAME_OVER then
    G.FUNCS.exit_overlay_menu()
  end
  local keep = {}
  local ready = {}
  for _, w in ipairs(AZB.waiters) do
    w.frames = w.frames + 1
    if AZB.is_settled() then ready[#ready+1] = {w, true, "settled"}
    elseif now() > w.deadline then ready[#ready+1] = {w, false, "timeout"}
    else keep[#keep+1] = w end
  end
  AZB.waiters = keep
  AZB.in_tick = true
  for _, r in ipairs(ready) do
    local ok, err = pcall(r[1].fn, r[2], r[3], r[1].frames)
    if not ok then sendErrorMessage("waiter failed: " .. tostring(err), "AZB") end
  end
  AZB.in_tick = false
end

function AZB.tick()
  AZB.frame = AZB.frame + 1
  if AZB.skip_tick then
    AZB.skip_tick = false
    AZB.quiet = 0
  elseif AZB.quiet_now() then
    AZB.quiet = AZB.quiet + 1
  else
    AZB.quiet = 0
  end
  run_waiters()
end

local love_update = love.update
love.update = function(dt)
  love_update(dt)

  local ok, err = pcall(AZB.tick)
  if not ok then sendErrorMessage("tick failed: " .. tostring(err), "AZB") end
end
