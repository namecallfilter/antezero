-- Headless Balatro harness.
--
-- Loads the unmodified game scripts from HARNESS_SRC with the LÖVE engine
-- replaced by inert stubs, then exposes a small API (table H) to start runs,
-- perform player actions through the game's own button callbacks, advance
-- frames until the game is waiting for input, and dump the logical state as
-- JSON for comparison with the Rust simulator.

local SRC = HARNESS_SRC

----------------------------------------------------------------------------
-- JSON encoding (deterministic: object keys sorted)
----------------------------------------------------------------------------
local json = {}
local function esc(s)
  return (s:gsub('[%c"\\]', function(c)
    if c == '"' then return '\\"' elseif c == '\\' then return '\\\\'
    elseif c == '\n' then return '\\n' else return string.format('\\u%04x', c:byte()) end
  end))
end
local function enc(v, out)
  local t = type(v)
  if t == 'nil' then out[#out+1] = 'null'
  elseif t == 'boolean' then out[#out+1] = v and 'true' or 'false'
  elseif t == 'number' then
    if v ~= v then out[#out+1] = '"nan"'
    elseif v == math.huge then out[#out+1] = '"inf"'
    elseif v == -math.huge then out[#out+1] = '"-inf"'
    else out[#out+1] = string.format('%.17g', v) end
  elseif t == 'string' then out[#out+1] = '"'..esc(v)..'"'
  elseif t == 'table' then
    if v.__array or v[1] ~= nil or next(v) == nil then
      out[#out+1] = '['
      for i = 1, #v do if i > 1 then out[#out+1] = ',' end enc(v[i], out) end
      out[#out+1] = ']'
    else
      local keys = {}
      for k in pairs(v) do keys[#keys+1] = tostring(k) end
      table.sort(keys)
      out[#out+1] = '{'
      for i, k in ipairs(keys) do
        if i > 1 then out[#out+1] = ',' end
        out[#out+1] = '"'..esc(k)..'":'
        local val = v[k]; if val == nil then val = v[tonumber(k)] end
        enc(val, out)
      end
      out[#out+1] = '}'
    end
  else out[#out+1] = '"<'..t..'>"' end
end
function json.encode(v) local out = {} enc(v, out) return table.concat(out) end
JSON = json

----------------------------------------------------------------------------
-- LÖVE stubs
----------------------------------------------------------------------------
local function noop() end
local function Obj(methods)
  return setmetatable(methods or {}, {__index = function() return noop end})
end
local function Image(w, h)
  return Obj{
    getDimensions = function() return w, h end,
    getWidth = function() return w end,
    getHeight = function() return h end,
  }
end
local FakeFont = Obj{
  getWidth = function(self, t) return 10*#tostring(t or 'x') end,
  getHeight = function() return 20 end,
}

local fake_time = 0
local function file_exists(path)
  local f = io.open(SRC..'/'..path, 'rb')
  if f then f:close() return true end
  return false
end
local function read_file(path)
  local f = io.open(SRC..'/'..path, 'rb')
  if not f then return nil end
  local s = f:read('*a'); f:close(); return s
end

love = {
  graphics = Obj{
    newImage = function() return Image(1024, 1024) end,
    newQuad = function() return Obj() end,
    newFont = function() return FakeFont end,
    setNewFont = function() return FakeFont end,
    newShader = function() return Obj() end,
    newCanvas = function() return Obj() end,
    newText = function() return Obj() end,
    newVideo = function() return Obj{getWidth = function() return 1 end, getHeight = function() return 1 end} end,
    getWidth = function() return 1920 end,
    getHeight = function() return 1080 end,
    getDimensions = function() return 1920, 1080 end,
    isActive = function() return false end,
    isCreated = function() return true end,
  },
  filesystem = Obj{
    getInfo = function(path) if file_exists(path) then return {type = 'file'} end return nil end,
    read = function(path) return read_file(path) end,
    getDirectoryItems = function() return {} end,
    load = function(path) local s = read_file(path); return s and loadstring(s) end,
    getSourceBaseDirectory = function() return SRC end,
  },
  thread = Obj{
    newThread = function() return Obj() end,
    getChannel = function() return Obj{pop = function() return nil end} end,
  },
  timer = Obj{
    getTime = function() return fake_time end,
    step = function() return 1/60 end,
    getFPS = function() return 60 end,
  },
  window = Obj{
    getMode = function() return 1920, 1080, {fullscreen = false, vsync = 1, display = 1} end,
    getDisplayCount = function() return 1 end,
    getDesktopDimensions = function() return 1920, 1080 end,
    getFullscreenModes = function() return {{width = 1920, height = 1080}} end,
    setMode = function() return true end,
    isOpen = function() return true end,
    toPixels = function(x) return x end,
    getTitle = function() return 'Balatro' end,
  },
  system = Obj{
    getOS = function() return 'Windows' end,
    getClipboardText = function() return '' end,
  },
  mouse = Obj{ getPosition = function() return 0, 0 end, isVisible = function() return false end },
  joystick = Obj{ getJoysticks = function() return {} end },
  audio = Obj{ newSource = function() return Obj() end },
  event = Obj{ poll = function() return function() return nil end end },
  touch = Obj{ getTouches = function() return {} end },
  data = Obj{ compress = function(_, _, s) return s end, decompress = function(_, _, s) return s end },
  arg = Obj{ parseGameArguments = function(a) return a end },
}

----------------------------------------------------------------------------
-- Boot the game
----------------------------------------------------------------------------
_RELEASE_MODE = true
_DEMO = false
dofile(SRC..'/main.lua')

-- No GC time-slicing (it spins on a frozen clock otherwise).
nuGC = function() end

-- Skip the splash/menu; just create a game object and room.
function Game:splash_screen()
  self:prep_stage(G.STAGES.MAIN_MENU, G.STATES.MENU, true)
end

G.SETTINGS.tutorial_complete = true
G.SETTINGS.tutorial_progress = nil
G:start_up()
G.SETTINGS.tutorial_complete = true
G.SETTINGS.tutorial_progress = nil
-- Canonical timing: the default game speed at 60 fps.
G.SETTINGS.GAMESPEED = 1
G.F_NO_SAVING = true

-- Fully unlocked & discovered profile.
local function unlock_all()
  for _, t in ipairs({G.P_CENTERS, G.P_BLINDS, G.P_TAGS, G.P_SEALS}) do
    for k, v in pairs(t) do
      v.unlocked = true; v.discovered = true; v.alerted = true
    end
  end
  G.P_LOCKED = {}
end
unlock_all()

----------------------------------------------------------------------------
-- Frame driving
----------------------------------------------------------------------------
H = {}
H.dt = 1/60
H.frame = 0
-- Frames since the current action started (timeline traces use it).
H.tframe = 0
H.trace = {}

function H.step(n)
  for _ = 1, (n or 1) do
    fake_time = fake_time + H.dt
    H.tframe = H.tframe + 1
    G:update(H.dt)
    H.frame = H.frame + 1
  end
end

-- Every action starts at G.TIMERS.TOTAL = 0 (pending timers are shifted),
-- so the simulator can reproduce the timer arithmetic bit for bit.
function H.rebase()
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
  H.tframe = 0
  H.trace = {}
end

local function T(s) H.trace[#H.trace+1] = H.tframe..' '..s end
local pseudoseed_orig = pseudoseed
function pseudoseed(key, predict)
  if not predict and key ~= 'seed' then T('rng '..key) end
  return pseudoseed_orig(key, predict)
end
local function area_name(a)
  if a == G.hand then return 'hand' elseif a == G.play then return 'play'
  elseif a == G.deck then return 'deck' elseif a == G.discard then return 'discard'
  elseif a == G.jokers then return 'jokers' elseif a == G.consumeables then return 'consumables' end
  return '?'
end
local emplace_orig = CardArea.emplace
function CardArea:emplace(card, location, stay_flipped)
  if H.in_draw and (card.ability.set == 'Default' or card.ability.set == 'Enhanced') then T('draw '..area_name(self)..' '..card.sort_id) end
  return emplace_orig(self, card, location, stay_flipped)
end
local remove_orig = Card.remove
function Card:remove()
  local owned = self.added_to_deck
  if not owned and G.playing_cards then
    for _, v in ipairs(G.playing_cards) do if v == self then owned = true end end
  end
  if owned then T('rm '..self.sort_id) end
  return remove_orig(self)
end
local eval_round_orig = G.FUNCS.evaluate_round
G.FUNCS.evaluate_round = function(...) T('eval_round'); return eval_round_orig(...) end
-- Mark the end of the shop population (update_shop's non-blockable event
-- defined at game.lua:3089, which polls the shop UI position).
local add_event_orig = EventManager.add_event
function EventManager:add_event(ev, queue, front)
  if ev.func then
    local info = debug.getinfo(ev.func, 'S')
    if info.linedefined == 3089 and info.short_src:find('game.lua', 1, true) then
      local f = ev.func
      ev.func = function() local r = f(); if r then T('shop_populate') end; return r end
    end
  end
  return add_event_orig(self, ev, queue, front)
end
-- draw_card (common_events.lua) with the emplace inside its event traced.
function draw_card(from, to, percent, dir, sort, card, delay, mute, stay_flipped, vol, discarded_only)
    percent = percent or 50
    delay = delay or 0.1
    if dir == 'down' then
        percent = 1-percent
    end
    sort = sort or false
    local drawn = nil

    G.E_MANAGER:add_event(Event({
        trigger = 'before',
        delay = delay,
        func = function()
            H.in_draw = true
            if card then
                if from then card = from:remove_card(card) end
                if card then drawn = true end
                local stay_flipped = G.GAME and G.GAME.blind and G.GAME.blind:stay_flipped(to, card)
                if G.GAME.modifiers.flipped_cards and to == G.hand then
                    if pseudorandom(pseudoseed('flipped_card')) < 1/G.GAME.modifiers.flipped_cards then
                        stay_flipped = true
                    end
                end
                to:emplace(card, nil, stay_flipped)
            else
                if to:draw_card_from(from, stay_flipped, discarded_only) then drawn = true end
            end
            H.in_draw = false
            if not mute and drawn then
                if from == G.deck or from == G.hand or from == G.play or from == G.jokers or from == G.consumeables or from == G.discard then
                    G.VIBRATION = G.VIBRATION + 0.6
                end
                play_sound('card1', 0.85 + percent*0.2/100, 0.6*(vol or 1))
            end
            if sort then
                to:sort()
            end
            return true
        end
      }))
end

local function blocking_pending()
  local n = 0
  for _, ev in ipairs(G.E_MANAGER.queues.base) do
    if ev.blocking then n = n + 1 end
  end
  return n
end

local function stable_state()
  local s = G.STATE
  local S = G.STATES
  if s == S.SELECTING_HAND then return G.STATE_COMPLETE and G.buttons ~= nil end
  if s == S.SHOP then return G.STATE_COMPLETE and G.shop ~= nil and G.shop_jokers ~= nil end
  if s == S.BLIND_SELECT then return G.STATE_COMPLETE and G.blind_select ~= nil end
  if s == S.ROUND_EVAL then return G.STATE_COMPLETE and G.round_eval ~= nil and G.GAME.current_round.dollars ~= nil end
  if s == S.TAROT_PACK or s == S.PLANET_PACK or s == S.SPECTRAL_PACK or s == S.STANDARD_PACK or s == S.BUFFOON_PACK then
    return G.STATE_COMPLETE and G.pack_cards ~= nil and G.pack_cards.cards ~= nil and #G.pack_cards.cards > 0
  end
  if s == S.GAME_OVER then return true end
  return false
end

-- Advance until the game waits for player input.
function H.settle(max_frames)
  max_frames = max_frames or 20000
  local quiet = 0
  for i = 1, max_frames do
    if G.STATE == G.STATES.GAME_OVER then return i end
    -- Winning opens a paused overlay; dismissing it is "endless mode".
    if G.SETTINGS.paused and G.OVERLAY_MENU then G.FUNCS.exit_overlay_menu() end
    H.step(1)
    local locked = false
    for k, v in pairs(G.CONTROLLER.locks) do if v then locked = true end end
    if blocking_pending() == 0 and not locked and stable_state() and (G.GAME.STOP_USE or 0) == 0 then
      quiet = quiet + 1
      if quiet >= 3 then return i end
    else
      quiet = 0
    end
  end
  error('settle: game did not stabilise (state '..tostring(G.STATE)..', blocking '..blocking_pending()..')')
end

----------------------------------------------------------------------------
-- Runs and actions
----------------------------------------------------------------------------
-- `unique_val` derives from the global UI node counter (UI-activity
-- dependent, not reproducible); model it as monotone in creation order,
-- exactly like the simulator does.
local card_init = Card.init
function Card:init(...)
  card_init(self, ...)
  self.unique_val = 1 - (self.sort_id*50)/1603301
end

function H.start(seed, back_key, stake)
  G.GAME.viewed_back = Back(G.P_CENTERS[back_key or 'b_red'])
  G:delete_run()
  H.rebase()
  G.sort_id = 0
  G.playing_card = 0
  G:start_run({stake = stake or 1, seed = seed})
  G.GAME.viewed_back = Back(G.P_CENTERS[back_key or 'b_red'])
  H.settle()
  return H.state()
end

local function area_card(area, i)
  local c = area and area.cards and area.cards[i]
  if not c then error('no card '..tostring(i)) end
  return c
end

local function highlight(indices)
  G.hand:unhighlight_all()
  for _, i in ipairs(indices or {}) do
    local c = area_card(G.hand, i)
    if not c.highlighted then G.hand:add_to_highlighted(c) end
  end
end

function H.act(a)
  local k = a.type
  -- Select the cards first and let the selection UI settle (as a player
  -- would before pressing a button); the action itself starts at TOTAL = 0.
  if a.cards and (k == 'play' or k == 'discard' or k == 'use' or k == 'buy_and_use' or k == 'pick') then
    highlight(a.cards)
    H.settle()
  end
  H.rebase()
  if k == 'select_blind' then
    local key = G.GAME.round_resets.blind_choices[G.GAME.blind_on_deck]
    G.FUNCS.select_blind({config = {ref_table = G.P_BLINDS[key]}, UIBox = G.blind_select_opts[string.lower(G.GAME.blind_on_deck)]})
  elseif k == 'skip_blind' then
    G.FUNCS.skip_blind({UIBox = G.blind_select_opts[string.lower(G.GAME.blind_on_deck)]})
  elseif k == 'reroll_boss' then
    G.FUNCS.reroll_boss()
  elseif k == 'play' then
    G.FUNCS.play_cards_from_highlighted()
  elseif k == 'discard' then
    G.FUNCS.discard_cards_from_highlighted()
  elseif k == 'use' then
    if not a.cards then G.hand:unhighlight_all(); H.settle(); H.rebase() end
    G.FUNCS.use_card({config = {ref_table = area_card(G.consumeables, a.index)}})
  elseif k == 'sell_joker' then
    G.FUNCS.sell_card({config = {ref_table = area_card(G.jokers, a.index)}})
  elseif k == 'sell_consumable' then
    G.FUNCS.sell_card({config = {ref_table = area_card(G.consumeables, a.index)}})
  elseif k == 'buy' then
    G.FUNCS.buy_from_shop({config = {ref_table = area_card(G.shop_jokers, a.index)}})
  elseif k == 'buy_and_use' then
    G.FUNCS.buy_from_shop({config = {ref_table = area_card(G.shop_jokers, a.index), id = 'buy_and_use'}})
  elseif k == 'redeem' then
    G.FUNCS.use_card({config = {ref_table = area_card(G.shop_vouchers, a.index)}})
  elseif k == 'open' then
    G.FUNCS.use_card({config = {ref_table = area_card(G.shop_booster, a.index)}})
  elseif k == 'reroll' then
    G.FUNCS.reroll_shop()
  elseif k == 'end_shop' then
    G.FUNCS.toggle_shop()
  elseif k == 'cash_out' then
    G.FUNCS.cash_out({config = {}})
  elseif k == 'pick' then
    G.FUNCS.use_card({config = {ref_table = area_card(G.pack_cards, a.index)}})
  elseif k == 'skip_pack' then
    G.FUNCS.skip_booster()
  elseif k == 'move_joker' then
    local c = table.remove(G.jokers.cards, a.from)
    table.insert(G.jokers.cards, a.to, c)
    G.jokers:set_ranks()
  elseif k == 'move_hand' then
    local c = table.remove(G.hand.cards, a.from)
    table.insert(G.hand.cards, a.to, c)
    G.hand:set_ranks()
  elseif k == 'sort_hand' then
    if a.by == 'suit' then G.FUNCS.sort_hand_suit() else G.FUNCS.sort_hand_value() end
  else
    error('unknown action '..tostring(k))
  end
  H.settle()
  return H.state()
end

-- Scenario injection (mirrors sim/src/inject.rs).
function H.inject(t)
  H.rebase()
  if t.kind == 'joker' or t.kind == 'consumable' then
    add_joker(t.key, t.edition, true, t.eternal)
  elseif t.kind == 'card' then
    local c = create_playing_card({front = G.P_CARDS[t.key], center = G.P_CENTERS[t.center or 'c_base']}, G.deck, true, true)
    if t.edition then c:set_edition({[t.edition] = true}, true, true) end
    if t.seal then c:set_seal(t.seal, true, true) end
  elseif t.kind == 'dollars' then
    G.GAME.dollars = G.GAME.dollars + t.amount
  elseif t.kind == 'voucher' then
    G.GAME.used_vouchers[t.key] = true
    Card.apply_to_run(nil, G.P_CENTERS[t.key])
  end
  H.settle()
  return H.state()
end

----------------------------------------------------------------------------
-- State dump
----------------------------------------------------------------------------
local state_names = {}
for k, v in pairs(G.STATES) do state_names[v] = k end

local function edition_of(c)
  return c.edition and c.edition.type or nil
end

local function card_info(c)
  local t = {
    key = c.config.center_key,
    sort_id = c.sort_id,
    edition = edition_of(c),
    seal = c.seal,
    debuff = c.debuff or false,
    cost = c.cost,
    sell_cost = c.sell_cost,
    facing = c.facing,
  }
  if c.ability.set == 'Default' or c.ability.set == 'Enhanced' then
    t.card = c.config.card_key
    t.rank = c.base.id
    t.suit = c.base.suit
    t.bonus = c.ability.bonus
    t.perma_bonus = c.ability.perma_bonus
  end
  local ab = {}
  for kk, vv in pairs(c.ability) do
    local tv = type(vv)
    if tv == 'number' or tv == 'string' or tv == 'boolean' then ab[kk] = vv
    elseif tv == 'table' and kk == 'extra' then
      local e = {}
      for k3, v3 in pairs(vv) do if type(v3) ~= 'table' then e[k3] = v3 end end
      ab.extra = e
    end
  end
  ab.consumeable = nil
  t.ability = ab
  return t
end

local function area_info(area)
  local out = {__array = true}
  if area and area.cards then
    for i, c in ipairs(area.cards) do out[i] = card_info(c) end
  end
  return out
end

function H.state()
  local g = G.GAME
  local s = {
    frame = H.frame,
    trace = H.trace,
    state = state_names[G.STATE] or tostring(G.STATE),
    ante = g.round_resets.ante,
    round = g.round,
    dollars = g.dollars,
    chips = g.chips,
    hands_left = g.current_round.hands_left,
    discards_left = g.current_round.discards_left,
    reroll_cost = g.current_round.reroll_cost,
    blind = g.blind and {name = g.blind.name, chips = g.blind.chips, disabled = g.blind.disabled, dollars = g.blind.dollars} or nil,
    blind_on_deck = g.blind_on_deck,
    blind_states = g.round_resets.blind_states,
    blind_choices = g.round_resets.blind_choices,
    blind_tags = g.round_resets.blind_tags,
    voucher = g.current_round.voucher,
    hand = area_info(G.hand),
    deck = area_info(G.deck),
    discard = area_info(G.discard),
    jokers = area_info(G.jokers),
    consumables = area_info(G.consumeables),
    shop_jokers = area_info(G.shop_jokers),
    shop_vouchers = area_info(G.shop_vouchers),
    shop_booster = area_info(G.shop_booster),
    pack_cards = G.pack_cards and area_info(G.pack_cards) or nil,
    pack_choices = g.pack_choices,
    joker_limit = G.jokers and G.jokers.config.card_limit,
    consumable_limit = G.consumeables and G.consumeables.config.card_limit,
    hand_limit = G.hand and G.hand.config.card_limit,
    hands = {},
    rng = {},
    won = g.won,
    skips = g.skips,
    free_rerolls = g.current_round.free_rerolls,
    rr_hands = g.round_resets.hands,
    rr_discards = g.round_resets.discards,
    rr_reroll_cost = g.round_resets.reroll_cost,
    interest_cap = g.interest_cap,
    interest_amount = g.interest_amount,
    discount_percent = g.discount_percent,
    bankrupt_at = g.bankrupt_at,
    probabilities = g.probabilities.normal,
    shop_joker_max = g.shop.joker_max,
    tarot_rate = g.tarot_rate, planet_rate = g.planet_rate, edition_rate = g.edition_rate,
    playing_card_rate = g.playing_card_rate, spectral_rate = g.spectral_rate,
    last_hand_played = g.last_hand_played,
    last_tarot_planet = g.last_tarot_planet,
    most_played = g.current_round.most_played_poker_hand,
    round_dollars = g.current_round.dollars,
    idol = {rank = g.current_round.idol_card.id, suit = g.current_round.idol_card.suit},
    mail = g.current_round.mail_card.id,
    ancient = g.current_round.ancient_card.suit,
    castle = g.current_round.castle_card.suit,
    hands_played = g.hands_played,
    hands_played_round = g.current_round.hands_played,
    discards_used = g.current_round.discards_used,
    unused_discards = g.unused_discards,
    ecto_minus = g.ecto_minus,
    gros_michel_extinct = g.pool_flags.gros_michel_extinct or false,
    boss_rerolled = g.round_resets.boss_rerolled,
    starting_deck_size = g.starting_deck_size,
    sort_mode = G.hand and G.hand.config.sort,
    pack_size = g.pack_size,
  }
  s.used_jokers = {}
  for k, v in pairs(g.used_jokers) do if v then s.used_jokers[#s.used_jokers+1] = k end end
  table.sort(s.used_jokers)
  s.used_jokers.__array = true
  local uv = {__array = true}
  for k, v in pairs(g.used_vouchers) do if v then uv[#uv+1] = k end end
  table.sort(uv)
  s.used_vouchers = uv
  s.tag_info = {__array = true}
  for i, t in ipairs(g.tags) do s.tag_info[i] = {key = t.key, triggered = t.triggered, orbital = t.ability and t.ability.orbital_hand} end
  for k, v in pairs(g.hands) do
    s.hands[k] = {level = v.level, chips = v.chips, mult = v.mult, played = v.played, played_this_round = v.played_this_round, visible = v.visible}
  end
  for k, v in pairs(g.pseudorandom) do s.rng[k] = v end
  return json.encode(s)
end

-- Entry point used by the Rust side: evaluates a Lua expression.
function oracle(cmd)
  local f = assert(loadstring('return '..cmd))
  local ok, res = xpcall(f, debug.traceback)
  if not ok then error(res, 0) end
  return res
end
