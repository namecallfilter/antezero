-- The bot toggle, in a run's options menu (Esc). Turned on, the agent
-- (antezero_agent.bot, a Python process this file starts and that talks to
-- the methods in endpoints.lua) plays the run from where it is; turned off,
-- or when the bot stops, the run is the player's again. It is off at the
-- start of every run, new or continued.
--
-- The agent rebuilds its simulator from the game's state before every
-- decision, so the bot can take over any run it can read. With "Bot runs
-- count" off, the run stops counting when the bot takes over, for the rest of
-- it: the saved run stays as it was then (Continue returns there), no
-- achievement is awarded, what the game saves meanwhile is the profile as it
-- was then (with the player's settings and choices for the next run), and
-- the profile, unlocks and discoveries are put back when the run ends. With
-- it on, the run counts like one the player played. Either way the tutorial
-- is skipped while the bot plays (and saved as it was), the run plays at the
-- bot's speed setting (the clock of core.lua), and input is limited to Esc,
-- Run Info and Options, whose menus pause it. At a win the bot waits on the
-- win screen and plays on if the player chooses Endless Mode.

local B = {
  protected = false, -- the player's state is set aside (see protect): the run does not count
  active = false,    -- the bot is playing (the toggle)
  run = nil,         -- {id, seed, back, stake} of the bot's turn at the run, while it is on screen
  runs = 0,
  acc = 0,
  line = '',         -- what the strip shows
}
AZB.bot = B

-- The player's settings (Mods > AnteZero > Config; defaults in config.lua).
local cfg = SMODS.current_mod.config

local function now() return love.timer.getTime() end

local TABLES = function() return {G.P_CENTERS, G.P_BLINDS, G.P_TAGS, G.P_SEALS} end

-- What only the player's run setup writes to the profile: the choices for
-- the next run. They are the player's even while a run does not count.
local CHOICES = {'MEMORY', 'last_choices'}

local function with_choices(saved, live)
  local p = {}
  for k, v in pairs(saved) do p[k] = v end
  for _, k in ipairs(CHOICES) do p[k] = live[k] end
  return p
end

local function protect()
  if B.protected or cfg.count then return end
  local s = {flags = {}, locked = {unpack(G.P_LOCKED)}, profile = copy_table(G.PROFILES[G.SETTINGS.profile]), won = G.GAME.won}
  for _, t in ipairs(TABLES()) do
    for _, v in pairs(t or {}) do
      if type(v) == 'table' then s.flags[v] = {v.unlocked, v.discovered, v.alerted, v.alert} end
    end
  end
  B.saved = s
  B.protected = true
  AZB.guard_saves(true)
end

-- Puts the profile and unlocks back as they were when the run stopped
-- counting; the set-aside copy stays as it was, for the next time.
local function restore()
  local s = B.saved
  for v, f in pairs(s.flags) do v.unlocked, v.discovered, v.alerted, v.alert = f[1], f[2], f[3], f[4] end
  G.P_LOCKED = {unpack(s.locked)}
  -- as unlock_card orders the decks
  table.sort(G.P_CENTER_POOLS.Back, function(a, b) return (a.order - (a.unlocked and 100 or 0)) < (b.order - (b.unlocked and 100 or 0)) end)
  local live = G.PROFILES[G.SETTINGS.profile]
  local profile = copy_table(with_choices(s.profile, live))
  for k in pairs(live) do live[k] = nil end
  for k, v in pairs(profile) do live[k] = v end
  -- Steamodded's per-stake views of the usage tables (wins_by_key, ...) were
  -- copied bound to the run's tables; bind them anew, as loading a profile does.
  if convert_save_data then convert_save_data(live, true) end
  set_discover_tallies()
  set_profile_progress()
  -- A progress save queued during the run would still write the run's unlocks and discoveries.
  if G.FILE_HANDLER and G.FILE_HANDLER.progress then G:save_progress() end
end

-- Ends the bot's part in the run on screen, which is leaving: its turn, the
-- skipped tutorial and, if the run did not count, the set-aside profile.
function B.release()
  B.active, B.run = false, nil
  if B.hud then B.hud:remove(); B.hud = nil end
  AZB.cancel_waiters()
  if B.tutorial then
    G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress = B.tutorial[1], B.tutorial[2]
    B.tutorial = nil
  end
  if not B.protected then return end
  restore()
  -- The game judges the run left behind (the streak) by G.GAME: the
  -- player's is the run they handed over, not the bot's win.
  G.GAME.won = B.saved.won
  B.saved = nil
  AZB.guard_saves(false)
  B.protected = false
end

-- The next run is chosen from what the player has: the New Run screen of a
-- run that does not count is built on the profile without the run's unlocks.
local run_setup = G.UIDEF.run_setup
function G.UIDEF.run_setup(...)
  if B.protected then restore() end
  return run_setup(...)
end

local unlock_achievement_ = unlock_achievement
function unlock_achievement(...)
  if B.protected then return end
  return unlock_achievement_(...)
end

-- The player's pending unlock notes (shown, then deleted, by the end
-- screens' New Run) wait for the player's next run.
local unlock_notify_ = unlock_notify
function unlock_notify(...)
  if B.protected then return end
  return unlock_notify_(...)
end

-- What a save the game sends while the bot has a part in the run writes
-- (core.lua's save channel asks; nil: nothing): the player's tutorial, and
-- while the run does not count, the profile and unlocks as they were when it
-- stopped counting. Run saves are stopped where they are made (G.F_NO_SAVING),
-- so one sent now was queued before and is the player's.
function B.outgoing(msg)
  local t = msg.type
  if not (B.tutorial or B.protected) or t == 'save_run' then return msg end
  if t ~= 'save_settings' and t ~= 'save_progress' then return not B.protected and msg or nil end
  local p = msg.save_progress
  local settings = p and p.SETTINGS or msg.save_settings
  if B.tutorial then
    local s = {}
    for k, v in pairs(settings) do s[k] = v end
    s.tutorial_complete, s.tutorial_progress = B.tutorial[1], B.tutorial[2]
    settings = s
  end
  if not B.protected then
    if p then return {type = t, save_progress = {UDA = p.UDA, SETTINGS = settings, PROFILE = p.PROFILE}} end
    return {type = t, save_settings = settings, profile_num = msg.profile_num, save_profile = msg.save_profile}
  end
  local s = B.saved
  local profile = with_choices(s.profile, G.PROFILES[G.SETTINGS.profile])
  if not p then return {type = t, save_settings = settings, profile_num = msg.profile_num, save_profile = profile} end
  local uda = {} -- as Game:save_progress writes the flags
  for _, tbl in ipairs(TABLES()) do
    for k, v in pairs(tbl or {}) do
      local f = s.flags[v]
      if f then uda[k] = (f[1] and 'u' or '') .. (f[2] and 'd' or '') .. (f[3] and 'a' or '') end
    end
  end
  return {type = t, save_progress = {UDA = uda, SETTINGS = settings, PROFILE = profile}}
end

function B.stepping()
  return B.active and not AZB.is_bridge()
end

function B.steps(dt)
  B.acc = B.acc + dt * 60 * cfg.speed
  local n = math.floor(B.acc)
  B.acc = B.acc - n
  return math.min(n, 16)
end

-- The command that starts the Python agent, as `antezero_agent.bot --install` stored it.
local function command()
  local ok, c = pcall(JSON.decode, love.filesystem.read('antezero.json') or '')
  return ok and type(c) == 'table' and type(c.command) == 'string' and c.command or nil
end

local function spawn()
  if B.contact and now() - B.contact < 3 then return end       -- it is polling
  if B.spawned and now() - B.spawned < 120 then return end      -- it is loading
  local port = AZB.server and AZB.server.port or (BB_SETTINGS and BB_SETTINGS.port) or 12346
  os.execute('start "AnteZero bot" /min ' .. command() .. ' --port ' .. port)
  B.spawned = now()
end

-- Why the bot cannot take over the run on screen, if it cannot.
local function refusal()
  if not command() then return {'Not set up. Run:', 'python -m antezero_agent.bot', '--install CHECKPOINT'} end
  if G.GAME.challenge then return {'The bot does not play challenge runs'} end
  if G.STATE == G.STATES.GAME_OVER then return {'The run is over'} end
end

-- The game numbers cards from a counter it does not save (G.sort_id), so a
-- run continued after a restart gives new cards the numbers of cards it
-- already has, and the simulator tells cards apart by number: the newer of
-- two cards with one number gets a new one, and the counter goes past them all.
local function renumber()
  local seen, top = {}, G.sort_id or 0
  for _, c in ipairs(G.I.CARD) do top = math.max(top, c.sort_id or 0) end
  for _, c in ipairs(G.I.CARD) do -- oldest first
    if c.sort_id and seen[c.sort_id] then
      top = top + 1
      c.sort_id = top
    end
    if c.sort_id then seen[c.sort_id] = true end
  end
  G.sort_id = top
end

local function take_over()
  protect()
  renumber()
  if not G.SETTINGS.tutorial_complete and not B.tutorial then
    B.tutorial = {G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress}
    G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress = true, nil
    -- as skipping a section of it ends its overlay
    if G.OVERLAY_TUTORIAL then
      if G.OVERLAY_TUTORIAL.Jimbo then G.OVERLAY_TUTORIAL.Jimbo:remove() end
      if G.OVERLAY_TUTORIAL.content then G.OVERLAY_TUTORIAL.content:remove() end
      G.OVERLAY_TUTORIAL:remove()
      G.OVERLAY_TUTORIAL = nil
      G.E_MANAGER:clear_queue('tutorial')
    end
  end
  B.runs = B.runs + 1
  B.run = {id = B.runs, seed = G.GAME.pseudorandom.seed, back = G.GAME.selected_back.effect.center.key, stake = G.GAME.stake}
  B.active, B.acc, B.seen, B.status, B.last, B.thinking, B.line = true, 0, now(), 'starting...', nil, nil, ''
  B.make_hud()
  spawn()
  AZB.mark_action()
end

local function stop(text)
  B.active = false
  B.status, B.last = text, nil
end

-- The toggle's callback (it has set B.active to `on`).
local function toggled(on)
  if not on then return stop('handed back to you') end
  if refusal() then B.active = false return end
  take_over()
end

-- Every way out of the run on screen ends the bot's part in it before
-- anything is written for the next one: the start of a run (the Play
-- buttons write its choices to the profile first, holding R the end of the
-- streak) and the way to the main menu.
local function release_first(t, name)
  local f = t and t[name]
  if not f then return end
  t[name] = function(...)
    B.release()
    return f(...)
  end
end
release_first(G.FUNCS, 'start_setup_run')
release_first(G.FUNCS, 'start_challenge_run')
release_first(G.FUNCS, 'go_to_menu')
release_first(SMODS.RunSelect and SMODS.RunSelect.Functions, 'start_run')

-- Holding R restarts the run on these conditions (and the ones of the game's code).
local key_hold_update = Controller.key_hold_update
function Controller:key_hold_update(key, dt)
  if key == 'r' and not (self.locked or self.locks.frame or self.frame_buttonpress or G.SETTINGS.paused)
      and (self.held_key_times[key] or 0) > 0.7 then
    B.release()
  end
  return key_hold_update(self, key, dt)
end

local start_run = Game.start_run
function Game:start_run(...)
  B.release()
  return start_run(self, ...)
end

local main_menu = Game.main_menu
function Game:main_menu(...)
  B.release()
  return main_menu(self, ...)
end

local atlas = SMODS.Atlas({key = 'bot', path = 'bot.png', px = 16, py = 16})

local function icon(size, cell)
  local s = Sprite(0, 0, size, size, G.ASSET_ATLAS[atlas.key], {x = cell, y = 0})
  s.states.drag.can = false
  return s
end

local function toggle_node()
  local why = refusal()
  return create_toggle({
    label = 'Bot plays this run', ref_table = B, ref_value = 'active', callback = toggled, w = 4,
    info = why or (B.protected and {'The run no longer counts'}
                   or cfg.count and {'The run counts (Mods > AnteZero > Config)'}
                   or {'From here the run does not count', '(Mods > AnteZero > Config)'}),
  })
end

-- Below the run's buttons (Settings ... Customize Deck).
local options = create_UIBox_options
function create_UIBox_options(...)
  local t = options(...)
  local box = G.STAGE == G.STAGES.RUN and t.nodes[1].nodes[1].nodes[1]
  if box then
    local rows = {}
    for i = 1, table.maxn(box.nodes) do rows[#rows+1] = box.nodes[i] end
    rows[#rows+1] = toggle_node()
    box.nodes = rows
  end
  return t
end

-- Built anew for every text: the game re-lays out a live text node only when
-- its character count changes, which leaves the backdrop at the old width.
function B.make_hud()
  if B.hud then B.hud:remove() end
  B.hud = UIBox{
    definition = {n = G.UIT.ROOT, config = {align = 'cm', colour = G.C.CLEAR}, nodes = {
      {n = G.UIT.R, config = {align = 'cm', padding = 0.07, r = 0.1, colour = adjust_alpha(G.C.BLACK, 0.8)}, nodes = {
        {n = G.UIT.O, config = {object = icon(0.3, 1)}},
        {n = G.UIT.T, config = {text = ' AnteZero  ', scale = 0.3, colour = G.C.ORANGE, shadow = true}},
        {n = G.UIT.T, config = {text = B.line, scale = 0.3, colour = G.C.UI.TEXT_LIGHT}},
      }},
    }},
    config = {instance_type = 'ALERT', align = 'tmi', offset = {x = 0, y = -0.62}, major = G.ROOM_ATTACH}, -- above the joker slots
  }
  B.hud.states.collide.can = false
end

local function card_name(c)
  if not c then return '?' end
  if c.ability.set == 'Default' or c.ability.set == 'Enhanced' then return c.base.value .. ' of ' .. c.base.suit end
  return c.ability.name
end

local function describe(a)
  local k = a.type
  local card = function(area, i) return card_name(area and area.cards and area.cards[i]) end
  if k == 'select_blind' then
    local b = G.P_BLINDS[G.GAME.round_resets.blind_choices[G.GAME.blind_on_deck]]
    return 'Play ' .. (b and b.name or 'blind')
  elseif k == 'skip_blind' then return 'Skip blind'
  elseif k == 'reroll_boss' then return 'Reroll boss'
  elseif k == 'play' then
    local h = G.GAME.current_round.current_hand.handname
    return 'Play ' .. ((h and h ~= '' and h ~= '????') and h or (#(a.cards or {}) .. ' cards')) -- ????: face-down cards
  elseif k == 'discard' then return 'Discard ' .. #(a.cards or {})
  elseif k == 'use' then return 'Use ' .. card(G.consumeables, a.index)
  elseif k == 'sell_joker' then return 'Sell ' .. card(G.jokers, a.index)
  elseif k == 'sell_consumable' then return 'Sell ' .. card(G.consumeables, a.index)
  elseif k == 'buy' or k == 'buy_and_use' then return 'Buy ' .. card(G.shop_jokers, a.index)
  elseif k == 'redeem' then return 'Redeem ' .. card(G.shop_vouchers, a.index)
  elseif k == 'open' then return 'Open ' .. card(G.shop_booster, a.index)
  elseif k == 'reroll' then return 'Reroll shop'
  elseif k == 'end_shop' then return 'Leave shop'
  elseif k == 'cash_out' then return 'Cash out'
  elseif k == 'pick' then return 'Take ' .. card(G.pack_cards, a.index)
  elseif k == 'skip_pack' then return 'Skip pack'
  elseif k == 'move_joker' then return 'Reorder jokers'
  end
  return 'Reorder hand'
end

function B.owns(id) return B.active and B.run ~= nil and B.run.id == id end

function B.acting(a, note)
  local ok, text = pcall(describe, a)
  B.last = (ok and text or a.type) .. (note and note ~= '' and ('   ' .. note) or '')
  B.seen, B.thinking = now(), nil
end

-- A menu the player opened (the win screen too: the player chooses there),
-- not the screen that ends the run.
local function paused()
  return G.OVERLAY_MENU ~= nil and G.STATE ~= G.STATES.GAME_OVER
end

local function win_screen()
  return G.OVERLAY_MENU ~= nil and G.OVERLAY_MENU:get_UIE_by_ID('you_win_UI') ~= nil
end

function B.tick()
  if not B.run then return end
  if paused() then -- the agent may be waiting in an action the menu holds
    B.seen, B.interrupted = now(), true
  end
  if B.active and now() - B.seen > (B.contact and 300 or 120) then
    stop(B.contact and 'stopped: the bot does not answer' or 'stopped: the bot did not start (see the "AnteZero bot" window)')
  end
  local line
  if not B.active then line = (B.status or '') .. (B.protected and '   (this run does not count)' or '')
  elseif win_screen() then line = 'won the run (Endless Mode lets the bot play on)'
  elseif paused() then line = 'paused (close the menu to go on)'
  elseif not B.last then line = B.status or ''
  elseif B.thinking and now() - B.thinking > 0.7 then line = B.last .. '   thinking...'
  else line = B.last end
  if line ~= B.line then
    B.line = line
    B.make_hud()
  end
end

local SPEEDS = {0.5, 1, 2, 4, 8, 16}

G.FUNCS.azb_bot_speed = function(args) cfg.speed = SPEEDS[args.to_key] end

SMODS.current_mod.config_tab = function()
  local at, names = 2, {}
  for i, v in ipairs(SPEEDS) do
    names[i] = v .. 'x'
    if v == cfg.speed then at = i end
  end
  return {n = G.UIT.ROOT, config = {align = 'cm', padding = 0.1, colour = G.C.CLEAR}, nodes = {
    create_toggle({label = 'Search (stronger, pauses to think)', ref_table = cfg, ref_value = 'search'}),
    create_toggle({label = 'Bot runs count', ref_table = cfg, ref_value = 'count',
                   info = {'Wins, losses, stats, stake progress, unlocks, discoveries', 'and achievements are saved to your profile'}}),
    create_option_cycle({label = 'Bot speed', scale = 0.8, options = names, opt_callback = 'azb_bot_speed', current_option = at}),
  }}
end

local function locked() return B.active and not G.OVERLAY_MENU and not G.SETTINGS.paused end

local function allowed(node)
  while node do
    local b = node.config and node.config.button
    if b == 'run_info' or b == 'options' then return true end
    node = node.parent
  end
  return false
end

local L_press = Controller.queue_L_cursor_press
function Controller:queue_L_cursor_press(x, y)
  if locked() and not allowed(self.hovering.target) then return end
  return L_press(self, x, y)
end

local R_press = Controller.queue_R_cursor_press
function Controller:queue_R_cursor_press(x, y)
  if locked() then return end
  return R_press(self, x, y)
end

local key_press = Controller.key_press
function Controller:key_press(key)
  if locked() and key ~= 'escape' and key ~= 'r' then return end -- the run's menu, and holding R
  return key_press(self, key)
end

local button_press = Controller.button_press
function Controller:button_press(button)
  if locked() and button ~= 'start' then return end
  return button_press(self, button)
end

AZB.rpc.register({
  name = 'az_bot',
  description = "The bot's turn at the run on screen, if it has one, and (with `run`) whether a menu was open since the last such call (`interrupted`); `status` is shown while it plays, `stop` ends its turn with that text",
  schema = {
    status = {type = 'string', required = false},
    stop = {type = 'string', required = false},
    run = {type = 'integer', required = false},
  },
  execute = function(args, send)
    B.contact = now()
    local interrupted = false
    if not args.run or B.owns(args.run) then
      if B.active then B.seen = now() end
      if args.stop and args.run then stop(args.stop)
      elseif args.status then
        if B.last then B.thinking = B.thinking or now() else B.status = args.status end
      end
    end
    if args.run and B.owns(args.run) then
      interrupted, B.interrupted = B.interrupted or false, false
    end
    send({active = B.active, run = B.active and B.run or nil, paused = paused(), interrupted = interrupted, search = cfg.search})
  end,
})
