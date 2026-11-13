-- The bot button: a toggle on the New Run screen's Play button. A run
-- started with it on is played by the agent (antezero_agent.bot, a Python
-- process this file starts and that talks to the methods in endpoints.lua).
--
-- A bot run is played in step with the simulator, so it runs under the
-- oracle's conditions: seeded, everything unlocked, the fixed clock of
-- core.lua. None of that is the player's: nothing is saved during the run,
-- and their profile, unlocks and tutorial state are put back when it ends.
-- While the bot plays, input is limited to Esc, Run Info and Options, whose
-- menus pause it.

local B = {
  enabled = false,   -- the toggle
  pending = false,   -- the run being started is the bot's
  protected = false, -- the player's state is set aside (see protect)
  active = false,    -- the bot is playing
  run = nil,         -- {id, seed, back, stake} of the bot's run, while it is on screen
  runs = 0,
  acc = 0,
  line = '',         -- what the strip shows
}
AZB.bot = B

-- The player's settings (Mods > AnteZero > Config; defaults in config.lua).
local cfg = SMODS.current_mod.config

local function now() return love.timer.getTime() end

----------------------------------------------------------------------------
-- The player's state
----------------------------------------------------------------------------
local TABLES = function() return {G.P_CENTERS, G.P_BLINDS, G.P_TAGS, G.P_SEALS} end

local function protect()
  if B.protected then return end
  local s = {flags = {}, locked = G.P_LOCKED, profile = copy_table(G.PROFILES[G.SETTINGS.profile]),
             tutorial = {G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress}}
  for _, t in ipairs(TABLES()) do
    for _, v in pairs(t or {}) do
      if type(v) == 'table' then s.flags[v] = {v.unlocked, v.discovered, v.alerted} end
    end
  end
  B.saved = s
  B.protected = true
  AZB.guard_saves(true)
end

function B.release()
  B.active, B.run = false, nil
  if B.hud then B.hud:remove(); B.hud = nil end
  AZB.cancel_waiters()
  if not B.protected then return end
  local s = B.saved
  for v, f in pairs(s.flags) do v.unlocked, v.discovered, v.alerted = f[1], f[2], f[3] end
  G.P_LOCKED = s.locked
  local live = G.PROFILES[G.SETTINGS.profile]
  for k in pairs(live) do live[k] = nil end
  for k, v in pairs(s.profile) do live[k] = v end
  G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress = s.tutorial[1], s.tutorial[2]
  B.saved = nil
  AZB.guard_saves(false)
  B.protected = false
end

----------------------------------------------------------------------------
-- The clock (core.lua's love.update)
----------------------------------------------------------------------------
function B.stepping()
  return B.protected and (B.active or B.pending) and not AZB.is_bridge()
end

function B.steps(dt)
  B.acc = B.acc + dt * 60 * cfg.speed
  local n = math.floor(B.acc)
  B.acc = B.acc - n
  return math.min(n, 16)
end

----------------------------------------------------------------------------
-- Starting a bot run
----------------------------------------------------------------------------
local rng
local function random_seed()
  rng = rng or love.math.newRandomGenerator(os.time(), math.floor(now() * 1e6))
  local a, out = '123456789ABCDEFGHIJKLMNPQRSTUVWXYZ', {}
  for i = 1, 8 do
    local k = rng:random(#a)
    out[i] = a:sub(k, k)
  end
  return table.concat(out)
end

-- The command that starts the Python agent, as `antezero_agent.bot --install` stored it.
local function command()
  local ok, cfg = pcall(JSON.decode, love.filesystem.read('antezero.json') or '')
  return ok and type(cfg) == 'table' and type(cfg.command) == 'string' and cfg.command or nil
end

local function spawn()
  if B.contact and now() - B.contact < 3 then return end       -- it is polling
  if B.spawned and now() - B.spawned < 120 then return end      -- it is loading
  local cmd = command()
  if not cmd then return end -- the toggle is off without it; B.tick reports a bot that never answers
  local port = AZB.server and AZB.server.port or (BB_SETTINGS and BB_SETTINGS.port) or 12346
  os.execute('start "AnteZero bot" /min ' .. cmd .. ' --port ' .. port)
  B.spawned = now()
end

local function stop(text)
  B.active = false
  B.status, B.last = text, nil
end

local start_run = Game.start_run
function Game:start_run(args)
  args = args or {}
  local bot = B.pending and not args.savetext and not args.challenge
  B.pending = false
  if not bot then
    B.release()
    return start_run(self, args)
  end
  protect()
  B.active, B.run = false, nil
  AZB.cancel_waiters()
  -- As the oracle harness starts a run (act.lua's AZB.start).
  AZB.unlock_all()
  G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress = true, nil
  if not args.seed or args.seed == '' or args.seed == B.auto_seed then
    args.seed = random_seed()
    B.auto_seed = args.seed
  end
  AZB.rebase()
  G.sort_id = 0
  G.playing_card = 0
  local r = start_run(self, args)
  B.runs = B.runs + 1
  B.run = {id = B.runs, seed = G.GAME.pseudorandom.seed, back = G.GAME.selected_back.effect.center.key, stake = G.GAME.stake}
  B.active, B.acc, B.seen, B.status, B.last, B.thinking, B.line = true, 0, now(), 'starting...', nil, nil, ''
  B.make_hud()
  spawn()
  AZB.mark_action()
  return r
end

local main_menu = Game.main_menu
function Game:main_menu(...)
  B.release()
  return main_menu(self, ...)
end

-- The Play buttons: the vanilla screen's, and Steamodded's run select.
local function play_button(name, is_new_run)
  local f = G.FUNCS[name]
  if not f then return end
  G.FUNCS[name] = function(...)
    B.pending = B.enabled and is_new_run()
    if B.pending then protect() end
    return f(...)
  end
end
play_button('start_setup_run', function() return G.SETTINGS.current_setup == 'New Run' end)
play_button('run_select_start_run', function() return true end)

----------------------------------------------------------------------------
-- The toggle
----------------------------------------------------------------------------
local atlas = SMODS.Atlas({key = 'bot', path = 'bot.png', px = 16, py = 16})

local function icon(size, cell)
  local s = Sprite(0, 0, size, size, G.ASSET_ATLAS[atlas.key], {x = cell, y = 0})
  s.states.drag.can = false
  return s
end

G.FUNCS.azb_toggle_bot = function() B.enabled = not B.enabled end

-- Off: dark with a grey bot. On: gold with a dark bot and a white edge.
G.FUNCS.azb_bot_toggle = function(e)
  e.config.colour = B.enabled and G.C.GOLD or G.C.BLACK
  e.config.outline_colour = B.enabled and G.C.WHITE or G.C.UI.TEXT_INACTIVE
  local sprite = e.children[1].config.object
  local cell = B.enabled and 0 or 2
  if sprite.sprite_pos.x ~= cell then sprite:set_sprite_pos({x = cell, y = 0}) end
end

-- On the Play button `e` while `show`, like the game's alert badges.
local function toggle_on(e, show)
  if show and not e.children.alert then
    local ready = command() ~= nil
    if not ready then B.enabled = false end
    e.children.alert = UIBox{
      definition = {n = G.UIT.ROOT, config = {align = 'cm', colour = G.C.CLEAR}, nodes = {
        {n = G.UIT.C, config = {align = 'cm', r = 0.1, minw = 0.66, minh = 0.66, colour = G.C.BLACK, outline = 1.5, outline_colour = G.C.UI.TEXT_INACTIVE,
            hover = true, shadow = true, button = ready and 'azb_toggle_bot' or nil, func = 'azb_bot_toggle',
            tooltip = {title = 'AnteZero', text = ready and {'Let the bot play this run'}
              or {'Not set up. Run:', 'python -m antezero_agent.bot', '--install CHECKPOINT'}}}, nodes = {
          {n = G.UIT.O, config = {object = icon(0.5, 2)}},
        }},
      }},
      config = {instance_type = 'ALERT', align = 'tri', offset = {x = 0.05, y = -0.25}, major = e, parent = e},
    }
  elseif not show and e.children.alert then
    e.children.alert:remove()
    e.children.alert = nil
  end
end

local function with_toggle(name, show)
  local f = G.FUNCS[name]
  if not f then return end
  G.FUNCS[name] = function(e)
    f(e)
    toggle_on(e, show(e))
  end
end
with_toggle('can_start_run', function(e) return G.SETTINGS.current_setup == 'New Run' and e.config.button ~= nil end)
with_toggle('run_select_can_change_page', function(e) return e.config.button == 'run_select_start_run' end)

----------------------------------------------------------------------------
-- What the bot is doing (a strip at the top of the screen)
----------------------------------------------------------------------------
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

-- An action of the bot is about to be performed (endpoints.lua, az_act).
function B.acting(a, note)
  local ok, text = pcall(describe, a)
  B.last = (ok and text or a.type) .. (note and note ~= '' and ('   ' .. note) or '')
  B.seen, B.thinking = now(), nil
end

local function paused() return G.OVERLAY_MENU ~= nil and not (G.GAME and G.GAME.won) end

function B.tick()
  if not B.run then return end
  if paused() then B.seen = now() end -- the agent may be waiting in an action the menu holds
  if B.active and now() - B.seen > (B.contact and 300 or 120) then
    stop(B.contact and 'stopped: the bot does not answer' or 'stopped: the bot did not start (see the "AnteZero bot" window)')
  end
  local line
  if not B.active then line = B.status or ''
  elseif paused() then line = 'paused (close the menu to go on)'
  elseif not B.last then line = B.status or ''
  elseif B.thinking and now() - B.thinking > 0.7 then line = B.last .. '   thinking...'
  else line = B.last end
  if line ~= B.line then
    B.line = line
    B.make_hud()
  end
end

----------------------------------------------------------------------------
-- Settings
----------------------------------------------------------------------------
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
    create_option_cycle({label = 'Bot speed', scale = 0.8, options = names, opt_callback = 'azb_bot_speed', current_option = at}),
  }}
end

----------------------------------------------------------------------------
-- Input while the bot plays
----------------------------------------------------------------------------
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
  if locked() and key ~= 'escape' then return end
  return key_press(self, key)
end

local button_press = Controller.button_press
function Controller:button_press(button)
  if locked() and button ~= 'start' then return end
  return button_press(self, button)
end

----------------------------------------------------------------------------
-- The agent's side
----------------------------------------------------------------------------
AZB.rpc.register({
  name = 'az_bot',
  description = "The bot run to play, if any; `status` is shown while it plays, `stop` ends the bot's part with that text",
  schema = {
    status = {type = 'string', required = false},
    stop = {type = 'string', required = false},
    run = {type = 'integer', required = false},
  },
  execute = function(args, send)
    B.contact = now()
    if not args.run or B.owns(args.run) then
      if B.active then B.seen = now() end
      if args.stop and args.run then stop(args.stop)
      elseif args.status then
        if B.last then B.thinking = B.thinking or now() else B.status = args.status end
      end
    end
    send({active = B.active, run = B.active and B.run or nil, paused = paused(), search = cfg.search})
  end,
})
