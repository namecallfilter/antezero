-- JSON-RPC methods registered on BalatroBot's dispatcher, plus two small
-- robustness patches to BalatroBot's server.

local function bad(send, msg)
  send({message = tostring(msg), name = BB_ERROR_NAMES.BAD_REQUEST})
end

local function err_text(e)
  if type(e) == 'table' and e.azb then return e.msg end
  return tostring(e)
end

-- Respond with the state once settled (or a timeout report).
local function respond_settled(send, timeout, extra)
  AZB.when_settled(function(ok, reason, frames)
    local r = {state = AZB.state(), settled = ok, reason = reason, frames = frames}
    if not ok then r.diag = AZB.diagnostics() end
    for k, v in pairs(extra or {}) do r[k] = v end
    send(r)
  end, timeout)
end

-- Run `fn` on a settled game: immediately if already settled, else once it is.
local function on_settled(send, timeout, fn)
  if AZB.is_settled() then return fn() end
  AZB.when_settled(function(ok)
    if ok then fn() else
      send({message = 'game did not settle before the action: ' .. AZB.json.encode(AZB.diagnostics()),
            name = BB_ERROR_NAMES.INVALID_STATE})
    end
  end, timeout)
end

local ENDPOINTS = {
  {
    name = 'az_info',
    description = 'antezero bridge version, config and settle diagnostics',
    schema = {},
    execute = function(_, send)
      local bs = {}
      for k, v in pairs(AZB.blocked_saves) do bs[k] = v end
      send({
        version = AZB.VERSION,
        smods_version = SMODS and SMODS.version or nil,
        balatrobot_version = SMODS.Mods and SMODS.Mods.balatrobot and SMODS.Mods.balatrobot.version or nil,
        config = AZB.config,
        blocked_saves = bs,
        settled = AZB.is_settled(),
        diag = AZB.diagnostics(),
        gamespeed = G.SETTINGS.GAMESPEED,
        fps_cap = G.FPS_CAP,
      })
    end,
  },
  {
    name = 'az_state',
    description = 'Oracle-shaped state dump (JSON string, harness H.state format)',
    schema = {
      settle = {type = 'boolean', required = false, description = 'wait until settled (default true)'},
      timeout = {type = 'number', required = false, description = 'settle timeout in seconds'},
    },
    execute = function(args, send)
      if args.settle == false then
        send({state = AZB.state(), settled = AZB.is_settled(), reason = 'immediate', frames = 0})
      else
        respond_settled(send, args.timeout)
      end
    end,
  },
  {
    name = 'az_start',
    description = 'Start a seeded run exactly like the oracle H.start(seed, back, stake)',
    schema = {
      seed = {type = 'string', required = true},
      back = {type = 'string', required = false, description = "back key, e.g. 'b_red'"},
      stake = {type = 'integer', required = false},
      timeout = {type = 'number', required = false},
    },
    execute = function(args, send)
      AZB.waiters = {} -- a new run invalidates pending waits
      local ok, e = pcall(AZB.start, args.seed, args.back, args.stake)
      if not ok then return bad(send, err_text(e)) end
      respond_settled(send, args.timeout)
    end,
  },
  {
    name = 'az_act',
    description = 'Perform one action exactly like the oracle H.act (1-based indices), then settle',
    schema = {
      action = {type = 'table', required = true},
      settle = {type = 'boolean', required = false},
      timeout = {type = 'number', required = false},
    },
    execute = function(args, send)
      local a = args.action
      local function finish()
        if args.settle == false then
          send({state = AZB.state(), settled = false, reason = 'immediate', frames = 0})
        else
          respond_settled(send, args.timeout)
        end
      end
      local function main()
        local ok, e = pcall(AZB.act_main, a)
        if not ok then return bad(send, err_text(e)) end
        finish()
      end
      -- Run `step` once settled again (the harness's intermediate H.settle).
      local function after_settle(step)
        AZB.when_settled(function(ok)
          if ok then step() else
            send({message = 'game did not settle mid-action: ' .. AZB.json.encode(AZB.diagnostics()),
                  name = BB_ERROR_NAMES.INTERNAL_ERROR})
          end
        end, args.timeout)
      end
      on_settled(send, args.timeout, function()
        local ok, needs = pcall(AZB.act_prepare, a)
        if not ok then return bad(send, err_text(needs)) end
        if needs then return after_settle(main) end
        if AZB.act_needs_unhighlight(a) then
          AZB.act_unhighlight()
          return after_settle(main)
        end
        main()
      end)
    end,
  },
  {
    name = 'az_inject',
    description = 'Scenario injection exactly like the oracle H.inject, then settle',
    schema = {
      inject = {type = 'table', required = true},
      timeout = {type = 'number', required = false},
    },
    execute = function(args, send)
      on_settled(send, args.timeout, function()
        local ok, e = pcall(AZB.inject, args.inject)
        if not ok then return bad(send, err_text(e)) end
        respond_settled(send, args.timeout)
      end)
    end,
  },
  {
    name = 'az_legal',
    description = 'Legal sim action codes at the current (settled) decision point',
    schema = {
      timeout = {type = 'number', required = false},
    },
    execute = function(args, send)
      on_settled(send, args.timeout, function()
        local ok, codes, ph = pcall(AZB.legal_codes)
        if not ok then return send({message = tostring(codes), name = BB_ERROR_NAMES.INTERNAL_ERROR}) end
        codes.__array = true
        send({codes_json = AZB.json.encode(codes), phase = ph, state = AZB.vanilla_state_name()})
      end)
    end,
  },
  {
    name = 'az_eval',
    description = 'Evaluate a Lua expression (debug; requires AZB_ALLOW_EVAL=1)',
    schema = {code = {type = 'string', required = true}},
    execute = function(args, send)
      if not AZB.config.allow_eval then
        return send({message = 'az_eval disabled (set AZB_ALLOW_EVAL=1)', name = BB_ERROR_NAMES.NOT_ALLOWED})
      end
      local f, perr = loadstring('return ' .. args.code)
      if not f then f, perr = loadstring(args.code) end
      if not f then return bad(send, perr) end
      local ok, res = xpcall(f, debug.traceback)
      if not ok then return send({message = tostring(res), name = BB_ERROR_NAMES.INTERNAL_ERROR}) end
      local okj, js = pcall(AZB.json.encode, res)
      send({result_json = okj and js or tostring(res)})
    end,
  },
}

local READ_ONLY = {
  health = true, gamestate = true, ['rpc.discover'] = true, screenshot = true, save = true,
}

local function install()
  if AZB._endpoints_installed then return true end
  if not (BB_DISPATCHER and BB_DISPATCHER.register and BB_SERVER) then return false end
  for _, ep in ipairs(ENDPOINTS) do
    local ok, err = BB_DISPATCHER.register(ep)
    if not ok then sendErrorMessage('register ' .. ep.name .. ': ' .. tostring(err), 'AZB') end
  end
  -- 1) Any other (BalatroBot) mutating method restarts the settle count.
  local dispatch = BB_DISPATCHER.dispatch
  BB_DISPATCHER.dispatch = function(request)
    local m = type(request) == 'table' and request.method or nil
    local mutating = type(m) == 'string' and not READ_ONLY[m] and m:sub(1, 3) ~= 'az_'
    if mutating then AZB.rebase() end
    dispatch(request)
    if mutating then AZB.mark_action() end
  end
  -- 2) BalatroBot sends on a non-blocking socket and ignores partial
  --    writes; large responses (our state dumps) would be truncated.
  local send_response = BB_SERVER.send_response
  BB_SERVER.send_response = function(response)
    if BB_SERVER.client_socket then BB_SERVER.client_socket:settimeout(10) end
    return send_response(response)
  end
  AZB._endpoints_installed = true
  sendInfoMessage('registered ' .. #ENDPOINTS .. ' endpoints on BalatroBot', 'AZB')
  return true
end

if not install() then
  -- BalatroBot not initialised yet: retry on the first frames.
  local love_update = love.update
  love.update = function(dt)
    if not AZB._endpoints_installed then install() end
    return love_update(dt)
  end
end
