-- The mod's JSON-RPC methods and where they are served.
--
-- With BalatroBot loaded (the developer tools start the game with it) the
-- methods are registered on its dispatcher. Without it, a normal game, this
-- file serves the same protocol itself: JSON-RPC 2.0 over HTTP POST "/", one
-- request per connection, on 127.0.0.1 only.

local socket = require("socket")

AZB.rpc = {endpoints = {}}

-- BalatroBot's error names (antezero_bridge.client.RPCError.name).
local ERR = {BAD_REQUEST = -32001, INVALID_STATE = -32002, NOT_ALLOWED = -32003, INTERNAL_ERROR = -32000}
AZB.rpc.BAD_REQUEST, AZB.rpc.INVALID_STATE = 'BAD_REQUEST', 'INVALID_STATE'
AZB.rpc.NOT_ALLOWED, AZB.rpc.INTERNAL_ERROR = 'NOT_ALLOWED', 'INTERNAL_ERROR'

function AZB.rpc.register(ep)
  AZB.rpc.endpoints[ep.name] = ep
  if BB_DISPATCHER then
    local ok, err = BB_DISPATCHER.register(ep)
    if not ok then sendErrorMessage('register ' .. ep.name .. ': ' .. tostring(err), 'AnteZero') end
  end
end

if AZB.is_bridge() then return end

local S = {port = tonumber(os.getenv("AZB_PORT")) or 12346}
AZB.server = S

-- A request's socket stays with its reply (some replies wait for the game to
-- settle), so other requests are served meanwhile.
local function reply(c, id, response)
  local body
  if response.message then
    local name = response.name or 'INTERNAL_ERROR'
    body = {jsonrpc = "2.0", id = id, error = {code = ERR[name] or ERR.INTERNAL_ERROR, message = response.message, data = {name = name}}}
  else
    body = {jsonrpc = "2.0", id = id, result = response}
  end
  local ok, text = pcall(JSON.encode, body)
  if not ok then text = '{"jsonrpc":"2.0","id":null,"error":{"code":-32000,"message":"unencodable result","data":{"name":"INTERNAL_ERROR"}}}' end
  c:settimeout(10) -- state dumps are larger than one non-blocking write
  c:send("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " .. #text .. "\r\nConnection: close\r\n\r\n" .. text)
  c:close()
end

local function dispatch(c, req)
  local id = req.id
  local function send(response) reply(c, id, response) end
  local ep = type(req.method) == 'string' and AZB.rpc.endpoints[req.method]
  if not ep then return send({message = 'unknown method ' .. tostring(req.method), name = 'BAD_REQUEST'}) end
  local params = type(req.params) == 'table' and req.params or {}
  for k, f in pairs(ep.schema) do
    local v = params[k]
    local t = type(v)
    if v == nil then
      if f.required then return send({message = "missing '" .. k .. "'", name = 'BAD_REQUEST'}) end
    elseif not (t == f.type or (f.type == 'integer' and t == 'number' and v == math.floor(v))) then
      return send({message = "'" .. k .. "' must be a " .. f.type, name = 'BAD_REQUEST'})
    end
  end
  local ok, err = pcall(ep.execute, params, send)
  if not ok then send({message = tostring(err), name = 'INTERNAL_ERROR'}) end
end

-- One request is read at a time (called once per game update).
function S.update()
  if not S.socket then return end
  if not S.client then
    local c = S.socket:accept()
    if not c then return end
    c:settimeout(0)
    S.client, S.buffer = c, ""
  end
  local c = S.client
  local chunk, err, partial = c:receive(8192)
  S.buffer = S.buffer .. (chunk or partial or "")
  local head = S.buffer:find("\r\n\r\n", 1, true)
  local len = head and tonumber(S.buffer:sub(1, head):lower():match("content%-length:%s*(%d+)")) or 0
  if head and #S.buffer >= head + 3 + len then
    local ok, req = pcall(JSON.decode, S.buffer:sub(head + 4, head + 3 + len))
    S.client, S.buffer = nil, nil
    if ok and type(req) == 'table' then dispatch(c, req) else reply(c, nil, {message = 'invalid JSON', name = 'BAD_REQUEST'}) end
  elseif err ~= "timeout" then -- the client went away
    c:close()
    S.client, S.buffer = nil, nil
  end
end

AZB.rpc.register({
  name = 'health', description = 'liveness', schema = {},
  execute = function(_, send) send({status = 'ok'}) end,
})

local sock, err = socket.bind("127.0.0.1", S.port)
if sock then
  sock:settimeout(0)
  S.socket = sock
else
  sendWarnMessage('port ' .. S.port .. ' is taken (' .. tostring(err) .. '): the bot button is off', 'AnteZero')
end
