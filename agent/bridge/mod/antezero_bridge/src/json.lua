-- Deterministic JSON encoder (object keys sorted, numbers as %.17g).
-- Verbatim copy of the encoder in sim/oracle/lua/harness.lua, so that the
-- Rust `oracle::json::parse` + `canon::from_lua` consume our dumps unchanged.
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
return json
