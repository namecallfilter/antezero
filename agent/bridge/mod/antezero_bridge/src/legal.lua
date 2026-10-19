-- Legal sim action codes at the current decision point: a port of
-- Game::check / Game::legal_actions in sim/src/actions.rs, evaluated on the
-- live game. Consumable usability calls the game's own
-- Card:can_use_consumeable() with each candidate selection temporarily set
-- as G.hand.highlighted. Like the sim, move_joker/move_hand/sort are legal
-- but not enumerated.

local function phase()
  local S = G.STATES
  local s = G.STATE
  if s == S.GAME_OVER then return 'game_over' end
  if s == S.BLIND_SELECT then return 'blind_select' end
  if s == S.SELECTING_HAND then return 'round' end
  if s == S.SHOP then return 'shop' end
  if s == S.ROUND_EVAL then return 'round_eval' end
  if AZB.is_pack_state(s) then return 'pack' end
  return nil
end

-- Card masks cover the first 32 hand cards (the sim's CardMask).
local MASK_BITS = 32

local function popcount(m)
  local n = 0
  while m > 0 do
    if m % 2 == 1 then n = n + 1 end
    m = math.floor(m / 2)
  end
  return n
end

local function bit(m, i) return math.floor(m / 2 ^ i) % 2 == 1 end

-- The lowest set bit of m > 0.
local function lowbit(m)
  local b = 1
  while math.floor(m / b) % 2 == 0 do b = b * 2 end
  return b
end

local function forced_mask()
  local m = 0
  for i, c in ipairs(G.hand and G.hand.cards or {}) do
    if i > MASK_BITS then break end
    if c.ability and c.ability.forced_selection then m = m + 2 ^ (i - 1) end
  end
  return m
end

local function includes(m, sub)
  for i = 0, MASK_BITS - 1 do
    if bit(sub, i) and not bit(m, i) then return false end
  end
  return true
end

-- All masks over the hand with 1..max cards, in increasing order (the sim's
-- for_each_subset: a mask with too many cards skips ahead by its lowest
-- set bit, so the work stays proportional to the result).
local function subsets(max)
  local n = math.min(#(G.hand and G.hand.cards or {}), MASK_BITS)
  local out = {}
  local total = 2 ^ n
  local m = 1
  while m < total do
    if popcount(m) > max then
      m = m + lowbit(m)
    else
      out[#out+1] = m
      m = m + 1
    end
  end
  return out
end

local function spendable() return G.GAME.dollars - G.GAME.bankrupt_at end

local function highlight_limits(c)
  local cons = c.ability and c.ability.consumeable
  if type(cons) ~= 'table' then return 0, 0 end
  return cons.max_highlighted or 0, cons.min_highlighted or 0
end

local function can_use_with(c, m)
  local tmp = {}
  if m and m > 0 then
    for i, hc in ipairs(G.hand.cards) do
      if bit(m, i - 1) then tmp[#tmp+1] = hc end
    end
  end
  local saved = G.hand.highlighted
  G.hand.highlighted = tmp
  local ok, res = pcall(c.can_use_consumeable, c)
  G.hand.highlighted = saved
  return ok and res and true or false
end

-- Port of Game::consumable_usable (area-independent parts are inside
-- can_use_consumeable itself).
local function usable_masks(c)
  local out = {}
  local key = c.config.center_key
  local fm = forced_mask()
  local max_h = highlight_limits(c)
  local uses_targets = max_h > 0 or key == 'c_aura'
  local ankh_blocked = key == 'c_ankh' and #G.jokers.cards >= G.jokers.config.card_limit
  if ankh_blocked then return out end
  if not uses_targets then
    -- (a card forced by The Cerulean Bell does not block these)
    if can_use_with(c, 0) then out[1] = 0 end
    return out
  end
  local max = (max_h == 0) and 1 or math.min(max_h, 5)
  for _, m in ipairs(subsets(max)) do
    if includes(m, fm) and can_use_with(c, m) then out[#out+1] = m end
  end
  return out
end

local function has_buy_space(c)
  local neg = (c.edition and c.edition.negative) and 1 or 0
  local set = c.ability.set
  if set == 'Voucher' or set == 'Default' or set == 'Enhanced' then return true end
  if set == 'Joker' then return #G.jokers.cards < G.jokers.config.card_limit + neg end
  if c.ability.consumeable then return #G.consumeables.cards < G.consumeables.config.card_limit + neg end
  return false
end

local function is_consumable(c) return c.ability and c.ability.consumeable ~= nil end

function AZB.legal_codes()
  local out = {}
  local function add(s) out[#out+1] = s end
  local ph = phase()
  if ph == nil or ph == 'game_over' then return out, ph end
  local hand_n = #(G.hand and G.hand.cards or {})
  local fm = forced_mask()
  if ph == 'blind_select' then
    add('select')
    if G.GAME.blind_on_deck == 'Small' or G.GAME.blind_on_deck == 'Big' then add('skip') end
    local uv = G.GAME.used_vouchers
    if (uv.v_retcon or (uv.v_directors_cut and not G.GAME.round_resets.boss_rerolled)) and spendable() - 10 >= 0 then
      add('reroll_boss')
    end
  elseif ph == 'round' then
    local can_discard = G.GAME.current_round.discards_left > 0
    for _, m in ipairs(subsets(5)) do
      if includes(m, fm) then
        add('play ' .. string.format('%d', m))
        if can_discard then add('discard ' .. string.format('%d', m)) end
      end
    end
  elseif ph == 'shop' then
    for i, c in ipairs(G.shop_jokers.cards) do
      local afford = not (c.cost > spendable() and c.cost > 0)
      if afford and has_buy_space(c) then add('buy ' .. (i - 1)) end
      if is_consumable(c) and afford then
        for _, m in ipairs(usable_masks(c)) do add(string.format('buy_use %d %d', i - 1, m)) end
      end
    end
    for i, c in ipairs(G.shop_vouchers.cards) do
      if not (c.cost > spendable()) then add('redeem ' .. (i - 1)) end
    end
    for i, c in ipairs(G.shop_booster.cards) do
      if not (c.cost > 0 and c.cost > spendable()) then add('open ' .. (i - 1)) end
    end
    local rc = G.GAME.current_round.reroll_cost
    if not (spendable() - rc < 0 and rc ~= 0) then add('reroll') end
    add('end_shop')
  elseif ph == 'pack' then
    for i, c in ipairs(G.pack_cards.cards) do
      if is_consumable(c) then
        for _, m in ipairs(usable_masks(c)) do add(string.format('pick %d %d', i - 1, m)) end
      elseif c.ability.set == 'Joker' then
        if (c.edition and c.edition.negative) or #G.jokers.cards < G.jokers.config.card_limit then
          add(string.format('pick %d 0', i - 1))
        end
      else
        add(string.format('pick %d 0', i - 1))
      end
    end
    local kind = AZB.opened_booster_kind()
    if #G.pack_cards.cards > 0 and (kind == 'Celestial' or kind == 'Standard' or kind == 'Buffoon'
        or hand_n > 0 or G.hand.config.card_limit <= 0) then
      add('skip_pack')
    end
  elseif ph == 'round_eval' then
    add('cash_out')
  end
  for i, c in ipairs(G.consumeables.cards) do
    for _, m in ipairs(usable_masks(c)) do add(string.format('use %d %d', i - 1, m)) end
  end
  for i, c in ipairs(G.jokers.cards) do
    if not c.ability.eternal then add('sell_joker ' .. (i - 1)) end
  end
  for i, c in ipairs(G.consumeables.cards) do
    if not c.ability.eternal then add('sell_consumable ' .. (i - 1)) end
  end
  return out, ph
end
