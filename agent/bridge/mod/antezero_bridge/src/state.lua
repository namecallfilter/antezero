-- State dump: a port of H.state() from sim/oracle/lua/harness.lua. The
-- top-level keys/shape are identical so `canon::from_lua` (Rust) and
-- antezero_bridge.canon.from_lua (Python) consume it unchanged. Extra
-- information lives under the `az` key, which canon ignores.

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

-- SMODS stores {key, ..., spawn = {[key] = bool}} where vanilla stores the
-- key (and sets it to nil on redeem; SMODS sets spawn[key] = false).
function AZB.vanilla_voucher(v)
  if type(v) ~= 'table' then return v end
  for _, k in ipairs(v) do
    if v.spawn == nil or v.spawn[k] then return k end
  end
  return nil
end

function AZB.state_table()
  local g = G.GAME
  local s = {
    frame = AZB.frame,
    state = AZB.vanilla_state_name(),
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
    voucher = AZB.vanilla_voucher(g.current_round.voucher),
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
    pack_size = g.azb_pack_size or g.pack_size, -- vanilla value (SMODS leaves pack_size alone)
  }
  local function keys_of(t)
    local out = {__array = true}
    for k, v in pairs(t or {}) do if v then out[#out+1] = k end end
    table.sort(out)
    return out
  end
  -- Vanilla (name-based) used_jokers; SMODS's key-based set goes to az.
  s.used_jokers = keys_of(g.azb_used_jokers or g.used_jokers)
  local uv = {__array = true}
  for k, v in pairs(g.used_vouchers) do if v then uv[#uv+1] = k end end
  table.sort(uv)
  s.used_vouchers = uv
  s.tag_info = {__array = true}
  for i, t in ipairs(g.tags) do s.tag_info[i] = {key = t.key, triggered = t.triggered, orbital = t.ability and t.ability.orbital_hand} end
  for k, v in pairs(g.hands) do
    s.hands[k] = {level = v.level, chips = v.chips, mult = v.mult, played = v.played, played_this_round = v.played_this_round, visible = v.visible}
  end
  local extra_rng = {}
  for k, v in pairs(g.pseudorandom) do
    if AZB.smods_only_rng(k) then extra_rng[k] = v else s.rng[k] = v end
  end
  -- Bridge-only extras (ignored by canon::from_lua).
  s.az = {
    raw_state = AZB.state_name(G.STATE),
    booster = SMODS and SMODS.OPENED_BOOSTER and SMODS.OPENED_BOOSTER.config and SMODS.OPENED_BOOSTER.config.center_key or nil,
    seed = g.pseudorandom and g.pseudorandom.seed or nil,
    stake = g.stake,
    back = g.selected_back_key and g.selected_back_key.key or nil,
    settled = AZB.is_settled(),
    paused = G.SETTINGS.paused and true or false,
    smods_used_jokers = keys_of(g.used_jokers),
    smods_rng = extra_rng,
  }
  return s
end

function AZB.state()
  return AZB.json.encode(AZB.state_table())
end
