-- State dump: a port of H.state() from sim/oracle/lua/harness.lua. The
-- top-level keys/shape are identical so `canon::from_lua` (Rust) and
-- antezero_bridge.canon.from_lua (Python) consume it unchanged. Extra
-- information lives under the `az` key, which canon ignores.

-- SMODS sets base.suit_nominal_original to the suit's suit_nominal (0.01 to
-- 0.04 for the vanilla suits), vanilla to a tenth of it.
local VANILLA_SUIT_ORIGINAL = {Diamonds = 0.001, Clubs = 0.002, Hearts = 0.003, Spades = 0.004}
local function vanilla_suit_original(v)
  for k, suit in pairs(SMODS.Suits or {}) do
    if VANILLA_SUIT_ORIGINAL[k] and suit.suit_nominal == v then return VANILLA_SUIT_ORIGINAL[k] end
  end
  return v
end

local function edition_of(c)
  return c.edition and c.edition.type or nil
end

local function card_info(c)
  local t = {
    key = c.config.center_key,
    sort_id = c.sort_id,
    edition = edition_of(c) or AZB.json.null,
    debuff = c.debuff or false,
    cost = c.cost,
    sell_cost = c.sell_cost,
    facing = c.facing,
  }
  if c.ability.set == 'Default' or c.ability.set == 'Enhanced' then
    t.card = c.config.card_key
    t.seal = c.seal or AZB.json.null
    t.rank = c.base.id
    t.suit = c.base.suit
    t.bonus = c.ability.bonus
    t.perma_bonus = c.ability.perma_bonus
    t.times_played = c.base.times_played
    t.suit_nominal_original = vanilla_suit_original(c.base.suit_nominal_original)
  end
  t.base_cost = c.base_cost
  t.extra_cost = c.extra_cost
  t.added_to_deck = c.added_to_deck or false
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
-- key (and sets it to nil on redeem; SMODS sets spawn[key] = false). Several
-- keys pending (another mod's rule) are written as a list, which the
-- importer refuses.
function AZB.vanilla_voucher(v)
  if type(v) ~= 'table' then return v end
  local keys = {__array = true}
  for _, k in ipairs(v) do
    if v.spawn == nil or v.spawn[k] then keys[#keys+1] = k end
  end
  if #keys > 1 then return keys end
  return keys[1]
end

-- The run's rules beyond its deck and stake (the importer refuses any it
-- does not have) and what the profile keeps out of the pools
-- (`get_current_pool`): the cards it has not unlocked, and the tags whose
-- prerequisite it has not discovered.
local function rules_and_profile(s, g)
  local m = {}
  for k, v in pairs(g.modifiers or {}) do
    if type(v) == 'table' then
      for k2, v2 in pairs(v) do m[k .. '.' .. tostring(k2)] = v2 end
    else
      m[k] = v
    end
  end
  s.modifiers = m
  s.challenge = g.challenge or AZB.json.null
  s.vouchers_in_shop = g.starting_params.vouchers_in_shop or AZB.json.null
  s.boosters_in_shop = g.starting_params.boosters_in_shop or AZB.json.null
  local POOLS = {Joker = true, Tarot = true, Planet = true, Spectral = true, Voucher = true}
  s.locked = {__array = true}
  for k, v in pairs(G.P_CENTERS) do
    if v.unlocked == false and POOLS[v.set] then s.locked[#s.locked+1] = k end
  end
  table.sort(s.locked)
  s.tags_locked = {__array = true}
  for k, v in pairs(G.P_TAGS) do
    if v.requires and not (G.P_CENTERS[v.requires] and G.P_CENTERS[v.requires].discovered) then s.tags_locked[#s.tags_locked+1] = k end
  end
  table.sort(s.tags_locked)
end

-- What the game may leave nil, written as null: a dump without the key is
-- then refused.
local NILS = {
  'blind_on_deck', 'blind_selected', 'voucher', 'pack_cards', 'pack_choices', 'pack_interrupt', 'last_hand_played',
  'last_tarot_planet', 'dollar_buffer', 'temp_reroll_cost', 'temp_handsize', 'hand_real_limit', 'joker_real_limit',
  'consumable_real_limit', 'mail', 'consumeable_usage_total',
}
local BLIND_NILS = {'disabled', 'only_hand', 'discards_sub', 'hands_sub'}

local function nulls(t, keys)
  for _, k in ipairs(keys) do
    if t[k] == nil then t[k] = AZB.json.null end
  end
end

-- What a simulator built from the dump needs beyond the compared state
-- (see `sim::dump`), as the harness's H.state adds it.
local function sim_state(s, g)
  local function true_keys(t, keep)
    local out = {__array = true}
    for k, v in pairs(t or {}) do if v and (not keep or keep(v)) then out[#out+1] = k end end
    table.sort(out)
    return out
  end
  s.back = g.selected_back.effect.center.key
  s.stake = g.stake
  s.win_ante = g.win_ante
  s.sort_id = G.sort_id
  s.win_notified = g.win_notified or false
  s.dollar_buffer = g.dollar_buffer
  s.rental_rate = g.rental_rate
  s.perishable_rounds = g.perishable_rounds
  s.joker_rate = g.joker_rate
  s.joker_buffer = g.joker_buffer
  s.consumeable_buffer = g.consumeable_buffer
  s.consumeable_usage_total = g.consumeable_usage_total
  s.planets_used = true_keys(g.consumeable_usage, function(v) return v.set == 'Planet' end)
  s.shop_open = G.shop ~= nil
  s.shop_free = g.shop_free or false
  s.shop_d6ed = g.shop_d6ed or false
  s.first_shop_buffoon = g.first_shop_buffoon or false
  s.reroll_cost_increase = g.current_round.reroll_cost_increase
  s.temp_reroll_cost = g.round_resets.temp_reroll_cost
  s.temp_handsize = g.round_resets.temp_handsize
  s.round_bonus = {next_hands = g.round_bonus.next_hands, discards = g.round_bonus.discards}
  s.used_packs = {__array = true}
  for i, p in ipairs(g.current_round.used_packs or {}) do s.used_packs[i] = p end
  s.pack_interrupt = g.PACK_INTERRUPT and AZB.state_name(g.PACK_INTERRUPT)
  s.facing_blind = g.facing_blind or false
  s.last_blind_boss = (g.last_blind and g.last_blind.boss) or false
  s.bosses_used = g.bosses_used.boss or g.bosses_used -- SMODS keeps the bosses' counts under `boss`
  s.blind_selected = g.round_resets.blind and g.round_resets.blind.name
  s.orbital_choices = {__array = true}
  for ante, c in pairs(g.orbital_choices or {}) do
    s.orbital_choices[#s.orbital_choices+1] = {ante = ante, Small = c.Small or AZB.json.null, Big = c.Big or AZB.json.null, Boss = c.Boss or AZB.json.null}
  end
  table.sort(s.orbital_choices, function(a, b) return a.ante < b.ante end)
  s.idol_card = {rank = g.current_round.idol_card.rank, suit = g.current_round.idol_card.suit, id = g.current_round.idol_card.id or AZB.json.null}
  s.mail_card = {rank = g.current_round.mail_card.rank, id = g.current_round.mail_card.id or AZB.json.null}
  s.hand_real_limit = G.hand and G.hand.config.real_card_limit
  s.joker_real_limit = G.jokers and G.jokers.config.real_card_limit
  s.consumable_real_limit = G.consumeables and G.consumeables.config.real_card_limit
  if g.blind then
    s.blind.mult = g.blind.mult
    s.blind.triggered = g.blind.triggered or false
    s.blind.prepped = g.blind.prepped or false
    s.blind.hands = true_keys(g.blind.hands)
    s.blind.only_hand = g.blind.only_hand or nil
    s.blind.discards_sub = g.blind.discards_sub
    s.blind.hands_sub = g.blind.hands_sub
    nulls(s.blind, BLIND_NILS)
  end
  rules_and_profile(s, g)
  nulls(s, NILS)
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
    pack_size = AZB.pack_size or g.pack_size, -- vanilla value (SMODS leaves pack_size alone)
  }
  local function keys_of(t)
    local out = {__array = true}
    for k, v in pairs(t or {}) do if v then out[#out+1] = k end end
    table.sort(out)
    return out
  end
  -- Vanilla (name-based) used_jokers in a bridge session (see core.lua),
  -- else the game's own; SMODS's key-based set also goes to az.
  s.used_jokers = keys_of(AZB.is_bridge() and g.azb_used_jokers or g.used_jokers)
  local uv = {__array = true}
  for k, v in pairs(g.used_vouchers) do if v then uv[#uv+1] = k end end
  table.sort(uv)
  s.used_vouchers = uv
  s.tag_info = {__array = true}
  for i, t in ipairs(g.tags) do s.tag_info[i] = {key = t.key, triggered = t.triggered, orbital = t.ability and t.ability.orbital_hand or AZB.json.null} end
  for k, v in pairs(g.hands) do
    s.hands[k] = {level = v.level, chips = v.chips, mult = v.mult, played = v.played, played_this_round = v.played_this_round, visible = v.visible}
  end
  sim_state(s, g)
  local extra_rng = {}
  for k, v in pairs(g.pseudorandom) do
    if AZB.smods_only_rng(k) then extra_rng[k] = v else s.rng[k] = v end
  end
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
