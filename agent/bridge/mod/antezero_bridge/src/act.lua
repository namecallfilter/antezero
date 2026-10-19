-- Runs, actions and injections: ports of H.start / H.act / H.inject from
-- sim/oracle/lua/harness.lua (same game entry points, same arguments, same
-- timing protocol), with light precondition checks so that a bad request
-- errors instead of wedging the game. Indices in action tables are 1-based,
-- exactly like the harness.
--
-- Timing protocol (harness): for play/discard/use/buy_and_use/pick with a
-- `cards` field, highlight the cards and settle first ("as a player would
-- before pressing a button"); then rebase G.TIMERS.TOTAL to 0 and press the
-- button; then settle. `use` without `cards` unhighlights, settles and
-- rebases again before using.

local function fail(msg) error({azb = true, msg = msg}, 0) end

local function area_card(area, i)
  local c = area and area.cards and area.cards[i]
  if not c then fail('no card ' .. tostring(i)) end
  return c
end

local function highlight(indices)
  G.hand:unhighlight_all()
  for _, i in ipairs(indices or {}) do
    local c = area_card(G.hand, i)
    if not c.highlighted then G.hand:add_to_highlighted(c) end
  end
end

function AZB.start(seed, back_key, stake)
  back_key = back_key or 'b_red'
  if not G.P_CENTERS[back_key] then fail('unknown back ' .. tostring(back_key)) end
  AZB.unlock_all()
  G.SETTINGS.tutorial_complete = true
  G.SETTINGS.tutorial_progress = nil
  G.GAME.viewed_back = Back(G.P_CENTERS[back_key])
  G:delete_run()
  G.SETTINGS.paused = false
  AZB.rebase()
  G.sort_id = 0
  G.playing_card = 0
  G:start_run({stake = stake or 1, seed = seed})
  G.GAME.viewed_back = Back(G.P_CENTERS[back_key])
  AZB.mark_action()
end

local S -- G.STATES, resolved lazily
local function in_states(list)
  for _, s in ipairs(list) do
    if s == 'pack' then
      if AZB.is_pack_state() then return true end
    elseif G.STATE == S[s] then return true end
  end
  return false
end

local REQUIRES = {
  select_blind = {'BLIND_SELECT'}, skip_blind = {'BLIND_SELECT'}, reroll_boss = {'BLIND_SELECT'},
  play = {'SELECTING_HAND'}, discard = {'SELECTING_HAND'},
  buy = {'SHOP'}, buy_and_use = {'SHOP'}, redeem = {'SHOP'}, open = {'SHOP'}, reroll = {'SHOP'}, end_shop = {'SHOP'},
  cash_out = {'ROUND_EVAL'},
  pick = {'pack'}, skip_pack = {'pack'},
  move_hand = {'SELECTING_HAND', 'pack'}, sort_hand = {'SELECTING_HAND', 'pack'},
}
local KNOWN = {
  use = true, sell_joker = true, sell_consumable = true, move_joker = true,
}
local SELECTS = {play = true, discard = true, use = true, buy_and_use = true, pick = true}

-- Phase 1: validate and (if needed) select cards. Returns true when the
-- game must settle before phase 2.
function AZB.act_prepare(a)
  S = G.STATES
  local k = a.type
  if G.STATE == S.GAME_OVER then fail('game over') end
  if not (REQUIRES[k] or KNOWN[k]) then fail('unknown action ' .. tostring(k)) end
  if REQUIRES[k] and not in_states(REQUIRES[k]) then
    fail(k .. ' not allowed in state ' .. AZB.vanilla_state_name())
  end
  if (k == 'play' or k == 'discard') and #(a.cards or {}) == 0 then fail('no cards') end
  if k == 'skip_blind' and G.GAME.blind_on_deck ~= 'Small' and G.GAME.blind_on_deck ~= 'Big' then
    fail('cannot skip ' .. tostring(G.GAME.blind_on_deck))
  end
  -- Index checks up front (nothing has been touched yet).
  if k == 'use' or k == 'sell_consumable' then area_card(G.consumeables, a.index)
  elseif k == 'sell_joker' then area_card(G.jokers, a.index)
  elseif k == 'buy' or k == 'buy_and_use' then area_card(G.shop_jokers, a.index)
  elseif k == 'redeem' then area_card(G.shop_vouchers, a.index)
  elseif k == 'open' then area_card(G.shop_booster, a.index)
  elseif k == 'pick' then area_card(G.pack_cards, a.index)
  elseif k == 'move_joker' then area_card(G.jokers, a.from); area_card(G.jokers, a.to)
  elseif k == 'move_hand' then area_card(G.hand, a.from); area_card(G.hand, a.to)
  end
  if a.cards and SELECTS[k] then
    for _, i in ipairs(a.cards) do area_card(G.hand, i) end
    highlight(a.cards)
    AZB.mark_action()
    return true
  end
  return false
end

-- `use` without a `cards` field needs an extra unhighlight + settle.
function AZB.act_needs_unhighlight(a)
  return a.type == 'use' and not a.cards
end

function AZB.act_unhighlight()
  AZB.rebase()
  G.hand:unhighlight_all()
  AZB.mark_action()
end

-- Phase 2: rebase and press the button.
function AZB.act_main(a)
  local k = a.type
  AZB.rebase()
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
    fail('unknown action ' .. tostring(k))
  end
  AZB.mark_action()
end

-- Scenario injection (mirrors H.inject / sim/src/inject.rs).
function AZB.inject(t)
  if G.STATE == G.STATES.GAME_OVER then fail('game over') end
  if (t.kind == 'joker' or t.kind == 'consumable' or t.kind == 'voucher') and not G.P_CENTERS[t.key] then
    fail('unknown key ' .. tostring(t.key))
  end
  if t.kind == 'card' and not G.P_CARDS[t.key] then fail('unknown card ' .. tostring(t.key)) end
  AZB.rebase()
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
  else
    fail('unknown inject kind ' .. tostring(t.kind))
  end
  AZB.mark_action()
end
