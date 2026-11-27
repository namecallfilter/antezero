-- gamestate_dataset_cv/src/run_tracker.lua
-- Dedicated telemetry logger tracking exact ground truth gamestates at each action frame.

local RunTracker = {}

RunTracker.enabled = false
RunTracker.log_file = nil
RunTracker.repo_log_file = nil
RunTracker.action_step = 0
RunTracker.run_id = nil
RunTracker._active_pack = nil
RunTracker.undo_stack = {}
RunTracker.log_lines = {}
RunTracker.MAX_UNDO = 50
RunTracker._in_undo = false

local function is_revertable_state()
    if not G or not G.GAME or not G.STATES then return false end
    if G.booster_pack or (G.pack_cards and G.pack_cards.cards and #G.pack_cards.cards > 0) then
        return false
    end
    return G.STATE == G.STATES.SELECTING_HAND
        or G.STATE == G.STATES.SHOP
        or G.STATE == G.STATES.BLIND_SELECT
        or G.STATE == G.STATES.ROUND_EVAL
end

local function capture_revert_savetext()
    if save_run and STR_PACK then
        local prev_no_save = G.F_NO_SAVING
        local prev_fh_run = G.FILE_HANDLER and G.FILE_HANDLER.run
        local prev_fh_q = G.FILE_HANDLER and G.FILE_HANDLER.update_queued
        G.F_NO_SAVING = false
        save_run()
        G.F_NO_SAVING = prev_no_save
        if G.FILE_HANDLER then
            G.FILE_HANDLER.run = prev_fh_run
            G.FILE_HANDLER.update_queued = prev_fh_q
        end
        if G.ARGS and G.ARGS.save_run then
            return { packed = STR_PACK(G.ARGS.save_run) }
        end
    end
    return nil
end

local function rewrite_telemetry_logs()
    local body = (#RunTracker.log_lines > 0) and (table.concat(RunTracker.log_lines, "\n") .. "\n") or ""
    if RunTracker.log_file then
        love.filesystem.write(RunTracker.log_file, body)
    end
    if RunTracker.repo_log_file and NFS and NFS.write then
        NFS.write(RunTracker.repo_log_file, body)
    end
end

local function make_run_id()
    local seed = (G and G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.seed) or "run"
    return string.format("%s_%d", tostring(seed), os.time and os.time() or 0)
end

function RunTracker.reset()
    RunTracker.action_step = 0
    RunTracker._active_pack = nil
    RunTracker.undo_stack = {}
    RunTracker.log_lines = {}
    RunTracker.run_id = make_run_id()
    rewrite_telemetry_logs()
    print("RUN_TRACKER: Reset telemetry log for run_id=" .. tostring(RunTracker.run_id))
end

function RunTracker.init()
    RunTracker.enabled = true
    RunTracker.log_file = "run_telemetry.jsonl"
    if NFS and NFS.write and G.DATASET_MOD and G.DATASET_MOD.mod_path then
        RunTracker.repo_log_file = G.DATASET_MOD.mod_path .. "../run_telemetry.jsonl"
    end
    RunTracker.reset()
    print("RUN_TRACKER: Initialized telemetry logger -> " .. tostring(RunTracker.log_file))
    if RunTracker.repo_log_file then
        print("RUN_TRACKER: Initialized repo mirror -> " .. tostring(RunTracker.repo_log_file))
    end
end

local function find_card_index_0based(area, target)
    if not area or not area.cards or not target then return nil end
    for i, c in ipairs(area.cards) do
        if c == target then
            return i - 1
        end
    end
    return nil
end

local function sanitize_extra(val)
    local t = type(val)
    if t == "number" or t == "string" or t == "boolean" then
        return val
    elseif t == "table" then
        local out = {}
        for k, v in pairs(val) do
            if type(k) == "string" or type(k) == "number" then
                local sv = sanitize_extra(v)
                if sv ~= nil then
                    out[tostring(k)] = sv
                end
            end
        end
        return out
    end
    return nil
end

local function serialize_card(c)
    if not c then return nil end
    local edition_name = "none"
    if c.edition then
        if type(c.edition) == "table" then
            if c.edition.foil or c.edition.type == "foil" then edition_name = "foil"
            elseif c.edition.holo or c.edition.type == "holo" then edition_name = "holographic"
            elseif c.edition.polychrome or c.edition.type == "polychrome" then edition_name = "polychrome"
            elseif c.edition.negative or c.edition.type == "negative" then edition_name = "negative"
            end
        elseif type(c.edition) == "string" then
            local ed_lower = string.lower(c.edition)
            if ed_lower:find("foil") then edition_name = "foil"
            elseif ed_lower:find("holo") then edition_name = "holographic"
            elseif ed_lower:find("poly") then edition_name = "polychrome"
            elseif ed_lower:find("neg") then edition_name = "negative"
            end
        end
    end

    local seal_name = "none"
    if c.seal then
        seal_name = string.lower(c.seal)
    end

    local enhancement_name = "none"
    local center_key = c.config and c.config.center and c.config.center.key or ""
    local ab_name = c.ability and (c.ability.name or c.ability.effect) or ""
    if center_key == "m_bonus" or ab_name:find("Bonus") then enhancement_name = "bonus"
    elseif center_key == "m_mult" or ab_name:find("Mult") then enhancement_name = "mult"
    elseif center_key == "m_wild" or ab_name:find("Wild") then enhancement_name = "wild"
    elseif center_key == "m_glass" or ab_name:find("Glass") then enhancement_name = "glass"
    elseif center_key == "m_steel" or ab_name:find("Steel") then enhancement_name = "steel"
    elseif center_key == "m_stone" or ab_name:find("Stone") then enhancement_name = "stone"
    elseif center_key == "m_gold" or ab_name:find("Gold") then enhancement_name = "gold"
    elseif center_key == "m_lucky" or ab_name:find("Lucky") then enhancement_name = "lucky"
    end

    local card_id = c.playing_card or c.sort_id or c.ID or 0

    return {
        id = card_id,
        sort_id = c.sort_id or 0,
        playing_card = c.playing_card or nil,
        key = c.config and c.config.center and c.config.center.key or nil,
        set = c.ability and c.ability.set or nil,
        name = c.ability and c.ability.name or (c.base and c.base.name) or "Unknown",
        cost = c.cost or 0,
        sell_cost = c.sell_cost or 0,
        extra_value = c.ability and c.ability.extra_value or 0,
        bonus_chips = c.ability and c.ability.perma_bonus or 0,
        edition = edition_name,
        seal = seal_name,
        enhancement = enhancement_name,
        debuffed = (c.debuff == true) or false,
        couponed = (c.ability and c.ability.couponed) or false,
        played_this_ante = (c.ability and c.ability.played_this_ante) or false,
        eternal = (c.ability and c.ability.eternal) or false,
        perishable = (c.ability and c.ability.perishable) or false,
        perish_tally = (c.ability and c.ability.perish_tally) or 5,
        rental = (c.ability and c.ability.rental) or false,
        facing_back = (c.facing == "back") or (c.ability and c.ability.wheel_flipped == true) or false,
        forced_selection = (c.ability and c.ability.forced_selection) or false,
        suit = c.base and c.base.suit or "none",
        value = c.base and c.base.value or "none"
    }
end

function RunTracker.snapshot_gamestate(action_name, action_metadata)
    if not RunTracker.enabled or not G.GAME then return end
    if G.DATASET_MOD and G.DATASET_MOD.config and G.DATASET_MOD.config.mode and G.DATASET_MOD.config.mode ~= "off" then
        return
    end
    if not RunTracker.run_id then
        RunTracker.run_id = make_run_id()
    end

    RunTracker.action_step = RunTracker.action_step + 1

    local jokers = {}
    if G.jokers and G.jokers.cards then
        for i, j in ipairs(G.jokers.cards) do
            local item = serialize_card(j)
            local ab = j.ability or {}
            local ex = type(ab.extra) == "table" and ab.extra or nil
            item.mult = ab.mult or (ex and (ex.mult or ex.s_mult)) or 0
            item.x_mult = ab.x_mult or ab.Xmult or (ex and (ex.x_mult or ex.Xmult)) or 1.0
            item.chips = (ex and (ex.chips or ex.chip_mod)) or ab.chips or (type(ab.extra) == "number" and ab.extra) or 0
            item.h_size = (ex and ex.h_size) or ab.h_size or 0
            item.invis_rounds = ab.invis_rounds or 0
            item.yorick_discards = ab.yorick_discards or nil
            item.loyalty_remaining = ab.loyalty_remaining or nil
            item.payout = (ex and ex.dollars) or ab.payout or nil
            item.to_do_poker_hand = ab.to_do_poker_hand or nil
            item.extra = sanitize_extra(ab.extra)
            table.insert(jokers, item)
        end
    end

    local consumables = {}
    if G.consumeables and G.consumeables.cards then
        for i, c in ipairs(G.consumeables.cards) do
            table.insert(consumables, serialize_card(c))
        end
    end

    local hand_cards = {}
    local selected_cards = {}
    if G.hand and G.hand.cards then
        for i, c in ipairs(G.hand.cards) do
            local sc = serialize_card(c)
            sc.selected = c.highlighted or false
            table.insert(hand_cards, sc)
            if c.highlighted then
                table.insert(selected_cards, sc)
            end
        end
    end

    local shop_cards = {}
    if G.shop_jokers and G.shop_jokers.cards then
        for i, c in ipairs(G.shop_jokers.cards) do
            table.insert(shop_cards, serialize_card(c))
        end
    end

    local shop_vouchers = {}
    if G.shop_vouchers and G.shop_vouchers.cards then
        for i, c in ipairs(G.shop_vouchers.cards) do
            table.insert(shop_vouchers, serialize_card(c))
        end
    end

    local shop_packs = {}
    if G.shop_booster and G.shop_booster.cards then
        for i, c in ipairs(G.shop_booster.cards) do
            local sp = serialize_card(c)
            sp.choose = (c.config and c.config.center and c.config.center.config and c.config.center.config.choose) or 1
            sp.extra = (c.ability and c.ability.extra) or 3
            table.insert(shop_packs, sp)
        end
    end

    local pack_cards = {}
    if G.pack_cards and G.pack_cards.cards then
        for i, c in ipairs(G.pack_cards.cards) do
            table.insert(pack_cards, serialize_card(c))
        end
    end

    local vouchers_redeemed = {}
    if G.GAME.used_vouchers then
        for k, v in pairs(G.GAME.used_vouchers) do
            if v then table.insert(vouchers_redeemed, k) end
        end
    end

    local hand_levels = {}
    local hand_play_counts = {}
    if G.GAME.hands then
        for k, v in pairs(G.GAME.hands) do
            hand_levels[k] = v.level or 1
            hand_play_counts[k] = v.played or 0
        end
    end

    local blind_name = G.GAME.blind and G.GAME.blind.name or "Small Blind"
    local boss_name = G.GAME.round_resets and G.GAME.round_resets.blind_choices and G.GAME.round_resets.blind_choices.Boss or nil

    local full_deck_cards = {}
    if G.playing_cards then
        for i, c in ipairs(G.playing_cards) do
            table.insert(full_deck_cards, serialize_card(c))
        end
    end

    local discard_cards = {}
    if G.discard and G.discard.cards then
        for i, c in ipairs(G.discard.cards) do
            table.insert(discard_cards, serialize_card(c))
        end
    end

    local draw_pile_cards = {}
    if G.deck and G.deck.cards then
        for i, c in ipairs(G.deck.cards) do
            table.insert(draw_pile_cards, serialize_card(c))
        end
    end

    local active_tags = {}
    if G.GAME.tags then
        for i, t in ipairs(G.GAME.tags) do
            table.insert(active_tags, {
                name = t.name or (t.key and string.gsub(t.key, "^tag_", "")) or "Unknown",
                key = t.key or nil,
                tally = t.tally or 1
            })
        end
    end

    local small_tag = G.GAME.round_resets and G.GAME.round_resets.blind_tags and G.GAME.round_resets.blind_tags.Small or nil
    local big_tag = G.GAME.round_resets and G.GAME.round_resets.blind_tags and G.GAME.round_resets.blind_tags.Big or nil

    local loc = "blind"
    if G.booster_pack or (G.pack_cards and G.pack_cards.cards and #G.pack_cards.cards > 0) then
        loc = "pack"
    elseif G.STATE == (G.STATES and G.STATES.SHOP) or (G.shop and not G.shop.alignment.offset.py) then
        loc = "shop"
    elseif G.STATE == (G.STATES and G.STATES.BLIND_SELECT) or G.blind_select then
        loc = "select"
    else
        loc = "blind"
    end

    local consumeable_usage = {}
    if G.GAME.consumeable_usage then
        for k, v in pairs(G.GAME.consumeable_usage) do
            table.insert(consumeable_usage, {key = k, count = v.count or 1, set = v.set or ""})
        end
    end

    local prng_streams = {}
    if G.GAME and G.GAME.pseudorandom then
        for k, v in pairs(G.GAME.pseudorandom) do
            if type(v) == "number" then
                prng_streams[k] = v
            end
        end
    end

    local cr = G.GAME.current_round or {}
    local cur_ante = (G.GAME.round_resets and G.GAME.round_resets.ante) or 1
    local orb_choices = (G.GAME.orbital_choices and G.GAME.orbital_choices[cur_ante]) or {}
    local mods = G.GAME.modifiers or {}
    local last_tp_key = G.GAME.last_tarot_planet
    local last_tp_name = (last_tp_key and G.P_CENTERS and G.P_CENTERS[last_tp_key] and G.P_CENTERS[last_tp_key].name) or nil
    if loc ~= "pack" then
        RunTracker._active_pack = nil
    end

    local joker_order_ids = {}
    for _, j in ipairs(jokers) do
        table.insert(joker_order_ids, j.id)
    end
    local hand_order_ids = {}
    for _, hc in ipairs(hand_cards) do
        table.insert(hand_order_ids, hc.id)
    end
    local meta = action_metadata or {}
    if not meta.joker_order then
        meta.joker_order = joker_order_ids
    end
    if not meta.hand_order then
        meta.hand_order = hand_order_ids
    end

    local payload = {
        run_id = RunTracker.run_id,
        step = RunTracker.action_step,
        timestamp = love.timer.getTime(),
        action = action_name,
        action_metadata = meta,
        state = {
            seed = (G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.seed) or "",
            seeded = (G.GAME and G.GAME.seeded) or false,
            stake = G.GAME.stake or 1,
            deck_name = (G.GAME.selected_back and G.GAME.selected_back.name) or "Red Deck",
            location = loc,
            round_won = (G.STATES and G.STATE == G.STATES.ROUND_EVAL) or false,
            pre_pack_location = (loc == "pack") and ((G.shop and "shop") or "select") or nil,
            active_pack = (loc == "pack") and RunTracker._active_pack or nil,
            money = G.GAME.dollars or 0,
            round_dollars = cr.dollars or 0,
            ante = cur_ante,
            round = G.GAME.round or 1,
            hands_left = cr.hands_left or 0,
            discards_left = cr.discards_left or 0,
            base_hands = (G.GAME and G.GAME.round_resets and G.GAME.round_resets.hands) or 4,
            base_discards = (G.GAME and G.GAME.round_resets and G.GAME.round_resets.discards) or 3,
            hands_played_this_round = cr.hands_played or 0,
            discards_used_this_round = cr.discards_used or 0,
            reroll_cost = cr.reroll_cost or 5,
            free_rerolls = cr.free_rerolls or 0,
            inflation = G.GAME.inflation or 0,
            edition_rate = G.GAME.edition_rate or 1,
            ante_scaling = (G.GAME.starting_params and G.GAME.starting_params.ante_scaling) or 1,
            temp_hand_size_bonus = (G.GAME.round_resets and G.GAME.round_resets.temp_handsize) or 0,
            gros_michel_extinct = (G.GAME.pool_flags and G.GAME.pool_flags.gros_michel_extinct) or false,
            enable_eternals_in_shop = mods.enable_eternals_in_shop or false,
            enable_perishables_in_shop = mods.enable_perishables_in_shop or false,
            enable_rentals_in_shop = mods.enable_rentals_in_shop or false,
            probability_normal = (G.GAME and G.GAME.probabilities and G.GAME.probabilities.normal) or 1,
            blinds_skipped = G.GAME.skips or 0,
            tarots_used = (G.GAME.consumeable_usage_total and G.GAME.consumeable_usage_total.tarot) or 0,
            planets_used = (G.GAME.consumeable_usage_total and G.GAME.consumeable_usage_total.planet) or 0,
            spectrals_used = (G.GAME.consumeable_usage_total and G.GAME.consumeable_usage_total.spectral) or 0,
            unused_discards = G.GAME.unused_discards or 0,
            last_tarot_planet = last_tp_key or nil,
            last_tarot_planet_name = last_tp_name,
            castle_suit = (cr.castle_card and cr.castle_card.suit) or "Spades",
            ancient_suit = (cr.ancient_card and cr.ancient_card.suit) or "Spades",
            mail_card_rank = (cr.mail_card and (cr.mail_card.rank or cr.mail_card.value)) or "Ace",
            idol_card = cr.idol_card and {rank = cr.idol_card.rank or cr.idol_card.value, suit = cr.idol_card.suit} or nil,
            ox_target_hand = cr.most_played_poker_hand or nil,
            locked_hand_type = (G.GAME.blind and G.GAME.blind.only_hand) or nil,
            small_blind_orbital_hand = orb_choices.Small or nil,
            big_blind_orbital_hand = orb_choices.Big or nil,
            target_chips = G.GAME.blind and G.GAME.blind.chips or 0,
            current_chips = G.GAME.chips or 0,
            blind = blind_name,
            boss_blind = boss_name,
            boss_disabled = (G.GAME.blind and G.GAME.blind.disabled == true) or false,
            small_blind_tag = small_tag,
            big_blind_tag = big_tag,
            active_tags = active_tags,
            jokers = jokers,
            joker_slots = (G.jokers and G.jokers.config and G.jokers.config.card_limit) or 5,
            consumables = consumables,
            consumable_slots = (G.consumeables and G.consumeables.config and G.consumeables.config.card_limit) or 2,
            consumeable_usage = consumeable_usage,
            hand = hand_cards,
            hand_max_size = (G.hand and G.hand.config and G.hand.config.card_limit) or 8,
            selected_cards = selected_cards,
            shop = shop_cards,
            shop_vouchers = shop_vouchers,
            shop_packs = shop_packs,
            pack_cards = pack_cards,
            pack_choices_remaining = (G.GAME and G.GAME.pack_choices) or 0,
            vouchers = vouchers_redeemed,
            hand_levels = hand_levels,
            hand_play_counts = hand_play_counts,
            deck_cards_count = (G.playing_cards and #G.playing_cards) or 52,
            deck = full_deck_cards,
            draw_pile = draw_pile_cards,
            discard_pile = discard_cards,
            remaining_deck_count = (G.deck and G.deck.cards and #G.deck.cards) or 0,
            prng_streams = prng_streams
        }
    }

    local json_str = (JSON and JSON.encode) and JSON.encode(payload) or nil
    if not json_str and G.JSON and G.JSON.encode then
        json_str = G.JSON.encode(payload)
    end

    if json_str then
        if is_revertable_state() then
            local savetext = capture_revert_savetext()
            if savetext then
                table.insert(RunTracker.undo_stack, {
                    savetext = savetext,
                    prev_line_count = #RunTracker.log_lines,
                    prev_action_step = RunTracker.action_step - 1,
                    action_name = action_name,
                })
                if #RunTracker.undo_stack > RunTracker.MAX_UNDO then
                    table.remove(RunTracker.undo_stack, 1)
                end
            end
        end
        table.insert(RunTracker.log_lines, json_str)
        love.filesystem.append(RunTracker.log_file, json_str .. "\n")
        if RunTracker.repo_log_file and NFS and NFS.append then
            NFS.append(RunTracker.repo_log_file, json_str .. "\n")
        end
    end
end

function RunTracker.undo_last_action()
    if RunTracker._in_undo then
        return false, "Undo already in progress"
    end
    if not RunTracker.enabled or not G or not G.GAME then
        return false, "No active run"
    end
    if #RunTracker.undo_stack == 0 then
        return false, "No revertable state in undo history"
    end

    local snap = table.remove(RunTracker.undo_stack)
    local actions_reverted = math.max(1, RunTracker.action_step - snap.prev_action_step)

    while #RunTracker.log_lines > snap.prev_line_count do
        table.remove(RunTracker.log_lines)
    end
    rewrite_telemetry_logs()

    RunTracker.action_step = snap.prev_action_step
    RunTracker._active_pack = nil

    local savetext = (snap.savetext and snap.savetext.packed and STR_UNPACK)
        and STR_UNPACK(snap.savetext.packed)
        or snap.savetext

    if G.delete_run and G.start_run then
        if G.E_MANAGER then G.E_MANAGER:clear_queue() end
        if G.screenwipe then
            G.screenwipe:remove()
            G.screenwipe = nil
        end
        RunTracker._in_undo = true
        local ok, err = pcall(function()
            G:delete_run()
            G:start_run({ savetext = savetext })
        end)
        RunTracker._in_undo = false
        if not ok then error(err) end
    elseif G.FUNCS and G.FUNCS.start_run then
        RunTracker._in_undo = true
        G.FUNCS.start_run(nil, { savetext = savetext })
        RunTracker._in_undo = false
    end

    return true, actions_reverted, snap.action_name, snap.prev_action_step
end

-- ========================================================
-- ACTION HOOKS (Intercept exact actions matching templates)
-- ========================================================

function RunTracker.hook_actions()
    -- 0. Reset run telemetry on new run / quick restart (preserve on Continue Run savetext, ignore dataset gen modes)
    if Game and Game.start_run then
        local orig_start_run = Game.start_run
        function Game:start_run(args)
            local is_cv_off = not (G.DATASET_MOD and G.DATASET_MOD.config and G.DATASET_MOD.config.mode and G.DATASET_MOD.config.mode ~= "off")
            local is_fresh_run = not (args and args.savetext)
            local was_undo = RunTracker._in_undo
            local ret = orig_start_run(self, args)
            if was_undo then
                RunTracker._active_pack = nil
            elseif is_fresh_run then
                RunTracker.undo_stack = {}
                if is_cv_off then
                    RunTracker.reset()
                end
            elseif is_cv_off then
                RunTracker.undo_stack = {}
                RunTracker.run_id = RunTracker.run_id or make_run_id()
                RunTracker._active_pack = nil
            end
            return ret
        end
    end

    -- 1. Play Hand
    local orig_play = G.FUNCS.play_cards_from_highlighted
    G.FUNCS.play_cards_from_highlighted = function(e)
        local h_cards = {}
        if G.hand and G.hand.highlighted then
            for _, hc in ipairs(G.hand.highlighted) do
                table.insert(h_cards, serialize_card(hc))
            end
        end
        RunTracker.snapshot_gamestate("Play Hand", {
            selected_cards = h_cards
        })
        return orig_play(e)
    end

    -- 2. Discard
    local orig_discard = G.FUNCS.discard_cards_from_highlighted
    G.FUNCS.discard_cards_from_highlighted = function(e, hook)
        local h_cards = {}
        if G.hand and G.hand.highlighted then
            for _, hc in ipairs(G.hand.highlighted) do
                table.insert(h_cards, serialize_card(hc))
            end
        end
        local meta = { selected_cards = h_cards }
        if hook then meta.hook = true end
        RunTracker.snapshot_gamestate("Discard", meta)
        return orig_discard(e, hook)
    end

    -- 3. Reroll Shop
    local orig_reroll = G.FUNCS.reroll_shop
    G.FUNCS.reroll_shop = function(e)
        RunTracker.snapshot_gamestate("Reroll Shop")
        return orig_reroll(e)
    end

    -- 4. Cash Out
    local orig_cash_out = G.FUNCS.cash_out
    G.FUNCS.cash_out = function(e)
        RunTracker.snapshot_gamestate("Cash Out", {
            include_round_payout = true,
            round_dollars = G.GAME and G.GAME.current_round and G.GAME.current_round.dollars or nil
        })
        return orig_cash_out(e)
    end

    -- 5. Buy Card / Buy and Use
    local orig_buy = G.FUNCS.buy_from_shop
    G.FUNCS.buy_from_shop = function(e)
        local card = e and e.config and e.config.ref_table
        local act_name = (e and e.config and e.config.id == 'buy_and_use') and "Buy and Use" or "Buy Card"
        local idx = find_card_index_0based(G.shop_jokers, card)
        local h_cards = {}
        if G.hand and G.hand.highlighted then
            for _, hc in ipairs(G.hand.highlighted) do
                table.insert(h_cards, serialize_card(hc))
            end
        end
        RunTracker.snapshot_gamestate(act_name, {
            card = card and card.ability and card.ability.name or nil,
            key = card and card.config and card.config.center and card.config.center.key or nil,
            cost = card and card.cost or 0,
            target_idx = idx,
            selected_cards = h_cards
        })
        return orig_buy(e)
    end

    -- 6. Sell Card
    local orig_sell = G.FUNCS.sell_card
    G.FUNCS.sell_card = function(e)
        local card = e and e.config and e.config.ref_table
        local idx = find_card_index_0based(G.jokers, card)
        if idx == nil and G.consumeables then
            local c_idx = find_card_index_0based(G.consumeables, card)
            if c_idx ~= nil then
                idx = ((G.jokers and G.jokers.cards and #G.jokers.cards) or 0) + c_idx
            end
        end
        RunTracker.snapshot_gamestate("Sell Card", {
            card = card and card.ability and card.ability.name or nil,
            key = card and card.config and card.config.center and card.config.center.key or nil,
            sell_cost = card and card.sell_cost or 0,
            target_idx = idx
        })
        return orig_sell(e)
    end

    -- 7. Select Blind
    local orig_select_blind = G.FUNCS.select_blind
    if orig_select_blind then
        G.FUNCS.select_blind = function(e)
            RunTracker.snapshot_gamestate("Select Blind")
            return orig_select_blind(e)
        end
    end

    -- 8. Skip Blind
    local orig_skip_blind = G.FUNCS.skip_blind
    if orig_skip_blind then
        G.FUNCS.skip_blind = function(e)
            RunTracker.snapshot_gamestate("Skip Blind")
            return orig_skip_blind(e)
        end
    end

    -- 9. Use Card (Use Consumable, Open Pack, Redeem Voucher, Use In Pack)
    local orig_use_card = G.FUNCS.use_card
    if orig_use_card then
        G.FUNCS.use_card = function(e, mute, nosave)
            if not (e and e.config and e.config.id == 'buy_and_use') then
                local card = e and e.config and e.config.ref_table
                if card and not (card.check_use and card:check_use()) then
                    local set_name = card.ability and card.ability.set or ""
                    local act_name = "Use Consumable"
                    local idx = nil
                    if set_name == "Booster" then
                        act_name = "Open Pack"
                        idx = find_card_index_0based(G.shop_booster, card)
                        local c_cfg = card.config and card.config.center and card.config.center.config or {}
                        RunTracker._active_pack = {
                            name = (card.ability and card.ability.name) or nil,
                            key = (card.config and card.config.center and card.config.center.key) or nil,
                            choose = c_cfg.choose or 1,
                            extra = (card.ability and card.ability.extra) or c_cfg.extra or 3,
                            cost = card.cost or 0,
                        }
                    elseif set_name == "Voucher" then
                        act_name = "Redeem Voucher"
                        idx = find_card_index_0based(G.shop_vouchers, card)
                    elseif card.area == G.pack_cards then
                        act_name = "Use In Pack"
                        idx = find_card_index_0based(G.pack_cards, card)
                    else
                        act_name = "Use Consumable"
                        idx = find_card_index_0based(G.consumeables, card)
                    end
                    local h_cards = {}
                    if G.hand and G.hand.highlighted then
                        for _, hc in ipairs(G.hand.highlighted) do
                            table.insert(h_cards, serialize_card(hc))
                        end
                    end
                    RunTracker.snapshot_gamestate(act_name, {
                        card = card.ability and card.ability.name or nil,
                        key = card.config and card.config.center and card.config.center.key or nil,
                        set = set_name,
                        cost = card.cost or 0,
                        target_idx = idx,
                        selected_cards = h_cards
                    })
                end
            end
            return orig_use_card(e, mute, nosave)
        end
    end

    -- 10. Skip Booster Pack
    local orig_skip_booster = G.FUNCS.skip_booster
    if orig_skip_booster then
        G.FUNCS.skip_booster = function(e)
            RunTracker.snapshot_gamestate("Skip Pack")
            return orig_skip_booster(e)
        end
    end

    -- 11. Next Round (Leave Shop)
    local orig_toggle_shop = G.FUNCS.toggle_shop
    if orig_toggle_shop then
        G.FUNCS.toggle_shop = function(e)
            RunTracker.snapshot_gamestate("Next Round")
            return orig_toggle_shop(e)
        end
    end

    -- 12. Reroll Boss Blind
    local orig_reroll_boss = G.FUNCS.reroll_boss
    if orig_reroll_boss then
        G.FUNCS.reroll_boss = function(e)
            RunTracker.snapshot_gamestate("Reroll Boss")
            return orig_reroll_boss(e)
        end
    end

    -- 13. Sort by Rank
    local orig_sort_rank = G.FUNCS.sort_hand_value
    if orig_sort_rank then
        G.FUNCS.sort_hand_value = function(e)
            local sel_ids = {}
            if G.hand and G.hand.highlighted then
                for _, c in ipairs(G.hand.highlighted) do
                    table.insert(sel_ids, c.playing_card or c.sort_id or 0)
                end
            end
            RunTracker.snapshot_gamestate("Sort by Rank", {
                sort_type = "rank",
                selected_card_ids = sel_ids
            })
            return orig_sort_rank(e)
        end
    end

    -- 14. Sort by Suit
    local orig_sort_suit = G.FUNCS.sort_hand_suit
    if orig_sort_suit then
        G.FUNCS.sort_hand_suit = function(e)
            local sel_ids = {}
            if G.hand and G.hand.highlighted then
                for _, c in ipairs(G.hand.highlighted) do
                    table.insert(sel_ids, c.playing_card or c.sort_id or 0)
                end
            end
            RunTracker.snapshot_gamestate("Sort by Suit", {
                sort_type = "suit",
                selected_card_ids = sel_ids
            })
            return orig_sort_suit(e)
        end
    end

    print("RUN_TRACKER: All 14 gameplay action hooks installed!")
end

return RunTracker
