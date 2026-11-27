-- gamestate_dataset_cv/src/ui.lua
-- Contains SMODS settings page (config_tab), in-game calibration overlay, and keypressed handlers.

local UI = {}

function G.DATASET_MOD:get_selected_object()
    local t = self.selected_target or 'roi_joker_bar'
    if t == 'roi_joker_bar' then return self.config.rois.joker_bar, 'roi', "ROI: Joker Bar (Top)"
    elseif t == 'roi_shop' then return self.config.rois.shop, 'roi', "ROI: Shop Items (Jokers/Cards)"
    elseif t == 'roi_voucher' then return self.config.rois.voucher, 'roi', "ROI: Voucher Shelf"
    elseif t == 'roi_booster' then return self.config.rois.booster, 'roi', "ROI: Booster Packs (Shelf)"
    elseif t == 'roi_pack' then return self.config.rois.pack, 'roi', "ROI: Booster Opening Screen"
    elseif t == 'roi_consumeables' then return self.config.rois.consumeables, 'roi', "ROI: Consumables (Top Right)"
    elseif t == 'roi_card_hand' then return self.config.rois.card_hand, 'roi', "ROI: Card Hand"
    elseif t == 'roi_pack_hand' then return self.config.rois.pack_hand, 'roi', "ROI: Pack Playing Card Hand"
    elseif t == 'roi_play_area' then return self.config.rois.play_area, 'roi', "ROI: Play Area (G.play)"
    elseif t == 'roi_hud_round_score' then return self.config.rois.hud_round_score, 'roi', "ROI: HUD Round Score"
    elseif t == 'roi_hud_money' then return self.config.rois.hud_money, 'roi', "ROI: HUD Money ($)"
    elseif t == 'roi_hud_hands' then return self.config.rois.hud_hands, 'roi', "ROI: HUD Hands"
    elseif t == 'roi_hud_discards' then return self.config.rois.hud_discards, 'roi', "ROI: HUD Discards"
    elseif t == 'roi_hud_ante' then return self.config.rois.hud_ante, 'roi', "ROI: HUD Ante"
    elseif t == 'roi_hud_round' then return self.config.rois.hud_round, 'roi', "ROI: HUD Round"
    elseif t == 'roi_hud_reroll_cost' then return self.config.rois.hud_reroll_cost, 'roi', "ROI: Shop Reroll Cost"
    elseif t == 'roi_hud_blind_target' then return self.config.rois.hud_blind_target, 'roi', "ROI: HUD Blind Target"
    elseif t == 'roi_seed_win' then return self.config.rois.seed_win, 'roi', "ROI: Seed (Win Modal)"
    elseif t == 'roi_seed_game_over' then return self.config.rois.seed_game_over, 'roi', "ROI: Seed (Game Over Modal)"
    elseif t == 'roi_view_deck' then return self.config.rois.view_deck, 'roi', "ROI: View Deck Region"
    elseif t == 'box_joker_bar' then return self.config.boxes.joker_bar_card, 'box', "Box: Joker Bar Card"
    elseif t == 'box_item' then return self.config.boxes.item_card, 'box', "Box: Shop/Pack Item"
    elseif t == 'box_consumeable' then return self.config.boxes.consumeable_card, 'box', "Box: Consumable Card"
    elseif t == 'box_voucher' then return self.config.boxes.voucher_card, 'box', "Box: Voucher Card"
    elseif t == 'box_booster' then return self.config.boxes.booster_card, 'box', "Box: Booster Pack"
    elseif t == 'box_card' then return self.config.boxes.card, 'box', "Box: Hand Card"
    elseif t == 'box_deck_card' then return self.config.boxes.deck_card, 'box', "Box: Deck Card Left Edge"
    end
    return nil, nil, "None"
end

-- Helper: Extract and sort all Joker centers with names and rarities
function UI.get_all_joker_entries()
    local entries = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == 'Joker' and k ~= 'j_undiscovered' and k ~= 'j_locked' then
                local display_name = (v.name and v.name ~= "") and v.name or k
                if G.localization and G.localization.descriptions and G.localization.descriptions.Joker and G.localization.descriptions.Joker[k] then
                    local loc_entry = G.localization.descriptions.Joker[k]
                    if loc_entry.name then
                        if type(loc_entry.name) == "table" then
                            display_name = table.concat(loc_entry.name, " ")
                        else
                            display_name = tostring(loc_entry.name)
                        end
                    end
                elseif type(localize) == "function" then
                    local ok, loc_name = pcall(localize, {type = 'name_text', set = 'Joker', key = k})
                    if ok and loc_name and loc_name ~= "" then
                        display_name = loc_name
                    end
                end
                table.insert(entries, {
                    key = k,
                    name = display_name,
                    rarity = v.rarity or 1,
                    center = v
                })
            end
        end
        table.sort(entries, function(a, b)
            return (a.name or a.key):lower() < (b.name or b.key):lower()
        end)
    end
    return entries
end

function UI.get_filtered_joker_entries(mod)
    local all_entries = UI.get_all_joker_entries()
    local r_filter = mod.test_joker_rarity_filter or "All"
    local search_q = (mod.test_joker_search and tostring(mod.test_joker_search) or ""):lower():match("^%s*(.-)%s*$")

    local target_r = nil
    if r_filter == "Common" then target_r = 1
    elseif r_filter == "Uncommon" then target_r = 2
    elseif r_filter == "Rare" then target_r = 3
    elseif r_filter == "Legendary" then target_r = 4
    end

    if not target_r and search_q == "" then
        return all_entries
    end

    local filtered = {}
    for _, ent in ipairs(all_entries) do
        local rarity_ok = (not target_r) or (ent.rarity == target_r)
        local search_ok = (search_q == "")
            or (ent.name and ent.name:lower():find(search_q, 1, true) ~= nil)
            or (ent.key and ent.key:lower():find(search_q, 1, true) ~= nil)
        if rarity_ok and search_ok then
            table.insert(filtered, ent)
        end
    end
    return filtered
end

function UI.spawn_test_joker(mod, entry_override)
    local entries = mod.test_joker_list or UI.get_filtered_joker_entries(mod)
    local idx = mod.test_joker_idx or 1
    local selected = entry_override or entries[idx]
    if not selected or not selected.key then
        mod.capture_flash = 1.5
        mod.capture_flash_text = "NO JOKER SELECTED!"
        return
    end

    if not G.jokers then
        mod.capture_flash = 2.0
        mod.capture_flash_text = "NO ACTIVE RUN DETECTED! START A RUN FIRST."
        print("GDCV_JOKER_SPAWN: Failed - G.jokers is nil (not in a run)")
        return
    end

    -- Create card safely in G.jokers area
    local target_key = selected.key
    local card = create_card('Joker', G.jokers, nil, nil, nil, nil, target_key, 'deck')
    if not card then
        mod.capture_flash = 2.0
        mod.capture_flash_text = "FAILED TO CREATE CARD: " .. tostring(target_key)
        return
    end

    -- Apply Edition
    local ed_choice = mod.test_joker_edition or "None"
    if ed_choice == "Foil" then
        card:set_edition({foil = true}, true, true)
    elseif ed_choice == "Holo" then
        card:set_edition({holo = true}, true, true)
    elseif ed_choice == "Polychrome" then
        card:set_edition({polychrome = true}, true, true)
    elseif ed_choice == "Negative" then
        card:set_edition({negative = true}, true, true)
    end

    -- Apply Sticker
    local st_choice = mod.test_joker_sticker or "None"
    if st_choice == "Eternal" and card.set_eternal then
        card:set_eternal(true)
    elseif st_choice == "Perishable" and card.set_perishable then
        card:set_perishable(true)
    elseif st_choice == "Rental" and card.set_rental then
        card:set_rental(true)
    end

    -- Register with active deck and board
    card:add_to_deck()
    G.jokers:emplace(card)

    -- Extra safety: expand card limit if slot capacity reached so test jokers always fit
    if G.jokers.config and G.jokers.config.card_limit and #G.jokers.cards > G.jokers.config.card_limit then
        G.jokers.config.card_limit = #G.jokers.cards
    end

    if play_sound then
        pcall(play_sound, 'card1')
    end
    if card.juice_up then
        pcall(function() card:juice_up(0.4, 0.4) end)
    end

    mod.capture_flash = 1.8
    mod.capture_flash_text = string.format("SPAWNED: %s (%s)", selected.name, target_key)
    print(string.format("GDCV_JOKER_SPAWN: Successfully spawned %s [%s] (Edition=%s, Sticker=%s)",
        selected.name, target_key, ed_choice, st_choice))
end

function UI.clear_all_jokers(mod)
    if not G.jokers or not G.jokers.cards or #G.jokers.cards == 0 then
        mod.capture_flash = 1.5
        mod.capture_flash_text = "NO JOKERS TO CLEAR!"
        return
    end

    local count = #G.jokers.cards
    for i = count, 1, -1 do
        local c = G.jokers.cards[i]
        if c then
            if c.remove_from_deck then
                pcall(function() c:remove_from_deck() end)
            end
            G.jokers:remove_card(c)
            if c.remove then
                pcall(function() c:remove() end)
            end
        end
    end

    mod.capture_flash = 1.8
    mod.capture_flash_text = string.format("REMOVED %d JOKERS FROM BOARD!", count)
    print(string.format("GDCV_JOKER_SPAWN: Cleared %d jokers from active joker bar", count))
end

function UI.setup_custom_test_suite(mod)
    if not G.jokers or not G.consumeables or not G.playing_cards then
        mod.capture_flash = 2.0
        mod.capture_flash_text = "START A RUN FIRST TO POPULATE TEST SUITE!"
        print("GDCV_TEST_SUITE: Failed - Not in an active run")
        return
    end

    -- 1. All custom Jokers for which the Python engine tracks state / hooks
    local custom_joker_keys = {
        -- Rotating targets
        "j_castle", "j_ancient", "j_mail", "j_idol", "j_todo_list",
        -- Scaling mult
        "j_green_joker", "j_ride_the_bus", "j_flash", "j_red_card",
        "j_trousers", "j_popcorn", "j_ceremonial",
        -- Scaling x_mult
        "j_campfire", "j_constellation", "j_hologram", "j_vampire", "j_obelisk",
        "j_yorick", "j_hit_the_road", "j_caino", "j_glass", "j_lucky_cat", "j_madness",
        -- Scaling chips & count-dependent chips
        "j_runner", "j_square", "j_wee", "j_ice_cream", "j_stone", "j_bull",
        "j_blue_joker", "j_stuntman", "j_steel_joker", "j_erosion",
        -- Sell value
        "j_egg", "j_gift", "j_swashbuckler",
        -- Spawners & card generation
        "j_riff_raff", "j_cartomancer", "j_vagabond", "j_sixth_sense", "j_seance",
        "j_superposition", "j_certificate", "j_marble", "j_hallucination", "j_8_ball",
        "j_dna", "j_trading", "j_perkeo",
        -- Mechanics & rules modifiers
        "j_midas_mask", "j_pareidolia", "j_oops", "j_four_fingers", "j_shortcut",
        "j_smeared", "j_splash", "j_mime", "j_blueprint", "j_brainstorm", "j_hack",
        "j_hanging_chad", "j_dusk", "j_sock_and_buskin", "j_burnt", "j_selzer",
        "j_turtle_bean", "j_ramen", "j_invisible", "j_burglar", "j_luchador",
        "j_mr_bones", "j_space", "j_business", "j_reserved_parking",
        "j_gros_michel", "j_cavendish", "j_cloud_9", "j_rocket", "j_satellite",
        "j_fortune_teller", "j_supernova", "j_throwback", "j_loyalty_card",
        "j_card_sharp", "j_to_the_moon", "j_astronomer", "j_diet_cola", "j_hiker",
        "j_photograph"
    }

    local owned_joker_keys = {}
    for _, j in ipairs(G.jokers.cards) do
        local k = j.config and j.config.center and j.config.center.key
        if k then owned_joker_keys[k] = true end
    end

    local jokers_spawned = 0
    for _, k in ipairs(custom_joker_keys) do
        if not owned_joker_keys[k] and G.P_CENTERS[k] then
            local card = create_card('Joker', G.jokers, nil, nil, true, nil, k, 'deck')
            if card then
                card:set_edition({negative = true}, true, true)
                card:add_to_deck()
                G.jokers:emplace(card)
                jokers_spawned = jokers_spawned + 1
            end
        end
    end
    G.jokers.config.card_limit = math.max(G.jokers.config.card_limit or 5, #G.jokers.cards)

    -- Ensure Ceremonial Dagger is moved to the far right so it doesn't destroy other Jokers on blind select
    local dagger_idx = nil
    if G.jokers and G.jokers.cards then
        for idx, j in ipairs(G.jokers.cards) do
            local k = j.config and j.config.center and j.config.center.key
            if k == "j_ceremonial" or (j.ability and j.ability.name == "Ceremonial Dagger") then
                dagger_idx = idx
                break
            end
        end
        if dagger_idx and dagger_idx < #G.jokers.cards then
            local dagger_card = table.remove(G.jokers.cards, dagger_idx)
            table.insert(G.jokers.cards, dagger_card)
            if G.jokers.set_ranks then G.jokers:set_ranks() end
            if G.jokers.align_cards then G.jokers:align_cards() end
        end
    end

    -- 2. Convert half of all cards in deck to Lucky Cards, half to Glass Cards
    local total_cards = #G.playing_cards
    local half_cards = math.floor(total_cards / 2)
    for i, c in ipairs(G.playing_cards) do
        if i <= half_cards then
            c:set_ability(G.P_CENTERS.m_lucky)
        else
            c:set_ability(G.P_CENTERS.m_glass)
        end
    end

    -- 3. Give Negative copies of all custom consumables (deck destruction / card manipulation)
    local custom_consumables = {
        -- Tarots
        "c_hanged_man", "c_death", "c_strength", "c_magician", "c_empress",
        "c_heirophant", "c_lovers", "c_chariot", "c_justice", "c_tower",
        "c_devil", "c_sun", "c_moon", "c_star", "c_world", "c_wheel_of_fortune",
        "c_fool", "c_hermit", "c_temperance",
        -- Spectrals
        "c_familiar", "c_grim", "c_incantation", "c_immolate", "c_cryptid",
        "c_ouija", "c_sigil", "c_aura", "c_deja_vu", "c_trance", "c_medium",
        "c_talisman", "c_ankh", "c_hex", "c_ectoplasm"
    }

    local owned_consumable_keys = {}
    for _, c in ipairs(G.consumeables.cards) do
        local k = c.config and c.config.center and c.config.center.key
        if k then owned_consumable_keys[k] = true end
    end

    local consumables_spawned = 0
    for _, k in ipairs(custom_consumables) do
        if not owned_consumable_keys[k] and G.P_CENTERS[k] then
            local c_set = (G.P_CENTERS[k] and G.P_CENTERS[k].set) or "Tarot"
            local card = create_card(c_set, G.consumeables, nil, nil, true, nil, k, 'deck')
            if card then
                card:set_edition({negative = true}, true, true)
                card:add_to_deck()
                G.consumeables:emplace(card)
                consumables_spawned = consumables_spawned + 1
            end
        end
    end
    G.consumeables.config.card_limit = math.max(G.consumeables.config.card_limit or 2, #G.consumeables.cards)

    if play_sound then pcall(play_sound, 'card1') end
    mod.capture_flash = 2.5
    mod.capture_flash_text = string.format("TEST SUITE: +%d JOKERS, +%d CONSUMABLES, %d CARDS 50/50 LUCKY/GLASS!",
        jokers_spawned, consumables_spawned, total_cards)
    print(string.format("GDCV_TEST_SUITE: Spawned %d Jokers (Negative), %d Consumables (Negative), Ceremonial Dagger on far right, converted %d deck cards (half Lucky / half Glass)",
        jokers_spawned, consumables_spawned, total_cards))

    if G.DATASET_MOD and G.DATASET_MOD.RunTracker and G.DATASET_MOD.RunTracker.snapshot_gamestate then
        G.DATASET_MOD.RunTracker.snapshot_gamestate("Setup Test Suite", {
            jokers_spawned = jokers_spawned,
            consumables_spawned = consumables_spawned,
            total_cards = total_cards
        })
    end
end

-- ==========================================
-- SMODS CONFIG TAB & MENU CALLBACKS
-- ==========================================
function UI.init_callbacks(mod)
    G.FUNCS = G.FUNCS or {}

    G.FUNCS.gdcv_change_mode = function(args)
        local mode_map = {
            ['Off'] = 'off',
            ['Card Classifier'] = 'card_classifier',
            ['Shop Classifier'] = 'shop_classifier',
            ['End Round Classifier'] = 'end_round_classifier',
            ['HUD OCR'] = 'hud_ocr',
            ['Seed OCR'] = 'seed_ocr',
            ['Deck Classifier'] = 'deck_classifier',
        }
        local chosen = mode_map[args.to_val] or 'off'
        mod:set_mode(chosen)
    end

    G.FUNCS.gdcv_change_calib_state = function(args)
        local state_map = {
            ['Setup Mode'] = 'setup',
            ['Generation Mode'] = 'generation'
        }
        mod.config.calibration_mode = state_map[args.to_val] or 'setup'
        mod.state = 'idle'
        if mod.config.calibration_mode ~= 'generation' then
            if mod.restore_joker_display_config then mod:restore_joker_display_config() end
            if mod.cleanup_seed_overlay then mod:cleanup_seed_overlay() end
            if mod.cleanup_deck_overlay and mod.config.mode == 'deck_classifier' then mod:cleanup_deck_overlay(true) end
            if mod.EndRoundCapture and mod._active_end_round_clip then mod.EndRoundCapture.stop(mod) end
        end
    end

    G.FUNCS.gdcv_change_max_frames = function(args)
        local frames = tonumber(args.to_val) or 10000
        mod.config.max_frames = frames
    end

    G.FUNCS.gdcv_change_max_clips = function(args)
        mod.config.max_clips = tonumber(args.to_val) or 1000
    end

    G.FUNCS.gdcv_change_er_batch = function(args)
        local n = tonumber(tostring(args.to_val or '2'):match('^(%d+)')) or 2
        mod.config.end_round_frames_per_tick = n
    end

    G.FUNCS.gdcv_change_perkeo_max = function(args)
        local max_c = tonumber(args.to_val) or 10
        mod.config.perkeo_max_consumables = max_c
    end

    G.FUNCS.gdcv_change_card_selection_chance = function(args)
        mod.config.card_selection_chance = args.to_val or '50%'
        print("DATASET: Card selection chance set to " .. tostring(mod.config.card_selection_chance))
    end

    G.FUNCS.gdcv_change_frame_wait_time = function(args)
        local secs = tonumber((tostring(args.to_val or '0.08s'):gsub('s$', ''))) or 0.08
        mod.config.frame_wait_time = secs
        print(string.format("DATASET: Frame wait time set to %.2fs", secs))
    end

    G.FUNCS.gdcv_save_config = function(e)
        mod:save_config()
        mod.capture_flash = 1.5
        mod.capture_flash_text = "CONFIG SAVED TO BOXES.LUA!"
    end

    G.FUNCS.gdcv_load_config = function(e)
        mod:load_config(true)
        mod.capture_flash = 1.5
        mod.capture_flash_text = "CONFIG LOADED FROM BOXES.LUA!"
    end

    G.FUNCS.gdcv_clear_dataset = function(e)
        if not mod.clear_confirm then
            mod.clear_confirm = true
            mod.clear_confirm_timer = 5.0
            mod.clear_button_text = "CONFIRM CLEAR?"
            mod.capture_flash = 1.0
            mod.capture_flash_text = "CLICK AGAIN WITHIN 5s TO ERASE ALL FRAMES!"
        else
            mod.clear_confirm = false
            mod.clear_confirm_timer = 0
            local count = mod:clear_all_dataset_data()
            mod.clear_button_text = "DATASET CLEARED!"
            mod.clear_reset_timer = 2.5
            mod.capture_flash = 2.0
            mod.capture_flash_text = string.format("CLEARED %d DATASET FILES!", count)
        end
    end

    G.FUNCS.gdcv_reload_mod = function(e)
        if mod.reload then
            mod:reload()
        else
            local ok, err = pcall(function()
                local function load_sub(rel)
                    if SMODS and SMODS.load_file then
                        local c = SMODS.load_file(rel, "gamestate_dataset_cv") or SMODS.load_file(rel)
                        if c then return c() end
                    end
                    if love.filesystem.getInfo("Mods/gamestate_dataset_cv/" .. rel) then
                        local content = love.filesystem.read("Mods/gamestate_dataset_cv/" .. rel)
                        local c = (loadstring or load)(content)
                        if c then return c() end
                    end
                end
                load_sub("src/config.lua")
                load_sub("src/cards_gen.lua")
                local shop_m = load_sub("src/shop_gen.lua")
                if shop_m then mod.ShopGen = shop_m end
                load_sub("src/capture.lua")
                local hud_m = load_sub("src/hud_gen.lua")
                if hud_m then mod.HudGen = hud_m end
                local seed_m = load_sub("src/seed_gen.lua")
                if seed_m then mod.SeedGen = seed_m end
                local deck_m = load_sub("src/deck_gen.lua")
                if deck_m then mod.DeckGen = deck_m end
                local er_gen = load_sub("src/end_round_gen.lua")
                if er_gen then mod.EndRoundGen = er_gen end
                if mod.EndRoundCapture then
                    if mod._active_end_round_clip and mod.EndRoundCapture.stop then
                        mod.EndRoundCapture.stop(mod)
                    end
                    if mod.EndRoundCapture.release_gpu_resources then
                        mod.EndRoundCapture.release_gpu_resources()
                    end
                end
                local er_cap = load_sub("src/end_round_capture.lua")
                if er_cap then mod.EndRoundCapture = er_cap end
                local ui_m = load_sub("src/ui.lua")
                if ui_m then
                    ui_m.init_callbacks(mod)
                    if SMODS and SMODS.current_mod then SMODS.current_mod.config_tab = ui_m.create_config_tab end
                    if SMODS and SMODS.Mods and SMODS.Mods["gamestate_dataset_cv"] then
                        SMODS.Mods["gamestate_dataset_cv"].config_tab = ui_m.create_config_tab
                    end
                end
            end)
            if ok then
                mod.capture_flash = 1.5
                mod.capture_flash_text = "MOD HOT-RELOADED!"
            else
                mod.capture_flash = 2.0
                mod.capture_flash_text = "RELOAD FAILED: " .. tostring(err):sub(1, 40)
            end
        end
    end

    G.FUNCS.gdcv_restart_game = function(e)
        if SMODS and SMODS.restart_game then
            SMODS.restart_game()
        else
            love.event.quit("restart")
        end
    end

    -- ==========================================
    -- PAGE & JOKER SPAWNER CALLBACKS
    -- ==========================================
    G.FUNCS.gdcv_switch_page = function(e)
        local target_page = e and e.config and e.config.ref_table and e.config.ref_table.page
        if not target_page then return end
        if G.CONTROLLER and G.CONTROLLER.text_input_hook then
            G.CONTROLLER.text_input_hook = nil
        end
        mod.config_page = target_page
        UI.refresh_page_holder(mod, e.UIBox)
    end

    G.FUNCS.gdcv_change_joker_rarity = function(args)
        local r_name = args.to_val or "All"
        mod.test_joker_rarity_filter = r_name
        mod.test_joker_page = 1
        UI.refresh_joker_grid(mod)
    end

    G.FUNCS.gdcv_apply_joker_search = function(e)
        if G.CONTROLLER and G.CONTROLLER.text_input_hook and G.FUNCS.text_input_key then
            pcall(G.FUNCS.text_input_key, {key = 'return'})
        else
            mod.test_joker_page = 1
            UI.refresh_joker_grid(mod, e and e.UIBox)
        end
    end

    G.FUNCS.gdcv_clear_joker_search = function(e)
        if G.CONTROLLER and G.CONTROLLER.text_input_hook then
            G.CONTROLLER.text_input_hook = nil
        end
        mod.test_joker_search = ""
        mod.test_joker_page = 1
        UI.refresh_page_holder(mod, e and e.UIBox)
    end

    G.FUNCS.gdcv_joker_page_prev = function(e)
        local total_pages = math.max(1, mod.test_joker_total_pages or 1)
        local cur = mod.test_joker_page or 1
        mod.test_joker_page = (cur > 1) and (cur - 1) or total_pages
        UI.refresh_joker_grid(mod, e and e.UIBox)
    end

    G.FUNCS.gdcv_joker_page_next = function(e)
        local total_pages = math.max(1, mod.test_joker_total_pages or 1)
        local cur = mod.test_joker_page or 1
        mod.test_joker_page = (cur < total_pages) and (cur + 1) or 1
        UI.refresh_joker_grid(mod, e and e.UIBox)
    end

    G.FUNCS.gdcv_spawn_grid_joker = function(e)
        local ent = e and e.config and e.config.ref_table
        if ent and ent.key then
            UI.spawn_test_joker(mod, ent)
        end
    end

    G.FUNCS.gdcv_change_joker_selection = function(args)
        local idx = args.to_key
        if not idx then
            local entries = mod.test_joker_list or UI.get_filtered_joker_entries(mod)
            for i, ent in ipairs(entries) do
                if ent.name == args.to_val then
                    idx = i
                    break
                end
            end
        end
        idx = idx or 1
        mod.test_joker_idx = idx
        local entries = mod.test_joker_list or UI.get_filtered_joker_entries(mod)
        local sel = entries[idx]
        mod.test_joker_name_display = sel and sel.name or "None"
        mod.test_joker_key_display = sel and ("[" .. sel.key .. "]") or "[-]"
    end

    G.FUNCS.gdcv_change_joker_edition = function(args)
        mod.test_joker_edition = args.to_val or "None"
    end

    G.FUNCS.gdcv_change_joker_sticker = function(args)
        mod.test_joker_sticker = args.to_val or "None"
    end

    G.FUNCS.gdcv_spawn_selected_joker = function(e)
        UI.spawn_test_joker(mod)
    end

    G.FUNCS.gdcv_clear_all_jokers = function(e)
        UI.clear_all_jokers(mod)
    end

    G.FUNCS.gdcv_setup_custom_test_suite = function(e)
        UI.setup_custom_test_suite(mod)
    end
end

UI.JOKER_GRID_COLS = 3
UI.JOKER_GRID_ROWS = 4
UI.JOKER_GRID_PER_PAGE = UI.JOKER_GRID_COLS * UI.JOKER_GRID_ROWS

local function make_cycle_row(label, id, options, current_option, opt_callback, w, label_w)
    return {
        n = G.UIT.R,
        config = {align = "cr", padding = 0.03},
        nodes = {
            {n = G.UIT.C, config = {align = "cl", minw = label_w or 3.1}, nodes = {
                {n = G.UIT.T, config = {text = label, scale = 0.32, colour = G.C.UI.TEXT_LIGHT}}
            }},
            {n = G.UIT.C, config = {align = "cm"}, nodes = {
                create_option_cycle({
                    id = id,
                    options = options,
                    current_option = current_option,
                    opt_callback = opt_callback,
                    w = w or 2.8,
                    h = 0.52,
                    scale = 0.60
                })
            }}
        }
    }
end

local function find_option_index(options, value, default_idx)
    local target = tostring(value)
    for i, opt in ipairs(options) do
        if tostring(opt) == target then return i end
    end
    return default_idx or 1
end

function UI.build_joker_grid_definition(mod)
    local entries = UI.get_filtered_joker_entries(mod)
    mod.test_joker_list = entries
    local per_page = UI.JOKER_GRID_PER_PAGE
    local total_matches = #entries
    local total_pages = math.max(1, math.ceil(total_matches / per_page))
    mod.test_joker_total_pages = total_pages
    mod.test_joker_page = math.max(1, math.min(mod.test_joker_page or 1, total_pages))

    local start_idx = (mod.test_joker_page - 1) * per_page + 1
    local rarity_colours = {
        [1] = G.C.BLUE,
        [2] = G.C.GREEN,
        [3] = G.C.RED,
        [4] = G.C.PURPLE
    }

    local grid_rows = {}
    if total_matches == 0 then
        table.insert(grid_rows, {
            n = G.UIT.R,
            config = {align = "cm", padding = 0.06, minh = 2.0, minw = 6.9},
            nodes = {
                {n = G.UIT.T, config = {text = "No Jokers match filter.", scale = 0.34, colour = G.C.UI.TEXT_INACTIVE}}
            }
        })
    else
        for r = 1, UI.JOKER_GRID_ROWS do
            local row_nodes = {}
            for c = 1, UI.JOKER_GRID_COLS do
                local idx = start_idx + (r - 1) * UI.JOKER_GRID_COLS + (c - 1)
                local ent = entries[idx]
                if ent then
                    local tile_col = rarity_colours[ent.rarity] or G.C.BLUE
                    table.insert(row_nodes, {
                        n = G.UIT.C,
                        config = {align = "cm", padding = 0.03},
                        nodes = {
                            UIBox_button({
                                col = true,
                                button = 'gdcv_spawn_grid_joker',
                                ref_table = {key = ent.key, name = ent.name, index = idx},
                                label = {ent.name or ent.key},
                                minw = 2.18,
                                maxw = 2.02,
                                minh = 0.44,
                                scale = 0.27,
                                colour = tile_col
                            })
                        }
                    })
                else
                    table.insert(row_nodes, {
                        n = G.UIT.C,
                        config = {align = "cm", padding = 0.03},
                        nodes = {{n = G.UIT.B, config = {w = 2.18, h = 0.44}}}
                    })
                end
            end
            table.insert(grid_rows, {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.01},
                nodes = row_nodes
            })
        end
    end

    -- Pagination Bar
    table.insert(grid_rows, {
        n = G.UIT.R,
        config = {align = "cm", padding = 0.04},
        nodes = {
            UIBox_button({
                col = true,
                button = 'gdcv_joker_page_prev',
                label = {'< Prev'},
                minw = 1.35,
                minh = 0.42,
                scale = 0.28,
                colour = G.C.L_BLACK
            }),
            {n = G.UIT.C, config = {align = "cm", minw = 3.4}, nodes = {
                {n = G.UIT.T, config = {
                    text = string.format("Page %d / %d  (%d jokers)", mod.test_joker_page, total_pages, total_matches),
                    scale = 0.28,
                    colour = G.C.UI.TEXT_LIGHT
                }}
            }},
            UIBox_button({
                col = true,
                button = 'gdcv_joker_page_next',
                label = {'Next >'},
                minw = 1.35,
                minh = 0.42,
                scale = 0.28,
                colour = G.C.L_BLACK
            })
        }
    })

    return {
        n = G.UIT.ROOT,
        config = {align = "cm", padding = 0.03, colour = G.C.CLEAR, minw = 7.0},
        nodes = grid_rows
    }
end

function UI.refresh_joker_grid(mod, root_uibox)
    local uibox = root_uibox or G.OVERLAY_MENU
    if not uibox or not uibox.get_UIE_by_ID then return end
    local holder = uibox:get_UIE_by_ID('gdcv_joker_grid_holder')
    if not holder then return end
    if holder.config.object and holder.config.object.remove then
        holder.config.object:remove()
    end
    holder.config.object = UIBox{
        definition = UI.build_joker_grid_definition(mod),
        config = {offset = {x = 0, y = 0}, parent = holder, type = 'cm'}
    }
    if holder.UIBox and holder.UIBox.recalculate then
        holder.UIBox:recalculate()
    end
end

function UI.refresh_page_holder(mod, root_uibox)
    local uibox = root_uibox or G.OVERLAY_MENU
    if not uibox or not uibox.get_UIE_by_ID then return end
    local active_page = mod.config_page or 'dataset'
    for _, pid in ipairs({'dataset', 'jokers', 'system'}) do
        local tab_btn = uibox:get_UIE_by_ID('gdcv_tab_' .. pid)
        if tab_btn and tab_btn.config then
            tab_btn.config.colour = (pid == active_page) and G.C.ORANGE or G.C.L_BLACK
        end
    end
    local holder = uibox:get_UIE_by_ID('gdcv_page_holder')
    if not holder then return end
    if holder.config.object and holder.config.object.remove then
        holder.config.object:remove()
    end
    holder.config.object = UIBox{
        definition = UI.build_page_definition(mod),
        config = {offset = {x = 0, y = 0}, parent = holder, type = 'cm'}
    }
    if holder.UIBox and holder.UIBox.recalculate then
        holder.UIBox:recalculate()
    end
end

function UI.build_page_definition(mod)
    local page = mod.config_page or 'dataset'

    if page == 'jokers' then
        mod.test_joker_rarity_filter = mod.test_joker_rarity_filter or "All"
        mod.test_joker_edition = mod.test_joker_edition or "None"
        mod.test_joker_sticker = mod.test_joker_sticker or "None"
        mod.test_joker_search = mod.test_joker_search or ""

        local rarity_options = {"All", "Common", "Uncommon", "Rare", "Legendary"}
        local rarity_idx = find_option_index(rarity_options, mod.test_joker_rarity_filter, 1)

        local edition_options = {"None", "Foil", "Holo", "Polychrome", "Negative"}
        local edition_idx = find_option_index(edition_options, mod.test_joker_edition, 1)

        local sticker_options = {"None", "Eternal", "Perishable", "Rental"}
        local sticker_idx = find_option_index(sticker_options, mod.test_joker_sticker, 1)

        return {
            n = G.UIT.ROOT,
            config = {align = "cm", padding = 0.05, r = 0.1, colour = G.C.L_BLACK, minw = 7.2},
            nodes = {
                -- Search + Rarity Row
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.03},
                    nodes = {
                        create_text_input({
                            w = 2.4,
                            h = 0.46,
                            text_scale = 0.34,
                            max_length = 24,
                            extended_corpus = true,
                            prompt_text = "Search joker...",
                            ref_table = mod,
                            ref_value = "test_joker_search",
                            callback = function()
                                mod.test_joker_page = 1
                                UI.refresh_joker_grid(mod)
                            end
                        }),
                        {n = G.UIT.B, config = {w = 0.08, h = 0.1}},
                        UIBox_button({
                            col = true,
                            button = 'gdcv_apply_joker_search',
                            label = {'Find'},
                            minw = 0.85,
                            minh = 0.46,
                            scale = 0.28,
                            colour = G.C.BLUE
                        }),
                        {n = G.UIT.B, config = {w = 0.06, h = 0.1}},
                        UIBox_button({
                            col = true,
                            button = 'gdcv_clear_joker_search',
                            label = {'X'},
                            minw = 0.50,
                            minh = 0.46,
                            scale = 0.28,
                            colour = G.C.RED
                        }),
                        {n = G.UIT.B, config = {w = 0.12, h = 0.1}},
                        create_option_cycle({
                            id = 'gdcv_joker_rarity_cycle',
                            options = rarity_options,
                            current_option = rarity_idx,
                            opt_callback = 'gdcv_change_joker_rarity',
                            w = 2.1,
                            h = 0.46,
                            scale = 0.55
                        })
                    }
                },
                -- Edition & Sticker Row
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.02},
                    nodes = {
                        {n = G.UIT.T, config = {text = "Ed:", scale = 0.28, colour = G.C.UI.TEXT_LIGHT}},
                        {n = G.UIT.B, config = {w = 0.06, h = 0.1}},
                        create_option_cycle({
                            id = 'gdcv_joker_edition_cycle',
                            options = edition_options,
                            current_option = edition_idx,
                            opt_callback = 'gdcv_change_joker_edition',
                            w = 2.2,
                            h = 0.44,
                            scale = 0.54
                        }),
                        {n = G.UIT.B, config = {w = 0.22, h = 0.1}},
                        {n = G.UIT.T, config = {text = "Stk:", scale = 0.28, colour = G.C.UI.TEXT_LIGHT}},
                        {n = G.UIT.B, config = {w = 0.06, h = 0.1}},
                        create_option_cycle({
                            id = 'gdcv_joker_sticker_cycle',
                            options = sticker_options,
                            current_option = sticker_idx,
                            opt_callback = 'gdcv_change_joker_sticker',
                            w = 2.2,
                            h = 0.44,
                            scale = 0.54
                        })
                    }
                },
                -- Clickable 3x4 Paginated Joker Grid
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.02},
                    nodes = {
                        {
                            n = G.UIT.O,
                            config = {
                                id = 'gdcv_joker_grid_holder',
                                object = UIBox{
                                    definition = UI.build_joker_grid_definition(mod),
                                    config = {offset = {x = 0, y = 0}, align = 'cm'}
                                }
                            }
                        }
                    }
                },
                -- Board Controls Row
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.03},
                    nodes = {
                        UIBox_button({
                            col = true,
                            button = 'gdcv_clear_all_jokers',
                            label = {'Clear Jokers'},
                            minw = 2.4,
                            minh = 0.46,
                            scale = 0.29,
                            colour = G.C.RED
                        }),
                        {n = G.UIT.B, config = {w = 0.2, h = 0.1}},
                        UIBox_button({
                            col = true,
                            button = 'gdcv_setup_custom_test_suite',
                            label = {'Setup Custom Test Suite'},
                            minw = 3.8,
                            minh = 0.46,
                            scale = 0.29,
                            colour = G.C.PURPLE
                        })
                    }
                }
            }
        }
    elseif page == 'system' then
        return {
            n = G.UIT.ROOT,
            config = {align = "cm", padding = 0.08, r = 0.1, colour = G.C.L_BLACK, minw = 7.2},
            nodes = {
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.05, r = 0.08, colour = G.C.BLACK, minw = 6.8},
                    nodes = {
                        {n = G.UIT.T, config = {
                            text = "Engine Parity Tracker: AUTO-ACTIVE (run_telemetry.jsonl)",
                            scale = 0.29,
                            colour = G.C.GREEN
                        }}
                    }
                },
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.05, r = 0.08, colour = G.C.BLACK, minw = 6.8},
                    nodes = {
                        {n = G.UIT.R, config = {align = "cm", padding = 0.02}, nodes = {
                            {n = G.UIT.T, config = {text = "Hotkeys: [F1/F2] Toggle Mode  |  [Tab] Setup / Auto-Gen", scale = 0.26, colour = G.C.UI.TEXT_LIGHT}}
                        }},
                        {n = G.UIT.R, config = {align = "cm", padding = 0.02}, nodes = {
                            {n = G.UIT.T, config = {text = "Setup: [1-9,0,V] Select ROI/Box  |  [WASD/Arrows] Move/Resize  |  [R] Regen  |  [P] Shop/Pack  |  [C] Capture", scale = 0.24, colour = G.C.UI.TEXT_INACTIVE}}
                        }}
                    }
                },
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.04},
                    nodes = {
                        UIBox_button({col = true, button = 'gdcv_save_config', label = {'Save Config'}, minw = 2.1, minh = 0.52, scale = 0.30, colour = G.C.GREEN}),
                        {n = G.UIT.B, config = {w = 0.15, h = 0.1}},
                        UIBox_button({col = true, button = 'gdcv_load_config', label = {'Load Config'}, minw = 2.1, minh = 0.52, scale = 0.30, colour = G.C.ORANGE}),
                        {n = G.UIT.B, config = {w = 0.15, h = 0.1}},
                        UIBox_button({col = true, button = 'gdcv_reload_mod', label = {'Hot Reload'}, minw = 2.1, minh = 0.52, scale = 0.30, colour = G.C.BLUE})
                    }
                },
                {
                    n = G.UIT.R,
                    config = {align = "cm", padding = 0.04},
                    nodes = {
                        {
                            n = G.UIT.C,
                            config = {
                                align = "cm",
                                padding = 0.05,
                                r = 0.1,
                                hover = true,
                                colour = G.C.RED,
                                button = 'gdcv_clear_dataset',
                                minw = 3.2,
                                minh = 0.52,
                                shadow = true
                            },
                            nodes = {
                                {n = G.UIT.R, config = {align = "cm", padding = 0}, nodes = {
                                    {n = G.UIT.T, config = {
                                        ref_table = mod,
                                        ref_value = "clear_button_text",
                                        text = mod.clear_button_text,
                                        scale = 0.30,
                                        colour = G.C.UI.TEXT_LIGHT,
                                        shadow = true
                                    }}
                                }}
                            }
                        },
                        {n = G.UIT.B, config = {w = 0.25, h = 0.1}},
                        UIBox_button({col = true, button = 'gdcv_restart_game', label = {'Restart Game'}, minw = 3.2, minh = 0.52, scale = 0.30, colour = G.C.PURPLE})
                    }
                }
            }
        }
    end

    -- Default page: 'dataset'
    local current_mode_idx = (mod.config.mode == 'card_classifier' and 2)
        or (mod.config.mode == 'shop_classifier' and 3)
        or (mod.config.mode == 'end_round_classifier' and 4)
        or (mod.config.mode == 'hud_ocr' and 5)
        or (mod.config.mode == 'seed_ocr' and 6)
        or (mod.config.mode == 'deck_classifier' and 7)
        or 1

    local max_frames_options = {'1000', '2500', '5000', '10000', '20000'}
    local current_mf_idx = find_option_index(max_frames_options, mod.config.max_frames or 10000, 4)

    local max_clips_options = {'250', '500', '1000', '2500', '5000'}
    local current_mc_idx = find_option_index(max_clips_options, mod.config.max_clips or 1000, 3)

    local er_batch_options = {'1 (4x)', '2 (8x)', '4 (16x)'}
    local cur_batch_val = tonumber(mod.config.end_round_frames_per_tick) or 2
    local current_er_idx = (cur_batch_val == 1 and 1) or (cur_batch_val == 4 and 3) or 2

    local perkeo_max_options = {'4', '5', '6', '7', '8', '9', '10'}
    local current_pm_idx = find_option_index(perkeo_max_options, mod.config.perkeo_max_consumables or 10, 7)

    local selection_chance_options = {'0%', '50%', '100%'}
    local current_sc_idx = find_option_index(selection_chance_options, mod.config.card_selection_chance or '50%', 2)

    local frame_wait_options = {'0.02s', '0.05s', '0.08s', '0.15s', '0.25s'}
    local current_fw_val = tonumber(mod.config.frame_wait_time) or 0.08
    local current_fw_idx = 3
    for idx, val in ipairs(frame_wait_options) do
        local num = tonumber((val:gsub('s$', '')))
        if num and math.abs(num - current_fw_val) < 0.005 then current_fw_idx = idx; break end
    end

    local jd_installed = (JokerDisplay ~= nil and type(JokerDisplay.Definitions) == 'table' and type(Card) == 'table' and type(Card.update_joker_display) == 'function')
    local jd_status_text = jd_installed
        and "JokerDisplay Mod: DETECTED (30% shown / randomized, 70% hidden)"
        or "JokerDisplay Mod: NOT FOUND (safe fallback active)"

    return {
        n = G.UIT.ROOT,
        config = {align = "cm", padding = 0.06, r = 0.1, colour = G.C.L_BLACK, minw = 7.2},
        nodes = {
            make_cycle_row("Classifier Mode", 'gdcv_mode_cycle', {'Off', 'Card Classifier', 'Shop Classifier', 'End Round Classifier', 'HUD OCR', 'Seed OCR', 'Deck Classifier'}, current_mode_idx, 'gdcv_change_mode', 3.6),
            make_cycle_row("Target Frame Cap", 'gdcv_mf_cycle', max_frames_options, current_mf_idx, 'gdcv_change_max_frames'),
            make_cycle_row("End-Round Clip Cap", 'gdcv_mc_cycle', max_clips_options, current_mc_idx, 'gdcv_change_max_clips'),
            make_cycle_row("End-Round Sim Batch", 'gdcv_er_cycle', er_batch_options, current_er_idx, 'gdcv_change_er_batch'),
            make_cycle_row("Perkeo Max Consumables", 'gdcv_pm_cycle', perkeo_max_options, current_pm_idx, 'gdcv_change_perkeo_max'),
            make_cycle_row("Card Selection Chance", 'gdcv_sc_cycle', selection_chance_options, current_sc_idx, 'gdcv_change_card_selection_chance'),
            make_cycle_row("Frame Wait Time", 'gdcv_fw_cycle', frame_wait_options, current_fw_idx, 'gdcv_change_frame_wait_time'),
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.04},
                nodes = {
                    {n = G.UIT.T, config = {text = jd_status_text, scale = 0.25, colour = jd_installed and G.C.GREEN or G.C.UI.TEXT_INACTIVE}}
                }
            }
        }
    }
end

function UI.create_config_tab()
    local mod = G.DATASET_MOD
    mod.clear_confirm = false
    mod.clear_confirm_timer = 0
    mod.clear_button_text = "Clear Dataset"
    mod.config_page = mod.config_page or 'dataset'
    mod.stats_text = string.format("Card: %d  |  Shop: %d  |  Deck: %d  |  EndRound Clips: %d  |  Phase: %s",
        mod.frame_counts and mod.frame_counts.card or 0,
        mod.frame_counts and mod.frame_counts.shop or 0,
        mod.frame_counts and mod.frame_counts.deck or 0,
        mod.frame_counts and mod.frame_counts.end_round or 0,
        string.upper(mod.shop_phase or 'shop'))

    local active_page = mod.config_page

    return {
        n = G.UIT.ROOT,
        config = {
            r = 0.1,
            minw = 7.4,
            align = "cm",
            padding = 0.08,
            colour = G.C.BLACK
        },
        nodes = {
            -- Compact Title
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.03},
                nodes = {
                    {n = G.UIT.T, config = {text = "GameState Dataset CV - Settings", scale = 0.38, colour = G.C.GOLD}}
                }
            },
            -- Top Page Navigation Tabs
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.04},
                nodes = {
                    UIBox_button({
                        col = true,
                        id = 'gdcv_tab_dataset',
                        button = 'gdcv_switch_page',
                        ref_table = {page = 'dataset'},
                        label = {'Dataset Gen'},
                        minw = 2.25,
                        minh = 0.46,
                        scale = 0.29,
                        colour = (active_page == 'dataset') and G.C.ORANGE or G.C.L_BLACK
                    }),
                    {n = G.UIT.B, config = {w = 0.12, h = 0.1}},
                    UIBox_button({
                        col = true,
                        id = 'gdcv_tab_jokers',
                        button = 'gdcv_switch_page',
                        ref_table = {page = 'jokers'},
                        label = {'Joker Spawner'},
                        minw = 2.25,
                        minh = 0.46,
                        scale = 0.29,
                        colour = (active_page == 'jokers') and G.C.ORANGE or G.C.L_BLACK
                    }),
                    {n = G.UIT.B, config = {w = 0.12, h = 0.1}},
                    UIBox_button({
                        col = true,
                        id = 'gdcv_tab_system',
                        button = 'gdcv_switch_page',
                        ref_table = {page = 'system'},
                        label = {'System & Tools'},
                        minw = 2.25,
                        minh = 0.46,
                        scale = 0.29,
                        colour = (active_page == 'system') and G.C.ORANGE or G.C.L_BLACK
                    })
                }
            },
            -- Dynamic Page Body
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.03},
                nodes = {
                    {
                        n = G.UIT.O,
                        config = {
                            id = 'gdcv_page_holder',
                            object = UIBox{
                                definition = UI.build_page_definition(mod),
                                config = {offset = {x = 0, y = 0}, align = 'cm'}
                            }
                        }
                    }
                }
            },
            -- Compact Live Stats Footer
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.03},
                nodes = {
                    {
                        n = G.UIT.T,
                        config = {
                            ref_table = mod,
                            ref_value = "stats_text",
                            text = mod.stats_text,
                            scale = 0.28,
                            colour = G.C.UI.TEXT_INACTIVE
                        }
                    }
                }
            }
        }
    }
end

-- ==========================================
-- IN-GAME HUD & CALIBRATION OVERLAY
-- ==========================================
function G.DATASET_MOD:draw_debug()
    if self.config.mode == 'off' then
        if self.capture_flash and self.capture_flash > 0 then
            love.graphics.push('all')
            local msg = self.capture_flash_text or ""
            love.graphics.setColor(0, 0, 0, 0.85 * math.min(1.0, self.capture_flash))
            love.graphics.rectangle('fill', 14, 14, 720, 34, 6, 6)
            love.graphics.setColor(0.3, 1.0, 0.4, math.min(1.0, self.capture_flash))
            love.graphics.print(msg, 24, 24)
            love.graphics.pop()
        end
        return
    end
    if self.config.calibration_mode ~= 'setup' then return end

    love.graphics.push('all')

    -- 1. Draw Active ROIs
    local function draw_roi_box(roi, name, is_selected, r, g, b)
        if not roi then return end
        love.graphics.setColor(r, g, b, is_selected and 1.0 or 0.5)
        love.graphics.setLineWidth(is_selected and 4 or 2)
        love.graphics.rectangle('line', roi.x, roi.y, roi.w, roi.h)

        love.graphics.setColor(r, g, b, 0.9)
        love.graphics.print(name, roi.x + 5, roi.y - 18)
    end

    if self.config.mode == 'card_classifier' then
        local is_sel = (self.selected_target == 'roi_card_hand')
        draw_roi_box(self.config.rois.card_hand, "[ROI: CARD HAND]", is_sel, 0, 1, 0)
        local is_play_sel = (self.selected_target == 'roi_play_area')
        draw_roi_box(self.config.rois.play_area, "[ROI: PLAY AREA]", is_play_sel, 0.2, 0.8, 1)
    elseif self.config.mode == 'end_round_classifier' then
        local is_play_sel = (self.selected_target == 'roi_play_area')
        draw_roi_box(self.config.rois.play_area, "[ROI: PLAY AREA (256x72 @ 15fps)]", is_play_sel, 0.2, 0.8, 1)
    elseif self.config.mode == 'hud_ocr' then
        local sel = self.selected_target
        draw_roi_box(self.config.rois.hud_round_score,  "[1: ROUND SCORE]",  sel == 'roi_hud_round_score',  0.2, 0.9, 1.0)
        draw_roi_box(self.config.rois.hud_money,        "[2: MONEY ($)]",    sel == 'roi_hud_money',        1.0, 0.85, 0.2)
        draw_roi_box(self.config.rois.hud_hands,        "[3: HANDS]",        sel == 'roi_hud_hands',        0.3, 0.7, 1.0)
        draw_roi_box(self.config.rois.hud_discards,     "[4: DISCARDS]",     sel == 'roi_hud_discards',     1.0, 0.4, 0.4)
        draw_roi_box(self.config.rois.hud_ante,         "[5: ANTE]",         sel == 'roi_hud_ante',         0.7, 0.5, 1.0)
        draw_roi_box(self.config.rois.hud_round,        "[6: ROUND]",        sel == 'roi_hud_round',        0.4, 1.0, 0.6)
        draw_roi_box(self.config.rois.hud_reroll_cost,  "[7: REROLL COST]",  sel == 'roi_hud_reroll_cost',  0.2, 1.0, 0.3)
        draw_roi_box(self.config.rois.hud_blind_target, "[8: BLIND TARGET]", sel == 'roi_hud_blind_target', 1.0, 0.6, 0.2)
        draw_roi_box(self.config.rois.pack_hand,        "[9: PACK HAND (CV)]", sel == 'roi_pack_hand',     1.0, 0.3, 0.85)
    elseif self.config.mode == 'seed_ocr' then
        local sel = self.selected_target
        draw_roi_box(self.config.rois.seed_win,       "[1: SEED (WIN MODAL)]",       sel == 'roi_seed_win',       0.2, 1.0, 0.5)
        draw_roi_box(self.config.rois.seed_game_over, "[2: SEED (GAME OVER MODAL)]", sel == 'roi_seed_game_over', 1.0, 0.45, 0.3)
    elseif self.config.mode == 'deck_classifier' then
        local sel = self.selected_target
        draw_roi_box(self.config.rois.view_deck, "[1: ROI VIEW DECK]", sel == 'roi_view_deck', 0.2, 1.0, 0.5)
    elseif self.config.mode == 'shop_classifier' then
        local sel = self.selected_target
        draw_roi_box(self.config.rois.joker_bar, "[ROI: JOKER BAR (TOP)]", sel == 'roi_joker_bar', 0.2, 0.8, 1)
        draw_roi_box(self.config.rois.consumeables, "[ROI: CONSUMABLES (TOP RIGHT)]", sel == 'roi_consumeables', 1.0, 0.4, 0.9)

        if (self.shop_phase or 'shop') == 'shop' or sel == 'roi_shop' then
            draw_roi_box(self.config.rois.shop, "[ROI: SHOP ITEMS]", sel == 'roi_shop', 1, 0.8, 0.2)
        end
        if (self.shop_phase or 'shop') == 'shop' or sel == 'roi_voucher' then
            draw_roi_box(self.config.rois.voucher, "[ROI: VOUCHER SHELF]", sel == 'roi_voucher', 0.3, 1.0, 0.4)
        end
        if (self.shop_phase or 'shop') == 'shop' or sel == 'roi_booster' then
            draw_roi_box(self.config.rois.booster, "[ROI: BOOSTER PACKS (SHELF)]", sel == 'roi_booster', 0.9, 0.6, 0.2)
        end
        if (self.shop_phase or 'shop') == 'pack' or sel == 'roi_pack' then
            local kind_name = (self.current_pack_kind and self.current_pack_kind ~= 'Pack' and self.current_pack_kind) or 'BOOSTER'
            local p_name = string.format("[ROI: %s OPENING SCREEN]", string.upper(kind_name))
            draw_roi_box(self.config.rois.pack, p_name, sel == 'roi_pack', 1, 0.3, 0.8)
        end
    end

    -- 2. Draw Card Bounding Boxes Polygons
    local function is_card_active(card)
        if not card or card.REMOVED then return false end
        if self.config.mode == 'card_classifier' then
            if G.hand and G.hand.cards then
                for _, c in ipairs(G.hand.cards) do
                    if c == card then return true end
                end
            end
            if self.config.calibration_mode == 'setup' and G.play and G.play.cards then
                for _, c in ipairs(G.play.cards) do
                    if c == card then return true end
                end
            end
        elseif self.config.mode == 'shop_classifier' then
            if G.jokers and G.jokers.cards then
                for _, c in ipairs(G.jokers.cards) do
                    if c == card then return true end
                end
            end
            if (self.shop_phase or 'shop') == 'shop' then
                if G.shop_jokers and G.shop_jokers.cards then
                    for _, c in ipairs(G.shop_jokers.cards) do
                        if c == card then return true end
                    end
                end
                if G.shop_vouchers and G.shop_vouchers.cards then
                    for _, c in ipairs(G.shop_vouchers.cards) do
                        if c == card then return true end
                    end
                end
                if G.shop_booster and G.shop_booster.cards then
                    for _, c in ipairs(G.shop_booster.cards) do
                        if c == card then return true end
                    end
                end
            else
                if G.pack_cards and G.pack_cards.cards then
                    for _, c in ipairs(G.pack_cards.cards) do
                        if c == card then return true end
                    end
                end
            end
            if G.consumeables and G.consumeables.cards then
                for _, c in ipairs(G.consumeables.cards) do
                    if c == card then return true end
                end
            end
        elseif self.config.mode == 'deck_classifier' then
            return card.area ~= nil and card.area.config ~= nil and card.area.config.view_deck == true and not card.area.REMOVED
        end
        return false
    end

    if self.captured_boxes then
        for card, box in pairs(self.captured_boxes) do
            if not is_card_active(card) then
                self.captured_boxes[card] = nil
            elseif box and box.points then
                local pts = box.points
                local is_sel_box = (self.selected_target == 'box_joker_bar' and box.box_name == 'joker_bar_card') or
                                   (self.selected_target == 'box_item' and box.box_name == 'item_card') or
                                   (self.selected_target == 'box_consumeable' and box.box_name == 'consumeable_card') or
                                   (self.selected_target == 'box_voucher' and box.box_name == 'voucher_card') or
                                   (self.selected_target == 'box_booster' and box.box_name == 'booster_card') or
                                   (self.selected_target == 'box_card' and box.box_name == 'card') or
                                   (self.selected_target == 'box_deck_card' and box.box_name == 'deck_card')

                if box.box_name == 'joker_bar_card' then
                    love.graphics.setColor(0, 1, 1, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'consumeable_card' then
                    love.graphics.setColor(1, 0.4, 1, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'item_card' then
                    love.graphics.setColor(1, 1, 0, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'voucher_card' then
                    love.graphics.setColor(0.3, 1, 0.3, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'booster_card' then
                    love.graphics.setColor(1, 0.6, 0.1, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'deck_card' then
                    love.graphics.setColor(0.2, 1.0, 0.5, is_sel_box and 1.0 or 0.6)
                else
                    love.graphics.setColor(0, 1, 1, is_sel_box and 1.0 or 0.6)
                end

                love.graphics.setLineWidth(is_sel_box and 3 or 1.5)
                if #pts >= 8 then
                    love.graphics.polygon('line', pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7], pts[8])
                end
            end
        end
    end

    -- 3. Draw HUD Text Panel
    local _, obj_type, obj_label = self:get_selected_object()
    local obj = select(1, self:get_selected_object())

    love.graphics.setColor(0, 0, 0, 0.8)
    love.graphics.rectangle('fill', 8, 8, 920, 185, 8, 8)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.setLineWidth(1)
    love.graphics.rectangle('line', 8, 8, 920, 185, 8, 8)

    local mode_str = (self.config.mode == 'card_classifier' and "CARD CLASSIFIER")
        or (self.config.mode == 'end_round_classifier' and "END ROUND CLASSIFIER")
        or (self.config.mode == 'hud_ocr' and "HUD OCR")
        or (self.config.mode == 'seed_ocr' and "SEED OCR")
        or (self.config.mode == 'deck_classifier' and "VIEW DECK CLASSIFIER")
        or "SHOP & JOKER CLASSIFIER"
    love.graphics.setColor(1, 0.85, 0.2, 1)
    love.graphics.print(string.format("GAMESTATE CV CALIBRATION  -  [%s]", mode_str), 18, 14)

    love.graphics.setColor(1, 1, 1, 1)
    local coord_str = ""
    if obj and obj_type == 'roi' then
        coord_str = string.format("X: %.1f, Y: %.1f, W: %.1f, H: %.1f", obj.x, obj.y, obj.w, obj.h)
    elseif obj and obj_type == 'box' then
        coord_str = string.format("W: %.3f, H: %.3f, OX: %.3f, OY: %.3f", obj.w, obj.h, obj.ox, obj.oy)
    end
    love.graphics.print(string.format("Editing Target: %s  [%s]", obj_label, coord_str), 18, 36)

    love.graphics.setColor(0.3, 0.9, 1, 1)
    if self.config.mode == 'shop_classifier' then
        if (self.shop_phase or 'shop') == 'pack' then
            love.graphics.print("Keys: 1: ROI Jokers | 3/0: ROI Pack Open | 4: ROI Consumables | 5: Box Joker | 6: Box Item", 18, 58)
        else
            love.graphics.print("Keys: 1: ROI Jokers | 2: ROI Shop | 3: ROI Boosters | 4: ROI Consumables | V: ROI Vouchers | 5-9: Boxes", 18, 58)
        end
    elseif self.config.mode == 'end_round_classifier' then
        love.graphics.print("Keys: 3: ROI Play Area | R: Preview Scenario | C: Record 1 Test Clip | TAB: Auto-Generate Clips", 18, 58)
    elseif self.config.mode == 'hud_ocr' then
        love.graphics.print("Keys: 1:Score | 2:Money | 3:Hands | 4:Discards | 5:Ante | 6:Round | 7:Reroll | 8:BlindTarget | 9:PackHand", 18, 58)
    elseif self.config.mode == 'seed_ocr' then
        love.graphics.print("Keys: 1: Seed ROI (Win) | 2: Seed ROI (Game Over) | R: Spawn Seed Modal | P: Win/GameOver", 18, 58)
    elseif self.config.mode == 'deck_classifier' then
        love.graphics.print("Keys: 1: ROI View Deck | 2: Box Deck Card Left Edge | R: Randomize Deck | P: Full/Remaining | C: Capture", 18, 58)
    else
        love.graphics.print("Keys: 1: ROI Card Hand | 2: Box Hand Card | 3: ROI Play Area", 18, 58)
    end

    love.graphics.setColor(0.9, 0.9, 0.9, 1)
    love.graphics.print("W, A, S, D: Move / Shift Offset  |  Arrow Keys: Adjust Size (Shift: Fast, Ctrl: Slow)", 18, 80)
    love.graphics.print("F1/F2: Switch Mode  |  TAB: Toggle Gen/Setup  |  R: Randomize  |  P: Shop/Pack  |  C: Capture", 18, 102)
    love.graphics.print("[, ]: Save / Load Config (dataset/boxes.lua)", 18, 124)

    love.graphics.setColor(1, 1, 0.4, 1)
    if self.config.mode == 'end_round_classifier' then
        local cur_clips = (self.frame_counts and self.frame_counts.end_round) or 0
        local max_clips = self.config.max_clips or 1000
        local clip = self._active_end_round_clip
        local rec_str = clip
            and string.format(" | REC %s frame=%d (%s)", clip.id, #clip.frame_bytes, clip.scenario and clip.scenario.archetype or "")
            or string.format(" | Batch=%dx", self.config.end_round_frames_per_tick or 2)
        love.graphics.print(string.format("Next Clip: %06d / %d%s | Lua Mem: %.1f MB", cur_clips, max_clips, rec_str, collectgarbage("count") / 1024), 18, 146)
    elseif self.config.mode == 'hud_ocr' then
        local cur_hud = (self.frame_counts and self.frame_counts.hud_ocr) or 0
        local max_frames = self.config.max_frames or 10000
        local phase_str = string.upper(self.hud_phase or 'blind')
        love.graphics.print(string.format("Next HUD Frame: %05d / %d | Phase: %s (%d/%d) | P: Blind/Shop | Lua Mem: %.1f MB",
            cur_hud, max_frames, phase_str, self.hud_phase_step or 0, self.config.frames_per_phase or 100, collectgarbage("count") / 1024), 18, 146)
    elseif self.config.mode == 'seed_ocr' then
        local cur_seed = (self.frame_counts and self.frame_counts.seed_ocr) or 0
        local max_frames = self.config.max_frames or 10000
        local phase_str = string.upper(self.seed_phase or 'win')
        love.graphics.print(string.format("Next Seed Frame: %05d / %d | Modal: %s (%d/%d) | P: Win/GameOver | Lua Mem: %.1f MB",
            cur_seed, max_frames, phase_str, self.seed_phase_step or 0, self.config.frames_per_phase or 100, collectgarbage("count") / 1024), 18, 146)
    elseif self.config.mode == 'deck_classifier' then
        local cur_deck = (self.frame_counts and self.frame_counts.deck) or 0
        local max_frames = self.config.max_frames or 10000
        local deck_info_str = self.last_deck_summary or ("Phase: " .. string.upper(self.deck_phase or 'full'))
        love.graphics.print(string.format("Next Deck Frame: %05d / %d | %s | Lua Mem: %.1f MB",
            cur_deck, max_frames, deck_info_str, collectgarbage("count") / 1024), 18, 146)
    else
        local cur_frame = (self.config.mode == 'shop_classifier') and self.frame_counts.shop or self.frame_counts.card
        local phase_extra = ""
        if self.config.mode == 'shop_classifier' then
            local p_kind = (self.shop_phase == 'pack') and (self.current_pack_kind or 'Pack') or "Shop"
            local perk_str = self.is_perkeo_active and " [PERKEO]" or ""
            local show_str = self.pack_has_showman and " [SHOWMAN]" or ""
            phase_extra = string.format(" | Phase: %s (%d/100)%s%s", p_kind, self.phase_step or 0, perk_str, show_str)
        end
        love.graphics.print(string.format("Next Frame: %04d / %d%s | Lua Mem: %.1f MB", cur_frame, self.config.max_frames or 10000, phase_extra, collectgarbage("count") / 1024), 18, 146)
    end

    local stg_cnt = (G.STAGE_OBJECTS and G.STAGE_OBJECTS[G.STAGE] and #G.STAGE_OBJECTS[G.STAGE]) or 0
    local mov_cnt = (G.MOVEABLES and #G.MOVEABLES) or 0
    local crd_cnt = (G.I and G.I.CARD and #G.I.CARD) or 0
    local spr_cnt = (G.I and G.I.SPRITE and #G.I.SPRITE) or 0
    local uib_cnt = (G.I and G.I.UIBOX and #G.I.UIBOX) or 0
    love.graphics.setColor(0.6, 0.8, 1, 0.8)
    love.graphics.print(string.format("Nodes: %d | Moveables: %d | Cards: %d | Sprites: %d | UIBoxes: %d", stg_cnt, mov_cnt, crd_cnt, spr_cnt, uib_cnt), 18, 168)

    -- Flash message
    if self.capture_flash and self.capture_flash > 0 then
        love.graphics.setColor(0, 1, 0, math.min(1.0, self.capture_flash))
        love.graphics.print(self.capture_flash_text or "SAVED!", 18, 196)
    end

    love.graphics.pop()
end

-- ==========================================
-- KEYPRESS HANDLER
-- ==========================================
local MODE_ORDER = { 'off', 'card_classifier', 'shop_classifier', 'end_round_classifier', 'hud_ocr', 'seed_ocr', 'deck_classifier' }
local MODE_DISPLAY_NAMES = {
    off = 'OFF',
    card_classifier = 'CARD CLASSIFIER',
    shop_classifier = 'SHOP CLASSIFIER',
    end_round_classifier = 'END ROUND CLASSIFIER',
    hud_ocr = 'HUD OCR',
    seed_ocr = 'SEED OCR',
    deck_classifier = 'DECK CLASSIFIER',
}

local function cycle_classifier_mode(mod, step)
    local cur_idx = 1
    for i, m in ipairs(MODE_ORDER) do
        if m == (mod.config and mod.config.mode) then
            cur_idx = i
            break
        end
    end
    local next_idx = ((cur_idx - 1 + step) % #MODE_ORDER) + 1
    local next_mode = MODE_ORDER[next_idx]
    mod:set_mode(next_mode)
    mod.capture_flash = 2.0
    mod.capture_flash_text = "MODE: " .. (MODE_DISPLAY_NAMES[next_mode] or string.upper(next_mode))
end

function G.DATASET_MOD:on_keypressed(key)
    if key == 'f1' then
        cycle_classifier_mode(self, -1)
        return
    elseif key == 'f2' then
        cycle_classifier_mode(self, 1)
        return
    end

    if self.config.mode == 'off' then
        if G and G.GAME and self.RunTracker and self.RunTracker.undo_last_action then
            local is_ctrl_z = (key == 'z' and love and love.keyboard and (love.keyboard.isDown('lctrl') or love.keyboard.isDown('rctrl')))
            if key == 'backspace' or is_ctrl_z then
                local ok, count_or_err, act_name, new_step = self.RunTracker.undo_last_action()
                self.capture_flash = 3.5
                if ok then
                    self.capture_flash_text = string.format(
                        "UNDO: Reverted %d action%s back (before %s | step %d)",
                        count_or_err,
                        count_or_err == 1 and "" or "s",
                        tostring(act_name or "?"),
                        tonumber(new_step) or 0
                    )
                else
                    self.capture_flash_text = "UNDO: " .. tostring(count_or_err)
                end
                print(self.capture_flash_text)
            end
        end
        return
    end

    if key == 'tab' then
        if self.config.calibration_mode == 'setup' then
            if (self.config.mode == 'hud_ocr' or self.config.mode == 'seed_ocr' or self.config.mode == 'deck_classifier') and not (G and G.GAME) then
                self.capture_flash = 2.0
                self.capture_flash_text = "START A RUN FIRST BEFORE GENERATING DATA!"
                return
            end
            self.config.calibration_mode = 'generation'
            self.state = 'idle'
            self.capture_flash = 2.0
            self.capture_flash_text = "AUTO-GENERATION MODE STARTED!"
            print("DATASET: Generation Mode Started")
        else
            self.config.calibration_mode = 'setup'
            self.state = 'idle'
            if self.EndRoundCapture and self.config.mode == 'end_round_classifier' then
                self.EndRoundCapture.stop(self)
            end
            if self.cleanup_seed_overlay and self.config.mode == 'seed_ocr' then
                self:cleanup_seed_overlay()
            end
            if self.cleanup_deck_overlay and self.config.mode == 'deck_classifier' then
                self:cleanup_deck_overlay(true)
            end
            if self.restore_joker_display_config then self:restore_joker_display_config() end
            self.capture_flash = 2.0
            self.capture_flash_text = "CALIBRATION SETUP MODE ACTIVE."
            print("DATASET: Setup Mode Started")
        end
    elseif self.config.calibration_mode == 'setup' then
        -- Target selections
        if self.config.mode == 'card_classifier' then
            if key == '1' then self.selected_target = 'roi_card_hand'
            elseif key == '2' then self.selected_target = 'box_card'
            elseif key == '3' then self.selected_target = 'roi_play_area'
            end
        elseif self.config.mode == 'end_round_classifier' then
            if key == '3' or key == '1' then self.selected_target = 'roi_play_area' end
        elseif self.config.mode == 'hud_ocr' then
            if key == '1' then self.selected_target = 'roi_hud_round_score'
            elseif key == '2' then self.selected_target = 'roi_hud_money'
            elseif key == '3' then self.selected_target = 'roi_hud_hands'
            elseif key == '4' then self.selected_target = 'roi_hud_discards'
            elseif key == '5' then self.selected_target = 'roi_hud_ante'
            elseif key == '6' then self.selected_target = 'roi_hud_round'
            elseif key == '7' then self.selected_target = 'roi_hud_reroll_cost'
            elseif key == '8' then self.selected_target = 'roi_hud_blind_target'
            elseif key == '9' then self.selected_target = 'roi_pack_hand'
            end
        elseif self.config.mode == 'seed_ocr' then
            if key == '1' then self.selected_target = 'roi_seed_win'
            elseif key == '2' then self.selected_target = 'roi_seed_game_over'
            end
        elseif self.config.mode == 'deck_classifier' then
            if key == '1' then self.selected_target = 'roi_view_deck'
            elseif key == '2' then self.selected_target = 'box_deck_card'
            end
        elseif self.config.mode == 'shop_classifier' then
            if key == '1' then self.selected_target = 'roi_joker_bar'
            elseif key == '2' then self.selected_target = 'roi_shop'
            elseif key == '3' then
                if self.shop_phase == 'pack' then
                    self.selected_target = 'roi_pack'
                else
                    self.selected_target = 'roi_booster'
                end
            elseif key == '4' then self.selected_target = 'roi_consumeables'
            elseif key == 'v' then self.selected_target = 'roi_voucher'
            elseif key == '0' then self.selected_target = 'roi_pack'
            elseif key == '5' then self.selected_target = 'box_joker_bar'
            elseif key == '6' then self.selected_target = 'box_item'
            elseif key == '7' then self.selected_target = 'box_consumeable'
            elseif key == '8' then self.selected_target = 'box_voucher'
            elseif key == '9' then self.selected_target = 'box_booster'
            end
        end

        -- Actions
        if key == 'r' then
            if self.config.mode == 'card_classifier' then
                self:generate_random_hand()
            elseif self.config.mode == 'end_round_classifier' then
                if self.EndRoundGen and G and G.STAGE == G.STAGES.RUN and G.STATE == G.STATES.SELECTING_HAND then
                    self.EndRoundGen.setup_clip_scenario(self)
                    self.EndRoundGen.restore_user_state(self)
                else
                    self.capture_flash = 2.0
                    self.capture_flash_text = "ENTER AN ACTIVE BLIND HAND (SELECTING_HAND) FIRST!"
                end
            elseif self.config.mode == 'hud_ocr' then
                self:generate_hud_step()
            elseif self.config.mode == 'seed_ocr' then
                self:generate_seed_step()
            elseif self.config.mode == 'deck_classifier' then
                self:setup_deck_screen()
            else
                if (self.shop_phase or 'shop') == 'shop' then
                    self:setup_shop_screen()
                else
                    self:setup_pack_screen()
                end
            end
            if self.ShopGen and self.ShopGen.cleanup_cycle then
                self.ShopGen.cleanup_cycle()
            else
                collectgarbage("collect")
                collectgarbage("collect")
            end
        elseif key == 'p' and self.config.mode == 'shop_classifier' then
            self:toggle_shop_screen_mode()
        elseif key == 'p' and self.config.mode == 'hud_ocr' then
            self:toggle_hud_screen_mode()
        elseif key == 'p' and self.config.mode == 'seed_ocr' then
            self:toggle_seed_screen_mode()
        elseif key == 'p' and self.config.mode == 'deck_classifier' then
            self:toggle_deck_screen_mode()
        elseif key == 't' and self.config.mode == 'card_classifier' then
            self:randomize_hand_selection()
        elseif key == 'c' then
            if self.config.mode == 'end_round_classifier' and self.EndRoundCapture then
                self.EndRoundCapture.run_single_test_clip(self)
            else
                self:capture_frame()
            end
        elseif key == ',' or key == '[' then
            self:save_config()
            self.capture_flash = 1.0
            self.capture_flash_text = "CONFIG SAVED!"
        elseif key == '.' or key == ']' then
            self:load_config(true)
            self.capture_flash = 1.0
            self.capture_flash_text = "CONFIG LOADED!"
        end
    end
end

-- ==========================================
-- UPDATE HOOK (NUDGING & GENERATION)
-- ==========================================
function G.DATASET_MOD:update(dt)
    if self.capture_flash and self.capture_flash > 0 then
        self.capture_flash = math.max(0, self.capture_flash - dt)
    end

    if self.clear_confirm then
        self.clear_confirm_timer = (self.clear_confirm_timer or 0) - dt
        if self.clear_confirm_timer <= 0 then
            self.clear_confirm = false
            self.clear_button_text = "Clear Dataset"
        end
    elseif self.clear_reset_timer then
        self.clear_reset_timer = self.clear_reset_timer - dt
        if self.clear_reset_timer <= 0 then
            self.clear_reset_timer = nil
            self.clear_button_text = "Clear Dataset"
        end
    end

    self.stats_text = string.format("Card: %d  |  Shop: %d  |  Deck: %d  |  EndRound Clips: %d  |  Phase: %s",
        self.frame_counts and self.frame_counts.card or 0,
        self.frame_counts and self.frame_counts.shop or 0,
        self.frame_counts and self.frame_counts.deck or 0,
        self.frame_counts and self.frame_counts.end_round or 0,
        string.upper(self.shop_phase or 'shop'))

    if not self.config or self.config.mode == 'off' then return end

    if self.config.calibration_mode == 'setup' then
        local obj, obj_type = self:get_selected_object()
        if not obj then return end

        if obj_type == 'roi' then
            local roi_speed = 750
            if love.keyboard.isDown('lctrl', 'rctrl') then
                roi_speed = 40
            elseif love.keyboard.isDown('lshift', 'rshift') then
                roi_speed = 2500
            end
            local roi_step = roi_speed * dt
            if love.keyboard.isDown('w') then obj.y = obj.y - roi_step end
            if love.keyboard.isDown('s') then obj.y = obj.y + roi_step end
            if love.keyboard.isDown('a') then obj.x = obj.x - roi_step end
            if love.keyboard.isDown('d') then obj.x = obj.x + roi_step end
            if love.keyboard.isDown('up') then obj.h = obj.h - roi_step end
            if love.keyboard.isDown('down') then obj.h = obj.h + roi_step end
            if love.keyboard.isDown('left') then obj.w = obj.w - roi_step end
            if love.keyboard.isDown('right') then obj.w = obj.w + roi_step end
        elseif obj_type == 'box' then
            local box_speed = 0.5
            if love.keyboard.isDown('lctrl', 'rctrl') then
                box_speed = 0.05
            elseif love.keyboard.isDown('lshift', 'rshift') then
                box_speed = 2.0
            end
            local step = box_speed * dt
            local moved = false
            if love.keyboard.isDown('w') then obj.oy = obj.oy - step; moved = true end
            if love.keyboard.isDown('s') then obj.oy = obj.oy + step; moved = true end
            if love.keyboard.isDown('a') then obj.ox = obj.ox - step; moved = true end
            if love.keyboard.isDown('d') then obj.ox = obj.ox + step; moved = true end
            if love.keyboard.isDown('up') then obj.h = obj.h - step; moved = true end
            if love.keyboard.isDown('down') then obj.h = obj.h + step; moved = true end
            if love.keyboard.isDown('left') then obj.w = obj.w - step; moved = true end
            if love.keyboard.isDown('right') then obj.w = obj.w + step; moved = true end
            if moved then
                self.captured_boxes = setmetatable({}, {__mode = "k"})
            end
        end
    elseif self.config.calibration_mode == 'generation' then
        if self.config.mode == 'end_round_classifier' then
            -- Clip stepping and capture run via EndRoundCapture.step_love_update & capture_after_draw.
            return
        end
        local cur_frame = (self.config.mode == 'shop_classifier' and self.frame_counts.shop)
            or (self.config.mode == 'hud_ocr' and self.frame_counts.hud_ocr)
            or (self.config.mode == 'seed_ocr' and self.frame_counts.seed_ocr)
            or (self.config.mode == 'deck_classifier' and self.frame_counts.deck)
            or self.frame_counts.card
        local max_frames = self.config.max_frames or 10000

        if cur_frame >= max_frames then
            self.config.calibration_mode = 'setup'
            self.state = 'idle'
            if self.restore_joker_display_config then self:restore_joker_display_config() end
            if self.cleanup_seed_overlay and self.config.mode == 'seed_ocr' then self:cleanup_seed_overlay() end
            if self.cleanup_deck_overlay and self.config.mode == 'deck_classifier' then self:cleanup_deck_overlay(true) end
            self.capture_flash = 3.0
            self.capture_flash_text = string.format("REACHED TARGET CAP (%d FRAMES)! GENERATION STOPPED.", max_frames)
            print(string.format("DATASET: Reached threshold cap of %d frames. Stopped generation.", max_frames))
            return
        end

        if self.state == 'idle' then
            if self.config.mode == 'card_classifier' then
                self:generate_random_hand()
                self:randomize_background()
            elseif self.config.mode == 'shop_classifier' then
                self:generate_shop_step()
            elseif self.config.mode == 'hud_ocr' then
                local ok = self:generate_hud_step()
                if not ok then
                    self.config.calibration_mode = 'setup'
                    return
                end
            elseif self.config.mode == 'seed_ocr' then
                local ok = self:generate_seed_step()
                if not ok then
                    self.config.calibration_mode = 'setup'
                    return
                end
            elseif self.config.mode == 'deck_classifier' then
                local ok = self:generate_deck_step()
                if not ok then
                    self.config.calibration_mode = 'setup'
                    return
                end
            end

            self.state = 'waiting'
            self.timer = tonumber(self.config.frame_wait_time) or 0.08
        elseif self.state == 'waiting' then
            self.timer = self.timer - dt
            if self.timer <= 0 then
                self.state = 'capturing'
                self:capture_frame()
            end
        end
    end
end

return UI
