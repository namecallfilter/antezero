-- gamestate_dataset_cv/src/shop_gen.lua
-- Contains shop screen, booster pack, joker bar, and top consumable area generation.

local ShopGen = {}

-- Cached Center Pools
ShopGen.cached_jokers = nil
ShopGen.cached_non_legendary_jokers = nil
ShopGen.cached_tarots = nil
ShopGen.cached_spectrals = nil
ShopGen.cached_planets = nil
ShopGen.cached_vouchers = nil
ShopGen.cached_boosters = nil

function ShopGen.get_all_vouchers()
    if ShopGen.cached_vouchers and #ShopGen.cached_vouchers > 0 then
        return ShopGen.cached_vouchers
    end
    local list = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == 'Voucher' and k ~= 'v_undiscovered' and k ~= 'v_locked' then
                table.insert(list, v)
            end
        end
        table.sort(list, function(a, b) return (a.key or "") < (b.key or "") end)
    end
    ShopGen.cached_vouchers = list
    return list
end

function ShopGen.get_all_boosters()
    if ShopGen.cached_boosters and #ShopGen.cached_boosters > 0 then
        return ShopGen.cached_boosters
    end
    local list = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == 'Booster' and k ~= 'p_undiscovered' and k ~= 'p_up' and k ~= 'p_profile' then
                table.insert(list, v)
            end
        end
        table.sort(list, function(a, b) return (a.key or "") < (b.key or "") end)
    end
    ShopGen.cached_boosters = list
    return list
end

function ShopGen.get_all_jokers(allow_legendary)
    if allow_legendary == nil then allow_legendary = true end
    if allow_legendary and ShopGen.cached_jokers and #ShopGen.cached_jokers > 0 then
        return ShopGen.cached_jokers
    elseif not allow_legendary and ShopGen.cached_non_legendary_jokers and #ShopGen.cached_non_legendary_jokers > 0 then
        return ShopGen.cached_non_legendary_jokers
    end

    local all_j = {}
    local non_leg_j = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == 'Joker' and k ~= 'j_undiscovered' then
                table.insert(all_j, v)
                if v.rarity ~= 4 then
                    table.insert(non_leg_j, v)
                end
            end
        end
        table.sort(all_j, function(a, b) return (a.key or "") < (b.key or "") end)
        table.sort(non_leg_j, function(a, b) return (a.key or "") < (b.key or "") end)
    end
    ShopGen.cached_jokers = all_j
    ShopGen.cached_non_legendary_jokers = non_leg_j

    return allow_legendary and all_j or non_leg_j
end

function ShopGen.get_copyable_jokers()
    if ShopGen.cached_copyable_jokers and #ShopGen.cached_copyable_jokers > 0 then
        return ShopGen.cached_copyable_jokers
    end
    local list = {}
    local all_j = ShopGen.get_all_jokers(true)
    for _, center in ipairs(all_j) do
        if center.blueprint_compat and center.key ~= 'j_blueprint' and center.key ~= 'j_brainstorm' then
            table.insert(list, center)
        end
    end
    ShopGen.cached_copyable_jokers = list
    return list
end

function ShopGen.get_all_consumables(set_type, include_hidden)
    local list = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == set_type and k ~= 'c_undiscovered' then
                local is_hidden = (k == 'c_soul' or k == 'c_black_hole')
                if include_hidden or not is_hidden then
                    table.insert(list, v)
                end
            end
        end
        table.sort(list, function(a, b) return (a.key or "") < (b.key or "") end)
    end
    return list
end

-- Cached BUMod Variants Manifest
ShopGen.bu_variants = nil

function ShopGen.load_bu_variants()
    if ShopGen.bu_variants ~= nil then return ShopGen.bu_variants end
    local ok, manifest = pcall(function()
        if SMODS and SMODS.load_file then
            local chunk = SMODS.load_file("src/bu_variants_manifest.lua", "gamestate_dataset_cv")
            if chunk then return chunk() end
        end
        return nil
    end)
    if ok and manifest then
        ShopGen.bu_variants = manifest
    else
        ShopGen.bu_variants = {}
    end
    return ShopGen.bu_variants
end

-- Cache loaded Love2D Images for BU variants so we don't reload from disk repeatedly
ShopGen.loaded_images = {}
function ShopGen.get_variant_image(filename)
    if ShopGen.loaded_images[filename] then
        return ShopGen.loaded_images[filename]
    end
    local path = "Mods/gamestate_dataset_cv/assets/bu_variants/" .. filename
    if love.filesystem.getInfo(path) then
        local img_data = love.image.newImageData(path)
        local img = love.graphics.newImage(img_data)
        ShopGen.loaded_images[filename] = img
        if img_data and img_data.release then img_data:release() end
        return img
    end
    return nil
end

ShopGen.cached_quads = {}
function ShopGen.get_quad(x, y, w, h, sw, sh)
    local k = string.format("%d_%d_%d_%d_%d_%d", x, y, w, h, sw, sh)
    if not ShopGen.cached_quads[k] then
        ShopGen.cached_quads[k] = love.graphics.newQuad(x, y, w, h, sw, sh)
    end
    return ShopGen.cached_quads[k]
end

local function apply_custom_center_sprite(card, asset_key, img, pos_x, pos_y, frame_w, frame_h)
    if not card or not img or not card.children or not card.children.center then return false end
    local center_sprite = card.children.center
    local orig_atlas = center_sprite.atlas
    local orig_px = (orig_atlas and orig_atlas.px and orig_atlas.px > 0) and orig_atlas.px or 71
    local orig_py = (orig_atlas and orig_atlas.py and orig_atlas.py > 0) and orig_atlas.py or 95
    local ratio_x = (center_sprite.scale and center_sprite.scale.x) and (center_sprite.scale.x / orig_px) or 1
    local ratio_y = (center_sprite.scale and center_sprite.scale.y) and (center_sprite.scale.y / orig_py) or 1

    center_sprite.atlas = {
        name = "gdcv_bu_" .. tostring(asset_key),
        image = img,
        px = frame_w,
        py = frame_h
    }
    center_sprite.scale = {
        x = frame_w * ratio_x,
        y = frame_h * ratio_y
    }
    if center_sprite.T and center_sprite.T.w and center_sprite.T.w > 0 and center_sprite.T.h and center_sprite.T.h > 0 then
        center_sprite.scale_mag = math.min(center_sprite.scale.x / center_sprite.T.w, center_sprite.scale.y / center_sprite.T.h)
    end
    if center_sprite.set_sprite_pos then
        center_sprite:set_sprite_pos({x = pos_x or 0, y = pos_y or 0})
    end
    return true
end

function G.DATASET_MOD:get_joker_pools()
    if not ShopGen.cached_tarots or #ShopGen.cached_tarots == 0 then
        ShopGen.cached_tarots = ShopGen.get_all_consumables('Tarot', false)
        ShopGen.cached_spectrals = ShopGen.get_all_consumables('Spectral', false)
        ShopGen.cached_planets = ShopGen.get_all_consumables('Planet', false)
    end
    return ShopGen.get_all_jokers(true), ShopGen.cached_tarots, ShopGen.cached_spectrals, ShopGen.cached_planets
end

-- Create a fully randomized Joker card
function G.DATASET_MOD:create_random_joker_card(area, forced_center, allow_legendary, spawn_x, spawn_y, boost_blueprint_odds)
    if allow_legendary == nil then allow_legendary = true end
    local jokers = ShopGen.get_all_jokers(allow_legendary)
    if #jokers == 0 then return nil end

    local center = forced_center
    if not center then
        if boost_blueprint_odds then
            -- 10x relative odds for j_blueprint and j_brainstorm
            local bp_center = G.P_CENTERS and G.P_CENTERS['j_blueprint']
            local bs_center = G.P_CENTERS and G.P_CENTERS['j_brainstorm']
            local total_weight = #jokers + 18
            local roll = math.random() * total_weight
            if roll < 10 and bp_center then
                center = bp_center
            elseif roll < 20 and bs_center then
                center = bs_center
            else
                center = jokers[math.random(#jokers)]
            end
        else
            center = jokers[math.random(#jokers)]
        end
    end

    local sx = spawn_x or (area.T.x + area.T.w / 2)
    local sy = spawn_y or area.T.y

    local card = Card(
        sx, sy,
        G.CARD_W, G.CARD_H,
        nil, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true,
            bypass_back = G.GAME and G.GAME.selected_back and G.GAME.selected_back.pos
        }
    )

    -- Editions distribution:
    -- None (60%), Foil (10%), Holo (10%), Polychrome (10%), Negative (10%)
    local ed_roll = math.random()
    if ed_roll < 0.10 then
        card:set_edition({foil = true}, true, true)
    elseif ed_roll < 0.20 then
        card:set_edition({holo = true}, true, true)
    elseif ed_roll < 0.30 then
        card:set_edition({polychrome = true}, true, true)
    elseif ed_roll < 0.40 then
        card:set_edition({negative = true}, true, true)
    end

    -- Stickers distribution:
    -- Eternal (15%), Perishable (15%) [mutually exclusive]
    -- Rental (20%) [independent]
    local stick_roll = math.random()
    if stick_roll < 0.15 then
        if card.set_eternal then card:set_eternal(true) else card.ability.eternal = true end
    elseif stick_roll < 0.30 then
        if card.set_perishable then card:set_perishable(true) else card.ability.perishable = true end
    end

    if math.random() < 0.20 then
        if card.set_rental then card:set_rental(true) else card.ability.rental = true end
    end

    -- Joker Stake Win Stickers (top-right sticker visible to YOLO detector):
    -- ~50% chance to display one of the 8 stake stickers (White..Gold)
    if math.random() < 0.50 then
        local stake_stickers = G.sticker_map or {'White', 'Red', 'Green', 'Black', 'Blue', 'Purple', 'Orange', 'Gold'}
        if #stake_stickers > 0 then
            local chosen_sticker = stake_stickers[math.random(#stake_stickers)]
            card.sticker = chosen_sticker
            card.sticker_run = chosen_sticker
        end
    else
        -- Prevent Card:update() from populating profile win stickers on un-stickered jokers
        card.sticker = nil
        card.sticker_run = 'NONE'
    end

    -- BUMod Variant Selection:
    -- Check if this Joker has historical BUMod sprite variations
    local bu_manifest = ShopGen.load_bu_variants()
    local c_key = center.key
    local variants = bu_manifest[c_key]
    -- Handle trans jokers which cover j_seance and j_banner
    if not variants and (c_key == 'j_seance' or c_key == 'j_banner') then
        variants = bu_manifest['jokers_trans']
    end

    card.custom_art_variant = "vanilla_or_active"
    if variants and #variants > 0 then
        if not card.edition and math.random() < 0.20 then
            card:set_edition({negative = true}, true, true)
        end
        -- Evenly disperse: 50% active/vanilla mod art, 50% split across historical BUMod variants
        if math.random() < 0.50 then
            local chosen_var = variants[math.random(#variants)]
            local img = ShopGen.get_variant_image(chosen_var.filename)
            if img then
                local w, h = img:getDimensions()
                local frame_w = (w > 200) and 142 or w
                local pos_x = (c_key == 'j_banner' and w > 200) and 1 or 0
                if apply_custom_center_sprite(card, chosen_var.filename, img, pos_x, 0, frame_w, h) then
                    card.custom_art_variant = string.format("bumod_v%d_%s", chosen_var.version or 1, chosen_var.commit or "hist")
                end
            end
        end
    end

    -- Blueprint Mod appearance effect:
    -- If this card is Blueprint or Brainstorm on the joker bar, apply the copying shader effect
    -- to mimic a random copyable joker ~70% of the time, and keep vanilla textures ~30% of the time.
    if (center.key == 'j_blueprint' or center.key == 'j_brainstorm') and area == G.jokers then
        if math.random() < 0.70 then
            local copyables = ShopGen.get_copyable_jokers()
            if copyables and #copyables > 0 then
                local chosen_target = copyables[math.random(#copyables)]
                local dummy = Card(
                    0, 0, G.CARD_W, G.CARD_H,
                    nil, chosen_target,
                    {bypass_discovery_center = true, bypass_discovery_ui = true}
                )
                card.dataset_mimic_target = dummy
                card.dataset_mimic_dummy = dummy
                card.custom_art_variant = "mimic_" .. chosen_target.key

                if Blueprint then
                    if center.key == 'j_brainstorm' and Blueprint.apply_brainstorm_sprite and Blueprint.brainstorm_enabled then
                        Blueprint.apply_brainstorm_sprite(card, dummy)
                    elseif center.key == 'j_blueprint' and Blueprint.apply_blueprint_sprite then
                        Blueprint.apply_blueprint_sprite(card, dummy)
                    end
                end
            end
        else
            card.dataset_mimic_target = false
            card.custom_art_variant = "vanilla"
            if Blueprint and Blueprint.restore_card_sprite then
                Blueprint.restore_card_sprite(card)
            end
        end
    end

    return card
end

-- Helper to initialize Card:update()-populated fields before synchronous UIBox/highlight calls
local function ensure_consumable_ready(card)
    if not card then return end
    card.eligible_strength_jokers = card.eligible_strength_jokers or {}
    card.eligible_editionless_jokers = card.eligible_editionless_jokers or {}
    if card.ability and card.ability.consumeable and card.ability.consumeable.max_highlighted then
        card.ability.consumeable.mod_num = card.ability.consumeable.mod_num or math.min(5, card.ability.consumeable.max_highlighted)
    end
end

-- Create a consumable card (Tarot, Spectral, Planet)
function G.DATASET_MOD:create_random_consumable_card(area, set_type, forced_center, spawn_x, spawn_y)
    local _, tarots, spectrals, planets = self:get_joker_pools()
    local pool = (set_type == 'Tarot' and tarots) or (set_type == 'Spectral' and spectrals) or planets
    if not pool or #pool == 0 then return nil end
    local center = forced_center or pool[math.random(#pool)]

    local sx = spawn_x or (area.T.x + area.T.w / 2)
    local sy = spawn_y or area.T.y

    local card = Card(
        sx, sy,
        G.CARD_W, G.CARD_H,
        nil, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true,
            bypass_back = G.GAME and G.GAME.selected_back and G.GAME.selected_back.pos
        }
    )
    ensure_consumable_ready(card)
    return card
end

-- Create a randomized playing card (Shop Magic Trick offering or Standard pack card)
function G.DATASET_MOD:create_random_playing_card(area, spawn_x, spawn_y)
    local base_card = self:get_random_card_center()
    if not base_card then return nil end

    local center = G.P_CENTERS.c_base
    if math.random() < 0.10 and G.P_CENTERS['m_stone'] then
        center = G.P_CENTERS['m_stone']
    else
        local enhancement = self:get_random_enhancement()
        if enhancement and G.P_CENTERS[enhancement] then
            center = G.P_CENTERS[enhancement]
        end
    end

    local sx = spawn_x or (area.T.x + area.T.w / 2)
    local sy = spawn_y or area.T.y

    local card = Card(
        sx, sy,
        G.CARD_W, G.CARD_H,
        base_card, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true,
            bypass_back = G.GAME and G.GAME.selected_back and G.GAME.selected_back.pos
        }
    )

    -- Force re-application of base card to guarantee sprite atlas is loaded (vanilla blank art fix)
    -- Pass initial=true so active blind never debuffs shop/pack playing cards
    if base_card and card.set_base then
        card:set_base(base_card, true)
    end
    card.debuff = false

    -- Edition: None, Foil, Holo, Polychrome
    local edition = self:get_random_edition()
    if edition then card:set_edition({[edition] = true}, true, true) end

    -- Seal: Gold, Red, Blue, Purple
    local seal = self:get_random_seal()
    if seal then card:set_seal(seal, true, true) end

    card.ability = card.ability or {}
    card.ability.set = (center and center.set) or 'Enhanced'
    card.custom_art_variant = "default"

    return card
end

-- Create a voucher card
function G.DATASET_MOD:create_random_voucher_card(area, forced_center, spawn_x, spawn_y)
    local vouchers = ShopGen.get_all_vouchers()
    if #vouchers == 0 then return nil end
    local center = forced_center or vouchers[math.random(#vouchers)]

    local sx = spawn_x or (area.T.x + area.T.w / 2)
    local sy = spawn_y or area.T.y

    local card = Card(
        sx, sy,
        G.CARD_W, G.CARD_H,
        G.P_CARDS.empty, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true
        }
    )
    card.shop_voucher = true
    card.custom_art_variant = "default"

    -- BUMod v_hone variant integration if v_hone rolled
    if center.key == 'v_hone' and math.random() < 0.50 then
        local img = ShopGen.get_variant_image("v_hone.png")
        if img then
            local w, h = img:getDimensions()
            if apply_custom_center_sprite(card, "v_hone", img, 0, 0, w, h) then
                card.custom_art_variant = "bumod_v_hone"
            end
        end
    end

    return card
end

local spectral_frame_map = {
    p_spectral_normal_1 = 0,
    p_spectral_normal_2 = 1,
    p_spectral_jumbo_1 = 2,
    p_spectral_mega_1 = 3
}

-- Create a booster pack card
function G.DATASET_MOD:create_random_booster_card(area, forced_center, spawn_x, spawn_y, booster_pos)
    local boosters = ShopGen.get_all_boosters()
    if #boosters == 0 then return nil end
    local center = forced_center or boosters[math.random(#boosters)]

    local sx = spawn_x or (area.T.x + area.T.w / 2)
    local sy = spawn_y or area.T.y

    local card = Card(
        sx, sy,
        G.CARD_W * 1.27, G.CARD_H * 1.27,
        G.P_CARDS.empty, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true
        }
    )
    if booster_pos then
        card.ability.booster_pos = booster_pos
    end
    card.custom_art_variant = "default"

    -- Mix in Balatro University custom spectral pack arts (~50% chance for spectral packs)
    local spec_idx = spectral_frame_map[center.key]
    if spec_idx ~= nil and math.random() < 0.50 then
        local img = ShopGen.get_variant_image("spectral_trans.png")
        if img then
            local w, h = img:getDimensions()
            local frame_w = math.floor(w / 4)
            if apply_custom_center_sprite(card, "spectral_trans", img, spec_idx, 0, frame_w, h) then
                card.custom_art_variant = "bumod_spectral_trans"
            end
        end
    end

    return card
end

-- Helper to randomize shop price ($0-$15) and sell value ($0-$15) as integers
local function randomize_card_costs(card)
    if not card then return end
    if card.set_cost then card:set_cost() end
    card.cost = math.random(0, 15)
    card.sell_cost = math.random(0, 15)
    card.sell_cost_label = (card.facing == 'back') and '?' or card.sell_cost
    card._gdcv_cost_randomized = true
end

-- Helper to attach shop card price tag and action buttons synchronously without queuing delayed events in G.E_MANAGER
local function attach_shop_card_ui_sync(card)
    if not card then return end
    randomize_card_costs(card)

    if card.children and card.children.price then
        pcall(function() card.children.price:remove() end)
        card.children.price = nil
    end
    if card.children and card.children.buy_button then
        pcall(function() card.children.buy_button:remove() end)
        card.children.buy_button = nil
    end
    if card.children and card.children.buy_and_use_button then
        pcall(function() card.children.buy_and_use_button:remove() end)
        card.children.buy_and_use_button = nil
    end

    local t1 = {
        n=G.UIT.ROOT, config = {minw = 0.6, align = 'tm', colour = darken(G.C.BLACK, 0.2), shadow = true, r = 0.05, padding = 0.05, minh = 1}, nodes={
            {n=G.UIT.R, config={align = "cm", colour = lighten(G.C.BLACK, 0.1), r = 0.1, minw = 1, minh = 0.55, emboss = 0.05, padding = 0.03}, nodes={
                {n=G.UIT.O, config={object = DynaText({
                    string = {{prefix = localize('$'), ref_table = card, ref_value = 'cost'}},
                    colours = {G.C.MONEY},
                    shadow = true,
                    silent = true,
                    bump = false,
                    scale = 0.5,
                    min_cycle_time = 0
                })}},
            }}
        }
    }

    local target_offset_y = (card.ability and card.ability.set == 'Booster') and 0.5 or 0.38

    card.children.price = UIBox{
        definition = t1,
        config = {align = "tm", offset = {x = 0, y = target_offset_y}, major = card, bond = 'Strong', parent = card}
    }
    if card.children.price then
        card.children.price.alignment.offset.y = target_offset_y
        if card.children.price.align_to_major then card.children.price:align_to_major() end
        if card.children.price.hard_set_VT then card.children.price:hard_set_VT() end
    end

    local t2 = (card.ability and card.ability.set == 'Voucher') and {
        n=G.UIT.ROOT, config = {ref_table = card, minw = 1.1, maxw = 1.3, padding = 0.1, align = 'bm', colour = G.C.GREEN, shadow = true, r = 0.08, minh = 0.94, func = 'can_redeem', one_press = true, button = 'redeem_from_shop', hover = true}, nodes={
            {n=G.UIT.T, config={text = localize('b_redeem'), colour = G.C.WHITE, scale = 0.4}}
        }
    } or (card.ability and card.ability.set == 'Booster') and {
        n=G.UIT.ROOT, config = {ref_table = card, minw = 1.1, maxw = 1.3, padding = 0.1, align = 'bm', colour = G.C.GREEN, shadow = true, r = 0.08, minh = 0.94, func = 'can_open', one_press = true, button = 'open_booster', hover = true}, nodes={
            {n=G.UIT.T, config={text = localize('b_open'), colour = G.C.WHITE, scale = 0.5}}
        }
    } or {
        n=G.UIT.ROOT, config = {ref_table = card, minw = 1.1, maxw = 1.3, padding = 0.1, align = 'bm', colour = G.C.GOLD, shadow = true, r = 0.08, minh = 0.94, func = 'can_buy', one_press = true, button = 'buy_from_shop', hover = true}, nodes={
            {n=G.UIT.T, config={text = localize('b_buy'), colour = G.C.WHITE, scale = 0.5}}
        }
    }

    card.children.buy_button = UIBox{
        definition = t2,
        config = {
            align = "bm",
            offset = {x=0, y=-0.3},
            major = card,
            bond = 'Weak',
            parent = card
        }
    }
    if card.children.buy_button then
        card.children.buy_button.states.visible = false
        if card.children.buy_button.align_to_major then card.children.buy_button:align_to_major() end
        if card.children.buy_button.hard_set_VT then card.children.buy_button:hard_set_VT() end
    end

    if card.ability and card.ability.consumeable then
        ensure_consumable_ready(card)
        local t3 = {
            n=G.UIT.ROOT, config = {id = 'buy_and_use', ref_table = card, minh = 1.1, padding = 0.1, align = 'cr', colour = G.C.RED, shadow = true, r = 0.08, minw = 1.1, func = 'can_buy_and_use', one_press = true, button = 'buy_from_shop', hover = true, focus_args = {type = 'none'}}, nodes={
                {n=G.UIT.B, config = {w=0.1, h=0.6}},
                {n=G.UIT.C, config = {align = 'cm'}, nodes={
                    {n=G.UIT.R, config = {align = 'cm', maxw = 1}, nodes={
                        {n=G.UIT.T, config={text = localize('b_buy'), colour = G.C.WHITE, scale = 0.5}}
                    }},
                    {n=G.UIT.R, config = {align = 'cm', maxw = 1}, nodes={
                        {n=G.UIT.T, config={text = localize('b_and_use'), colour = G.C.WHITE, scale = 0.3}}
                    }},
                }} 
            }
        }
        card.children.buy_and_use_button = UIBox{
            definition = t3,
            config = {
                align = "cr",
                offset = {x=-0.3, y=0},
                major = card,
                bond = 'Weak',
                parent = card
            }
        }
        if card.children.buy_and_use_button then
            card.children.buy_and_use_button.states.visible = false
            if card.children.buy_and_use_button.align_to_major then card.children.buy_and_use_button:align_to_major() end
            if card.children.buy_and_use_button.hard_set_VT then card.children.buy_and_use_button:hard_set_VT() end
        end
    end
end

-- Helper to safely and completely sever all references to a Card and its child objects
local function fully_remove_card(card)
    if not card then return end
    if G.DATASET_MOD and G.DATASET_MOD.captured_boxes then
        G.DATASET_MOD.captured_boxes[card] = nil
    end
    if card.dataset_mimic_dummy then
        fully_remove_card(card.dataset_mimic_dummy)
        card.dataset_mimic_dummy = nil
        card.dataset_mimic_target = nil
    end
    if card.blueprint_sprite_copy then
        pcall(function() card.blueprint_sprite_copy:remove() end)
        card.blueprint_sprite_copy = nil
    end
    if card.area then
        card:remove_from_area()
    end
    if card.children then
        for k, child in pairs(card.children) do
            if child then
                if child.remove then
                    pcall(function() child:remove() end)
                end
                card.children[k] = nil
            end
        end
    end
    pcall(function() card:remove() end)
    card.children = nil
end
ShopGen.fully_remove_card = fully_remove_card

-- Helper to safely remove all cards from a CardArea and clean up their bounding boxes
local function clear_card_area(area)
    if area and area.cards then
        for i = #area.cards, 1, -1 do
            local c = area.cards[i]
            fully_remove_card(c)
        end
        area.cards = {}
    end
    if area and area.highlighted then
        area.highlighted = {}
    end
end
ShopGen.clear_card_area = clear_card_area

-- Sweep through all engine tracking tables and remove any dead objects
local function prune_engine_tables()
    if G.MOVEABLES then
        for i = #G.MOVEABLES, 1, -1 do
            if G.MOVEABLES[i] and G.MOVEABLES[i].REMOVED then
                table.remove(G.MOVEABLES, i)
            end
        end
    end
    if G.I then
        for _, tbl in pairs(G.I) do
            if type(tbl) == 'table' then
                for i = #tbl, 1, -1 do
                    if tbl[i] and tbl[i].REMOVED then
                        table.remove(tbl, i)
                    end
                end
            end
        end
    end
    if G.STAGE_OBJECTS then
        for _, stage_tbl in pairs(G.STAGE_OBJECTS) do
            if type(stage_tbl) == 'table' then
                for i = #stage_tbl, 1, -1 do
                    if stage_tbl[i] and stage_tbl[i].REMOVED then
                        table.remove(stage_tbl, i)
                    end
                end
            end
        end
    end
    if G.ANIMATIONS then
        for i = #G.ANIMATIONS, 1, -1 do
            if G.ANIMATIONS[i] and G.ANIMATIONS[i].REMOVED then
                table.remove(G.ANIMATIONS, i)
            end
        end
    end
end
ShopGen.prune_engine_tables = prune_engine_tables

-- Clear any lingering controller targets pointing to removed objects
local function clear_controller_targets()
    if G.CONTROLLER then
        if G.CONTROLLER.hovering then
            if G.CONTROLLER.hovering.target and G.CONTROLLER.hovering.target.REMOVED then
                G.CONTROLLER.hovering.target = nil
            end
            if G.CONTROLLER.hovering.prev_target and G.CONTROLLER.hovering.prev_target.REMOVED then
                G.CONTROLLER.hovering.prev_target = nil
            end
        end
        if G.CONTROLLER.focused and G.CONTROLLER.focused.target and G.CONTROLLER.focused.target.REMOVED then
            G.CONTROLLER.focused.target = nil
        end
        if G.CONTROLLER.clicked and G.CONTROLLER.clicked.target and G.CONTROLLER.clicked.target.REMOVED then
            G.CONTROLLER.clicked.target = nil
        end
        if G.CONTROLLER.dragging and G.CONTROLLER.dragging.target and G.CONTROLLER.dragging.target.REMOVED then
            G.CONTROLLER.dragging.target = nil
        end
    end
end
ShopGen.clear_controller_targets = clear_controller_targets

-- Clear all event queues across all categories
local function clear_all_event_queues()
    if G.E_MANAGER and G.E_MANAGER.queues then
        for _, q in pairs(G.E_MANAGER.queues) do
            if type(q) == 'table' then
                for i = #q, 1, -1 do
                    if not q[i].no_delete then
                        table.remove(q, i)
                    end
                end
            end
        end
    end
end
ShopGen.clear_all_event_queues = clear_all_event_queues

-- Full sweep & GC cycle
function ShopGen.cleanup_cycle()
    prune_engine_tables()
    clear_controller_targets()
    clear_all_event_queues()
    collectgarbage("collect")
    collectgarbage("collect")
end

local function snap_ui_node(n)
    if not n then return end
    if n.velocity then
        n.velocity.x = 0
        n.velocity.y = 0
        n.velocity.r = 0
        n.velocity.scale = 0
    end
    if n.hard_set_VT then n:hard_set_VT() end
    if n.VT and n.T then
        n.VT.scale = n.T.scale
        n.VT.r = n.T.r
    end
    if n.children then
        for _, c in pairs(n.children) do
            snap_ui_node(c)
        end
    end
end

-- Helper to immediately snap cards and areas to avoid easing/sliding delays and eliminate tilt/rotation
local function snap_card_area(area)
    if not area then return end
    if area.align_cards then area:align_cards() end
    if area.hard_set_VT then area:hard_set_VT() end
    if area.VT and area.T then
        area.VT.scale = area.T.scale
        area.VT.r = area.T.r
    end
    if area.cards then
        for _, card in ipairs(area.cards) do
            -- Keep cards completely upright without ambient wobbling or fan-tilt
            card.T.r = 0
            card.ambient_tilt = 0
            if card.tilt_var then
                card.tilt_var.mx = 0
                card.tilt_var.my = 0
                card.tilt_var.dx = 0
                card.tilt_var.dy = 0
                card.tilt_var.amt = 0
            end
            if card.velocity then
                card.velocity.x = 0
                card.velocity.y = 0
                card.velocity.r = 0
                card.velocity.scale = 0
            end
            card.juice = nil
            card.pinch = {x = false, y = false}
            if card.hard_set_VT then card:hard_set_VT() end
            if card.VT and card.T then
                card.VT.r = 0
                card.VT.scale = card.T.scale
            end

            if card.children then
                for _, child in pairs(card.children) do
                    if child then
                        if child.T then child.T.r = 0 end
                        if child.velocity then
                            child.velocity.x = 0
                            child.velocity.y = 0
                            child.velocity.r = 0
                            child.velocity.scale = 0
                        end
                        if child.align_to_major then child:align_to_major() end
                        if child.hard_set_VT then child:hard_set_VT() end
                        if child.VT and child.T then
                            child.VT.r = 0
                            child.VT.scale = child.T.scale
                        end
                        if child.UIRoot then snap_ui_node(child.UIRoot) end
                    end
                end
            end
        end
    end
end

-- =========================================================
-- JOKER DISPLAY MOD INTEGRATION (OPTIONAL RUNTIME SUPPORT)
-- Randomizes all displayed JokerDisplay values in 30% of frames;
-- cleanly hides all tooltips in the other 70% as if mod is absent.
-- =========================================================
local JD_SUITS = {'Spades', 'Hearts', 'Clubs', 'Diamonds'}
local JD_RANKS = {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace'}
local JD_RANK_IDS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
local JD_POKER_HANDS = {
    'High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush',
    'Full House', 'Four of a Kind', 'Straight Flush', 'Five of a Kind', 'Flush House', 'Flush Five'
}

local function is_joker_display_available()
    return JokerDisplay ~= nil
        and type(JokerDisplay.Definitions) == 'table'
        and type(JokerDisplay.config) == 'table'
        and type(Card) == 'table'
        and type(Card.update_joker_display) == 'function'
end

function G.DATASET_MOD:save_joker_display_config()
    if not is_joker_display_available() then return end
    if self._saved_jd_config == nil then
        self._saved_jd_config = {
            enabled = JokerDisplay.config.enabled,
            hide_by_default = JokerDisplay.config.hide_by_default
        }
    end
end

function G.DATASET_MOD:restore_joker_display_config()
    if not is_joker_display_available() then
        self._saved_jd_config = nil
        return
    end
    if self._saved_jd_config ~= nil then
        JokerDisplay.config.enabled = self._saved_jd_config.enabled
        JokerDisplay.config.hide_by_default = self._saved_jd_config.hide_by_default
        self._saved_jd_config = nil
    end
    if G.jokers and G.jokers.cards then
        for _, card in ipairs(G.jokers.cards) do
            if card then
                card._jd_randomized_override = nil
                if JokerDisplay.config.enabled then
                    if card.joker_display_values then
                        card.joker_display_values.disabled = JokerDisplay.config.hide_by_default or false
                    end
                    pcall(function() card:update_joker_display(true, true, "gdcv_restore") end)
                elseif card.joker_display_remove then
                    pcall(function() card:joker_display_remove() end)
                end
            end
        end
    end
end

local function safe_localize(arg, category, fallback)
    if type(localize) ~= 'function' then return fallback or tostring(arg) end
    local ok, res = pcall(localize, arg, category)
    if ok and res ~= nil and res ~= 'ERROR' then return res end
    return fallback or tostring(arg)
end

local function pick_jd_chips()
    local r = math.random()
    if r < 0.12 then return 0
    elseif r < 0.52 then return math.random(4, 60)
    elseif r < 0.80 then return math.random(65, 350)
    elseif r < 0.95 then return math.random(360, 2500)
    else return ({12500, 48000, 125000, 1.4e6, 2.5e11})[math.random(5)]
    end
end

local function pick_jd_mult()
    local r = math.random()
    if r < 0.12 then return 0
    elseif r < 0.57 then return math.random(2, 25)
    elseif r < 0.85 then return math.random(26, 150)
    elseif r < 0.95 then return math.random(155, 600)
    else return ({1200, 5400, 24000, 1.5e6})[math.random(4)]
    end
end

local function pick_jd_xmult()
    local r = math.random()
    if r < 0.14 then return 1
    elseif r < 0.44 then return math.random(2, 12)
    elseif r < 0.82 then
        if math.random() < 0.5 then
            return 1 + math.random(1, 35) * 0.25
        else
            return math.random(11, 150) / 10
        end
    elseif r < 0.94 then
        return math.random(20, 400) + ({0, 0.25, 0.5, 0.75})[math.random(4)]
    else
        return ({1.2e11, 3.8e12, 7.4e15, 1.9e24, 6.2e42})[math.random(5)]
    end
end

local function pick_jd_dollars()
    local r = math.random()
    if r < 0.20 then return 0
    elseif r < 0.75 then return math.random(1, 20)
    elseif r < 0.95 then return math.random(21, 75)
    else return math.random(80, 250)
    end
end

local function randomize_jd_round_state()
    if not G.GAME then return end
    G.GAME.probabilities = G.GAME.probabilities or {normal = 1}
    G.GAME.probabilities.normal = ({1, 1, 1, 2, 4, 8})[math.random(6)]

    G.GAME.current_round = G.GAME.current_round or {}
    G.GAME.current_round.hands_left = math.random(0, 5)
    G.GAME.current_round.hands_played = math.random(0, 4)
    G.GAME.current_round.discards_left = math.random(0, 5)
    G.GAME.current_round.discards_used = math.random(0, 3)

    G.GAME.current_round.ancient_card = G.GAME.current_round.ancient_card or {}
    G.GAME.current_round.ancient_card.suit = JD_SUITS[math.random(#JD_SUITS)]

    G.GAME.current_round.castle_card = G.GAME.current_round.castle_card or {}
    G.GAME.current_round.castle_card.suit = JD_SUITS[math.random(#JD_SUITS)]

    local mail_idx = math.random(#JD_RANKS)
    G.GAME.current_round.mail_card = G.GAME.current_round.mail_card or {}
    G.GAME.current_round.mail_card.rank = JD_RANKS[mail_idx]
    G.GAME.current_round.mail_card.id = JD_RANK_IDS[mail_idx]

    local idol_idx = math.random(#JD_RANKS)
    G.GAME.current_round.idol_card = G.GAME.current_round.idol_card or {}
    G.GAME.current_round.idol_card.rank = JD_RANKS[idol_idx]
    G.GAME.current_round.idol_card.id = JD_RANK_IDS[idol_idx]
    G.GAME.current_round.idol_card.suit = JD_SUITS[math.random(#JD_SUITS)]
end

local function apply_jd_override_to_card(card)
    local ov = card and card._jd_randomized_override
    if not ov then return end
    if ov.sell_cost ~= nil then
        card.sell_cost = ov.sell_cost
    end
    if card.ability and ov.ability then
        for k, v in pairs(ov.ability) do
            if type(card.ability[k]) == type(v) then
                card.ability[k] = v
            end
        end
        if type(card.ability.extra) == 'table' and ov.ability_extra then
            for k, v in pairs(ov.ability_extra) do
                if type(card.ability.extra[k]) == type(v) then
                    card.ability.extra[k] = v
                end
            end
        end
    end
    local jvals = card.joker_display_values
    if jvals and ov.jvals then
        for k, v in pairs(ov.jvals) do
            if k == 'active' then
                if type(jvals.active) == 'boolean' and ov.jvals.active_bool ~= nil then
                    jvals.active = ov.jvals.active_bool
                elseif type(jvals.active) == 'string' and ov.jvals.active_str ~= nil then
                    jvals.active = ov.jvals.active_str
                end
            elseif k == 'dollars' then
                if type(jvals.dollars) == 'number' and type(v) == 'number' then
                    jvals.dollars = v
                elseif type(jvals.dollars) == 'string' and ov.jvals.trading_dollars ~= nil then
                    jvals.dollars = ov.jvals.trading_dollars
                end
            elseif k ~= 'active_bool' and k ~= 'active_str' and k ~= 'trading_dollars' then
                if type(jvals[k]) == type(v) then
                    jvals[k] = v
                end
            end
        end
    end
end

local function ensure_joker_display_hook()
    if not is_joker_display_available() then return end
    if Card._gdcv_jd_calc_hooked or type(Card.calculate_joker_display) ~= 'function' then return end
    Card._gdcv_jd_calc_hooked = true
    local orig_calc_jd = Card.calculate_joker_display
    function Card:calculate_joker_display(custom_parent)
        local ret = orig_calc_jd(self, custom_parent)
        if not custom_parent and self._jd_randomized_override then
            apply_jd_override_to_card(self)
        end
        return ret
    end
end

local function build_jd_override_for_card(card)
    local c_key = card.config and card.config.center and card.config.center.key or ""
    local active_bool = (math.random() < 0.55)
    local chips = pick_jd_chips()
    local mult = pick_jd_mult()
    local x_mult = pick_jd_xmult()
    local dollars = pick_jd_dollars()
    local hand_name = JD_POKER_HANDS[math.random(#JD_POKER_HANDS)]
    local suit_a = JD_SUITS[math.random(#JD_SUITS)]
    local suit_c = JD_SUITS[math.random(#JD_SUITS)]
    local rank_m = JD_RANKS[math.random(#JD_RANKS)]
    local rank_i = JD_RANKS[math.random(#JD_RANKS)]
    local suit_i = JD_SUITS[math.random(#JD_SUITS)]

    local active_text_loc = safe_localize(active_bool and 'jdis_active' or 'jdis_inactive', nil, active_bool and "Active!" or "Inactive")
    local active_str = active_text_loc
    if c_key == 'j_invisible' then
        local req = math.random(2, 3)
        local cur = active_bool and req or math.random(0, req - 1)
        active_str = active_bool and safe_localize('jdis_active', nil, "Active!") or string.format("%d/%d", cur, req)
    end

    local trading_dollars = active_bool and ("+$" .. tostring(math.random(1, 10))) or "-"
    local rem = math.random(0, 5)
    local loyalty_str = (rem == 0)
        and safe_localize('jdis_active', nil, "Active!")
        or safe_localize({type = 'variable', key = 'jdis_loyalty_remaining', vars = {rem}}, nil, tostring(rem) .. " remaining")
    local num = ({1, 1, 1, 2, 4, 8})[math.random(6)]
    local den = ({2, 3, 4, 5, 6, 8, 10, 15, 1000})[math.random(9)]
    local odds_str = safe_localize({type = 'variable', key = 'jdis_odds', vars = {num, den}}, nil, string.format("%d in %d", num, den))
    local idol_str = safe_localize({
        type = 'variable',
        key = 'jdis_rank_of_suit',
        vars = {safe_localize(rank_i, 'ranks', rank_i), safe_localize(suit_i, 'suits_plural', suit_i)}
    }, nil, rank_i .. " of " .. suit_i)

    local inv_req = math.random(2, 3)
    local inv_cur = active_bool and inv_req or math.random(0, inv_req - 1)
    local perish_tally = math.random(1, (G.GAME and G.GAME.perishable_rounds) or 5)

    return {
        sell_cost = math.random(1, 99),
        small = (math.random() < 0.20),
        ability = {
            mult = mult,
            x_mult = x_mult,
            extra = (math.random() < 0.5) and chips or mult,
            driver_tally = math.random(0, 32),
            caino_xmult = x_mult,
            invis_rounds = inv_cur,
            loyalty_remaining = rem,
            perish_tally = perish_tally,
            to_do_poker_hand = hand_name,
            stone_tally = math.random(0, 20),
            steel_tally = math.random(0, 20),
            nine_tally = math.random(0, 20),
            h_size = math.random(1, 5),
            discards = math.random(0, 5)
        },
        ability_extra = {
            chips = chips,
            chip_mod = math.random(1, 30),
            mult = mult,
            Xmult = x_mult,
            dollars = dollars,
            h_size = math.random(1, 5),
            discards = math.random(0, 23),
            yorick_discards = math.random(0, 23)
        },
        jvals = {
            chips = chips,
            mult = mult,
            x_mult = x_mult,
            count = math.random(0, 6),
            dollars = dollars,
            trading_dollars = trading_dollars,
            active = active_bool,
            active_bool = active_bool,
            active_str = active_str,
            active_text = active_text_loc,
            is_active = active_bool,
            start_count = math.random(1, 8),
            yorick_discards = math.random(0, 23),
            ancient_card_suit = safe_localize(suit_a, 'suits_plural', suit_a),
            castle_card_suit = safe_localize(suit_c, 'suits_plural', suit_c),
            mail_card_rank = safe_localize(rank_m, 'ranks', rank_m),
            to_do_poker_hand = safe_localize(hand_name, 'poker_hands', hand_name),
            idol_card = idol_str,
            loyalty_text = loyalty_str,
            odds = odds_str
        }
    }
end

-- Populate the top Joker Bar (G.jokers) with 0 to 9 jokers
function G.DATASET_MOD:populate_joker_bar()
    if not G.jokers then return end

    self.captured_boxes = self.captured_boxes or setmetatable({}, {__mode = "k"})
    clear_card_area(G.jokers)

    local jd_present = is_joker_display_available()
    local show_jd = false
    if jd_present then
        self:save_joker_display_config()
        ensure_joker_display_hook()
        -- 30% of training frames show randomized JokerDisplay tooltips; 70% completely hide them
        show_jd = (math.random() < 0.30)
        JokerDisplay.config.enabled = show_jd
        JokerDisplay.config.hide_by_default = false
        if show_jd then
            randomize_jd_round_state()
        end
    end

    -- Distribution: 0 to 9 jokers, with 7, 8, 9 appearing less commonly
    -- 0-6: 12% each (84% total)
    -- 7-9: ~5.33% each (16% total)
    local num_jokers
    local r = math.random()
    if r < 0.84 then
        num_jokers = math.random(0, 6)
    else
        num_jokers = math.random(7, 9)
    end

    G.jokers.config.card_limit = math.max(5, num_jokers)

    -- Debuff simulation (e.g. boss blinds Crimson Heart / Verdant Leaf)
    -- 10% chance that 1 or more jokers on the bar are debuffed
    local has_debuffs = (math.random() < 0.10) and (num_jokers > 0)
    local debuff_indices = {}
    if has_debuffs then
        -- 50% chance 1 random joker debuffed, 30% chance 2 debuffed, 20% chance all debuffed (Verdant Leaf)
        local d_roll = math.random()
        if d_roll < 0.50 then
            debuff_indices[math.random(1, num_jokers)] = true
        elseif d_roll < 0.80 then
            local idx1 = math.random(1, num_jokers)
            local idx2 = math.random(1, num_jokers)
            debuff_indices[idx1] = true
            debuff_indices[idx2] = true
        else
            for k = 1, num_jokers do
                debuff_indices[k] = true
            end
        end
    end

    -- 50% chance for each shop generation to boost Blueprint / Brainstorm spawn odds by 10x
    local boost_blueprint = (math.random() < 0.50)

    for i = 1, num_jokers do
        local card = self:create_random_joker_card(G.jokers, nil, true, nil, nil, boost_blueprint)
        if card then
            if debuff_indices[i] then
                if card.set_debuff then
                    card:set_debuff(true)
                else
                    card.debuff = true
                end
            end
            if jd_present then
                if show_jd then
                    card._jd_randomized_override = build_jd_override_for_card(card)
                    card.joker_display_values = {
                        disabled = false,
                        small = card._jd_randomized_override.small
                    }
                    apply_jd_override_to_card(card)
                else
                    card._jd_randomized_override = nil
                    card.joker_display_values = { disabled = true, small = false }
                    card.joker_display_installed = true
                end
            end
            G.jokers:emplace(card)
        end
    end

    if G.jokers.align_cards then
        G.jokers:align_cards()
    end

    -- Synchronize JokerDisplay UIs after all jokers (including Blueprint/Brainstorm neighbors) are emplaced
    if jd_present and G.jokers.cards then
        for _, card in ipairs(G.jokers.cards) do
            if show_jd then
                if card.ability and card.ability.perishable and card._jd_randomized_override and card._jd_randomized_override.ability then
                    card.ability.perish_tally = card._jd_randomized_override.ability.perish_tally
                end
                pcall(function() card:update_joker_display(true, true, "gdcv_setup") end)
                if card.children then
                    if card.children.joker_display and card.children.joker_display.recalculate then card.children.joker_display:recalculate(true) end
                    if card.children.joker_display_small and card.children.joker_display_small.recalculate then card.children.joker_display_small:recalculate(true) end
                    if card.children.joker_display_debuff and card.children.joker_display_debuff.recalculate then card.children.joker_display_debuff:recalculate(true) end
                    if card.children.joker_display_perishable and card.children.joker_display_perishable.recalculate then card.children.joker_display_perishable:recalculate() end
                    if card.children.joker_display_rental and card.children.joker_display_rental.recalculate then card.children.joker_display_rental:recalculate() end
                end
            elseif card.joker_display_remove then
                pcall(function() card:joker_display_remove() end)
            end
        end
    end
end

-- =========================================================
-- POPULATE CONSUMABLES AREA (NORMAL & PERKEO MODES)
-- =========================================================
function G.DATASET_MOD:populate_consumables_area()
    if not G.consumeables then return end

    self.captured_boxes = self.captured_boxes or setmetatable({}, {__mode = "k"})
    clear_card_area(G.consumeables)

    local _, tarots, spectrals, planets = self:get_joker_pools()

    -- 10% chance for Perkeo mode
    local is_perkeo = (math.random() < 0.10)
    self.is_perkeo_active = is_perkeo

    -- Slot limit under deck / vouchers (1-3), biased toward multi-slot consumables
    local r_slot = math.random()
    local slot_limit = (r_slot < 0.20) and 1 or ((r_slot < 0.60) and 2 or 3)

    if not is_perkeo then
        -- Normal Circumstances (90% of the time):
        -- 0..slot_limit cards, with extra weight on full 3-card non-negative Crystal Ball / Magic Deck layouts
        local num_cards = (slot_limit == 3 and math.random() < 0.50) and 3 or math.random(0, slot_limit)
        G.consumeables.config.card_limit = slot_limit

        -- 10% chance for Showman effect allowing duplicates
        local has_showman = (math.random() < 0.10)
        local used_centers = {}

        local types = {'Tarot', 'Spectral', 'Planet'}
        for i = 1, num_cards do
            local chosen_type = types[math.random(#types)]
            local pool = (chosen_type == 'Tarot' and tarots) or (chosen_type == 'Spectral' and spectrals) or planets

            local chosen_center = nil
            if pool and #pool > 0 then
                if has_showman then
                    chosen_center = pool[math.random(#pool)]
                else
                    for attempt = 1, 20 do
                        local cand = pool[math.random(#pool)]
                        if not used_centers[cand.key] then
                            chosen_center = cand
                            break
                        end
                    end
                    chosen_center = chosen_center or pool[math.random(#pool)]
                    used_centers[chosen_center.key] = true
                end
            end

            local card = self:create_random_consumable_card(G.consumeables, chosen_type, chosen_center)
            if card then
                -- Sometimes a consumable can be Negative even without full slots (e.g. from Perkeo, Sixth Sense, etc.)
                if math.random() < 0.20 then
                    card:set_edition({negative = true}, true, true)
                end
                G.consumeables:emplace(card)
            end
        end
    else
        -- Perkeo Mode (10% of the time):
        -- Up to configurable max consumables (self.config.perkeo_max_consumables), weighted towards having
        -- the same type appear multiple times, and anything exceeding the slot limit is negative!
        local max_limit = math.max(slot_limit + 1, self.config.perkeo_max_consumables or 10)
        local num_cards = math.random(slot_limit + 1, max_limit)
        G.consumeables.config.card_limit = math.max(slot_limit, num_cards)

        -- Pick the primary duplicated consumable
        local types = {'Tarot', 'Spectral', 'Planet'}
        local dup_type = types[math.random(#types)]
        local dup_pool = (dup_type == 'Tarot' and tarots) or (dup_type == 'Spectral' and spectrals) or planets
        local dup_center = dup_pool and dup_pool[math.random(#dup_pool)]

        for i = 1, num_cards do
            local card_type = dup_type
            local card_center = dup_center

            -- 75% chance primary duplicate, 25% chance another random consumable
            if math.random() > 0.75 or not dup_center then
                card_type = types[math.random(#types)]
                local pool = (card_type == 'Tarot' and tarots) or (card_type == 'Spectral' and spectrals) or planets
                card_center = pool and pool[math.random(#pool)]
            end

            local card = self:create_random_consumable_card(G.consumeables, card_type, card_center)
            if card then
                -- Any card exceeding the consumable slot limit is Negative
                if i > slot_limit then
                    card:set_edition({negative = true}, true, true)
                end
                G.consumeables:emplace(card)
            end
        end
    end

    -- Randomly shuffle the order of consumables in G.consumeables
    if G.consumeables.cards and #G.consumeables.cards > 1 then
        for i = #G.consumeables.cards, 2, -1 do
            local j = math.random(i)
            G.consumeables.cards[i], G.consumeables.cards[j] = G.consumeables.cards[j], G.consumeables.cards[i]
        end
    end

    if G.consumeables.align_cards then
        G.consumeables:align_cards()
    end
end

-- =========================================================
-- CARD SELECTION & BUTTON STATE SIMULATION
-- =========================================================
local BUTTON_STATES = {"Normal", "Hovered", "Clicked", "Grayed Out"}

local function collect_button_nodes(node, list)
    if not node then return end
    if node.config and (node.config.button or (node.config.func and (
        node.config.func:find('buy') or node.config.func:find('sell') or 
        node.config.func:find('use') or node.config.func:find('select') or 
        node.config.func:find('redeem') or node.config.func:find('open')
    ))) then
        table.insert(list, node)
        return
    end
    if node.children then
        for _, child in pairs(node.children) do
            collect_button_nodes(child, list)
        end
    end
end

local function collect_text_nodes(node, list)
    if not node then return end
    if node.UIT == G.UIT.T or (node.config and node.config.text_drawable) then
        table.insert(list, node)
    end
    if node.children then
        for _, child in pairs(node.children) do
            collect_text_nodes(child, list)
        end
    end
end

local function get_card_button_nodes(card)
    local buttons = {}
    if card.children then
        if card.children.buy_button and card.children.buy_button.UIRoot then
            collect_button_nodes(card.children.buy_button.UIRoot, buttons)
        end
        if card.children.buy_and_use_button and card.children.buy_and_use_button.UIRoot then
            collect_button_nodes(card.children.buy_and_use_button.UIRoot, buttons)
        end
        if card.children.use_button and card.children.use_button.UIRoot then
            collect_button_nodes(card.children.use_button.UIRoot, buttons)
        end
    end
    return buttons
end

local function apply_state_to_button(card, btn_node, state_name)
    local default_col = G.C.ORANGE
    local default_action = btn_node.config.button or 'buy_from_shop'

    if btn_node.config.button == 'sell_card' or (btn_node.config.func and btn_node.config.func:find('sell')) then
        default_col = G.C.GREEN
        default_action = 'sell_card'
    elseif btn_node.config.id == 'buy_and_use' or (btn_node.config.func and btn_node.config.func:find('buy_and_use')) then
        default_col = G.C.RED
        default_action = 'buy_from_shop'
    elseif btn_node.config.button == 'buy_from_shop' or (btn_node.config.func and btn_node.config.func:find('buy')) then
        default_col = G.C.ORANGE
        default_action = 'buy_from_shop'
    elseif btn_node.config.func == 'can_use_consumeable' or (btn_node.config.button == 'use_card' and card.ability and card.ability.consumeable) then
        default_col = G.C.RED
        default_action = 'use_card'
    elseif btn_node.config.func == 'can_select_card' or btn_node.config.button == 'use_card' then
        default_col = G.C.GREEN
        default_action = 'use_card'
    elseif btn_node.config.func == 'can_redeem' or btn_node.config.button == 'redeem_from_shop' then
        default_col = G.C.GREEN
        default_action = 'redeem_from_shop'
    elseif btn_node.config.func == 'can_open' or btn_node.config.button == 'open_booster' then
        default_col = G.C.GREEN
        default_action = 'open_booster'
    end

    -- Clear func so Balatro update callbacks do not overwrite our state
    btn_node.config.func = nil

    local text_nodes = {}
    collect_text_nodes(btn_node, text_nodes)

    if state_name == "Normal" then
        btn_node.config.colour = default_col
        btn_node.config.button = default_action
        btn_node.config.shadow = true
        btn_node.states.hover.is = false
        btn_node.last_clicked = nil
        for _, tn in ipairs(text_nodes) do
            tn.config.colour = (default_action == 'sell_card') and G.C.UI.TEXT_LIGHT or G.C.WHITE
        end
    elseif state_name == "Hovered" then
        btn_node.config.colour = default_col
        btn_node.config.button = default_action
        btn_node.config.shadow = true
        btn_node.config.hover = true
        btn_node.states.hover.is = true
        btn_node.last_clicked = nil
        for _, tn in ipairs(text_nodes) do
            tn.config.colour = (default_action == 'sell_card') and G.C.UI.TEXT_LIGHT or G.C.WHITE
        end
    elseif state_name == "Clicked" then
        btn_node.config.colour = default_col
        btn_node.config.button = default_action
        btn_node.config.shadow = true
        btn_node.config.hover = true
        btn_node.states.hover.is = true
        btn_node.last_clicked = (G.TIMERS and G.TIMERS.REAL or 0) + 100
        for _, tn in ipairs(text_nodes) do
            tn.config.colour = (default_action == 'sell_card') and G.C.UI.TEXT_LIGHT or G.C.WHITE
        end
    elseif state_name == "Grayed Out" then
        btn_node.config.colour = G.C.UI.BACKGROUND_INACTIVE
        btn_node.config.button = nil
        btn_node.config.shadow = true
        btn_node.states.hover.is = false
        btn_node.last_clicked = nil
        for _, tn in ipairs(text_nodes) do
            tn.config.colour = G.C.UI.TEXT_INACTIVE
        end
    end
end

local function apply_random_button_states_to_card(card)
    local buttons = get_card_button_nodes(card)
    if #buttons == 0 then return end

    -- Reset all buttons to Normal initially
    for _, btn in ipairs(buttons) do
        apply_state_to_button(card, btn, "Normal")
    end

    -- If 1 button: pick among {"Normal", "Hovered", "Clicked", "Grayed Out"} (25% each)
    -- If multiple buttons: pick a random one to receive the state; keep the other Normal
    local target_btn = buttons[math.random(#buttons)]
    local chosen_state = BUTTON_STATES[math.random(#BUTTON_STATES)]
    apply_state_to_button(card, target_btn, chosen_state)
end

local function select_card_in_area(area, card)
    if not area or not card then return end

    -- Unhighlight other cards in this area
    if area.highlighted then
        for i = #area.highlighted, 1, -1 do
            local other = area.highlighted[i]
            if other ~= card then
                other:highlight(false)
                if other.children and other.children.buy_button then
                    other.children.buy_button.states.visible = false
                end
                if other.children and other.children.buy_and_use_button then
                    other.children.buy_and_use_button.states.visible = false
                end
            end
            table.remove(area.highlighted, i)
        end
    end
    area.highlighted = {card}

    if not card._gdcv_cost_randomized then
        randomize_card_costs(card)
    end
    ensure_consumable_ready(card)
    card:highlight(true)

    if area.align_cards then
        area:align_cards()
    end

    if card.children then
        if card.children.buy_button then
            card.children.buy_button.states.visible = true
            if card.children.buy_button.align_to_major then card.children.buy_button:align_to_major() end
        end
        if card.children.buy_and_use_button then
            card.children.buy_and_use_button.states.visible = true
            if card.children.buy_and_use_button.align_to_major then card.children.buy_and_use_button:align_to_major() end
        end
        if card.children.use_button then
            card.children.use_button.states.visible = true
            if card.children.use_button.align_to_major then card.children.use_button:align_to_major() end
        end
    end

    apply_random_button_states_to_card(card)
end

function G.DATASET_MOD:apply_shop_card_selections()
    local chance_str = tostring(self.config and self.config.card_selection_chance or '50%')
    local chance = 0.50
    if chance_str == '0%' or chance_str == '0' then
        chance = 0.0
    elseif chance_str == '100%' or chance_str == '100' then
        chance = 1.0
    else
        chance = 0.50
    end

    local regions = {
        G.jokers,
        G.consumeables,
        (self.shop_phase == 'pack') and G.pack_cards or G.shop_jokers
    }

    for _, area in ipairs(regions) do
        if area and area.cards and #area.cards > 0 then
            -- Unhighlight all cards in this area first
            if area.highlighted then
                for i = #area.highlighted, 1, -1 do
                    local c = area.highlighted[i]
                    if c then
                        c:highlight(false)
                        if c.children and c.children.buy_button then
                            c.children.buy_button.states.visible = false
                        end
                        if c.children and c.children.buy_and_use_button then
                            c.children.buy_and_use_button.states.visible = false
                        end
                    end
                    table.remove(area.highlighted, i)
                end
            end
            for _, c in ipairs(area.cards) do
                c.highlighted = false
                if c.children and c.children.buy_button then
                    c.children.buy_button.states.visible = false
                end
                if c.children and c.children.buy_and_use_button then
                    c.children.buy_and_use_button.states.visible = false
                end
            end

            -- Roll chance to select a card in this region
            if chance > 0 and (chance >= 1.0 or math.random() < chance) then
                local chosen_card = area.cards[math.random(#area.cards)]
                select_card_in_area(area, chosen_card)
            end
        end
    end
end



-- =========================================================
-- SETUP SHOP SCREEN
-- =========================================================
function G.DATASET_MOD:setup_shop_screen()
    if not G.GAME then return end

    if G.pack_cards then
        clear_card_area(G.pack_cards)
    end
    if G.booster_pack then
        G.booster_pack:remove()
        G.booster_pack = nil
        G.pack_cards = nil
        prune_engine_tables()
    end

    -- Randomize deck skin collabs so Magic Trick playing cards in shop cover all skins
    if self.randomize_deck_appearance then
        self:randomize_deck_appearance()
    end

    -- Cycle shop slots: 2 (no voucher) -> 3 (overstock tier 1) -> 4 (overstock tier 2)
    self.shop_slot_cycle = ((self.shop_slot_cycle or 1) % 3) + 2
    local num_slots = self.shop_slot_cycle

    G.GAME.shop = G.GAME.shop or {}
    G.GAME.shop.joker_max = num_slots

    -- Build G.shop UIBox ONCE to prevent leaking G.SHOP_SIGN, AnimatedSprites, and duplicate CardAreas
    if not G.shop then
        local saved_joker_max = G.GAME.shop.joker_max
        G.GAME.shop.joker_max = 2
        if G.UIDEF and G.UIDEF.shop then
            G.shop = UIBox{
                definition = G.UIDEF.shop(),
                config = {align = 'tmi', offset = {x = 0, y = -5.3}, major = G.hand, bond = 'Weak'}
            }
        end
        G.GAME.shop.joker_max = saved_joker_max
    end

    -- Update slot limits dynamically using vanilla base game sizing so cards are never squished
    if G.shop_jokers then
        G.shop_jokers.config.card_limit = num_slots
        G.shop_jokers.T.w = num_slots * 1.02 * G.CARD_W
    end
    if G.shop and G.shop.recalculate then
        G.shop:recalculate()
    end

    -- Instantly snap shop offset, alignment, and entire internal UI tree so position shifts happen immediately without sliding lag
    if G.shop then
        G.shop.alignment.offset.y = -5.3
        if G.shop.align_to_major then G.shop:align_to_major() end
        if G.shop.hard_set_VT then G.shop:hard_set_VT() end
        if G.shop.velocity then G.shop.velocity.x = 0; G.shop.velocity.y = 0 end
        if G.shop.UIRoot then
            local function snap_node(n)
                if not n then return end
                if n.hard_set_VT then n:hard_set_VT() end
                if n.children then
                    for _, child in pairs(n.children) do
                        snap_node(child)
                    end
                end
            end
            snap_node(G.shop.UIRoot)
        end
    end
    G.STATE = G.STATES.SHOP
    G.STATE_COMPLETE = true
    if G.SHOP_SIGN then
        G.SHOP_SIGN.alignment.offset.y = 0
        if G.SHOP_SIGN.align_to_major then G.SHOP_SIGN:align_to_major() end
        snap_ui_node(G.SHOP_SIGN)
    end

    if G.shop_jokers then
        clear_card_area(G.shop_jokers)
        G.shop_jokers.config.card_limit = num_slots
        G.shop_jokers.config.temp_limit = num_slots

        -- 10% chance for Showman effect (allows duplicate cards in shop)
        local has_showman = (math.random() < 0.10)
        local used_shop_keys = {}

        local jokers, tarots, spectrals, planets = self:get_joker_pools()

        -- Populate slots: 60% Joker, 15% Playing Card (Magic Trick / Illusion voucher), 10% Tarot, 8% Spectral, 7% Planet
        -- Note: Legendary jokers do NOT appear in the shop
        local non_leg_jokers = ShopGen.get_all_jokers(false)
        for i = 1, num_slots do
            local roll = math.random()
            local card = nil
            local card_type = 'Joker'

            local slot_x = G.shop_jokers.T.x + (G.shop_jokers.T.w - G.shop_jokers.card_w) * ((i - 1) / math.max(num_slots - 1, 1)) + 0.5 * (G.shop_jokers.card_w - G.CARD_W)
            local slot_y = G.shop_jokers.T.y + G.shop_jokers.T.h / 2 - G.CARD_H / 2

            if roll < 0.60 then
                card_type = 'Joker'
                local chosen_center = nil
                if #non_leg_jokers > 0 then
                    if has_showman then
                        chosen_center = non_leg_jokers[math.random(#non_leg_jokers)]
                    else
                        for attempt = 1, 25 do
                            local cand = non_leg_jokers[math.random(#non_leg_jokers)]
                            if not used_shop_keys[cand.key] then
                                chosen_center = cand; break
                            end
                        end
                        chosen_center = chosen_center or non_leg_jokers[math.random(#non_leg_jokers)]
                        used_shop_keys[chosen_center.key] = true
                    end
                end
                card = self:create_random_joker_card(G.shop_jokers, chosen_center, false, slot_x, slot_y)
            elseif roll < 0.75 then
                card_type = 'PlayingCard'
                card = self:create_random_playing_card(G.shop_jokers, slot_x, slot_y)
            else
                local pool
                if roll < 0.85 then
                    card_type = 'Tarot'; pool = tarots
                elseif roll < 0.93 then
                    card_type = 'Spectral'; pool = spectrals
                else
                    card_type = 'Planet'; pool = planets
                end

                local chosen_center = nil
                if pool and #pool > 0 then
                    if has_showman then
                        chosen_center = pool[math.random(#pool)]
                    else
                        for attempt = 1, 25 do
                            local cand = pool[math.random(#pool)]
                            if not used_shop_keys[cand.key] then
                                chosen_center = cand; break
                            end
                        end
                        chosen_center = chosen_center or pool[math.random(#pool)]
                        used_shop_keys[chosen_center.key] = true
                    end
                end
                card = self:create_random_consumable_card(G.shop_jokers, card_type, chosen_center, slot_x, slot_y)
            end

            if card then
                table.insert(G.shop_jokers.cards, card)
                card:set_card_area(G.shop_jokers)
            end
        end

        G.shop_jokers:set_ranks()
        if G.shop_jokers.align_cards then
            G.shop_jokers:align_cards()
        end

        snap_card_area(G.shop_jokers)

        if G.shop_jokers.cards then
            for _, c in ipairs(G.shop_jokers.cards) do
                attach_shop_card_ui_sync(c)
            end
        end
    end

    -- Populate Vouchers (G.shop_vouchers):
    -- Distribution: 90% 1 voucher, 3% 2 vouchers, 2% 3-4 vouchers, 5% 0 vouchers
    local v_roll = math.random()
    local num_vouchers = 1
    if v_roll < 0.90 then
        num_vouchers = 1
    elseif v_roll < 0.93 then
        num_vouchers = 2
    elseif v_roll < 0.95 then
        num_vouchers = math.random(3, 4)
    else
        num_vouchers = 0
    end

    if G.shop_vouchers then
        clear_card_area(G.shop_vouchers)
        G.shop_vouchers.config.card_limit = math.max(1, num_vouchers)
        G.shop_vouchers.config.temp_limit = math.max(1, num_vouchers)

        if num_vouchers > 0 then
            local all_vouchers = ShopGen.get_all_vouchers()
            local used_v_keys = {}

            for i = 1, num_vouchers do
                local chosen_v = nil
                for attempt = 1, 30 do
                    local cand = all_vouchers[math.random(#all_vouchers)]
                    if not used_v_keys[cand.key] then
                        chosen_v = cand
                        break
                    end
                end
                chosen_v = chosen_v or all_vouchers[math.random(#all_vouchers)]
                used_v_keys[chosen_v.key] = true

                local slot_x
                if num_vouchers == 1 then
                    slot_x = G.shop_vouchers.T.x + 0.5 * (G.shop_vouchers.T.w - G.CARD_W)
                else
                    slot_x = G.shop_vouchers.T.x + (G.shop_vouchers.T.w - G.CARD_W) * ((i - 1) / (num_vouchers - 1))
                end
                local slot_y = G.shop_vouchers.T.y + G.shop_vouchers.T.h / 2 - G.CARD_H / 2

                local card = self:create_random_voucher_card(G.shop_vouchers, chosen_v, slot_x, slot_y)
                if card then
                    table.insert(G.shop_vouchers.cards, card)
                    card:set_card_area(G.shop_vouchers)
                end
            end

            G.shop_vouchers:set_ranks()
            for k, c in ipairs(G.shop_vouchers.cards) do
                c.T.r = 0
                c.ambient_tilt = 0
                if num_vouchers == 1 then
                    c.T.x = G.shop_vouchers.T.x + 0.5 * (G.shop_vouchers.T.w - G.CARD_W)
                else
                    c.T.x = G.shop_vouchers.T.x + (G.shop_vouchers.T.w - G.CARD_W) * ((k - 1) / (num_vouchers - 1))
                end
                c.T.y = G.shop_vouchers.T.y + G.shop_vouchers.T.h / 2 - G.CARD_H / 2
            end
            snap_card_area(G.shop_vouchers)

            if G.shop_vouchers.cards then
                for _, c in ipairs(G.shop_vouchers.cards) do
                    attach_shop_card_ui_sync(c)
                end
            end
        end
    end

    -- Populate Booster Packs (G.shop_booster):
    -- Distribution: 49.9% 2 packs, 49.9% 1 pack, 0.2% empty (0 packs)
    local p_roll = math.random()
    local num_packs = 2
    if p_roll < 0.499 then
        num_packs = 2
    elseif p_roll < 0.998 then
        num_packs = 1
    else
        num_packs = 0
    end

    if G.shop_booster then
        clear_card_area(G.shop_booster)
        G.shop_booster.config.card_limit = 2
        G.shop_booster.config.temp_limit = 2

        if G.shop_booster._gdcv_align_hooked then
            G.shop_booster.align_cards = CardArea.align_cards
            G.shop_booster._gdcv_align_hooked = nil
        end

        if num_packs > 0 then
            local all_boosters = ShopGen.get_all_boosters()
            local used_p_keys = {}

            if num_packs == 2 then
                for i = 1, 2 do
                    local chosen_b = nil
                    for attempt = 1, 30 do
                        local cand = all_boosters[math.random(#all_boosters)]
                        if not used_p_keys[cand.key] then
                            chosen_b = cand
                            break
                        end
                    end
                    chosen_b = chosen_b or all_boosters[math.random(#all_boosters)]
                    used_p_keys[chosen_b.key] = true

                    local slot_x = (i == 1) and G.shop_booster.T.x or (G.shop_booster.T.x + G.shop_booster.T.w - 1.27 * G.CARD_W)
                    local slot_y = G.shop_booster.T.y + G.shop_booster.T.h / 2 - (1.27 * G.CARD_H) / 2

                    local card = self:create_random_booster_card(G.shop_booster, chosen_b, slot_x, slot_y, i)
                    if card then
                        table.insert(G.shop_booster.cards, card)
                        card:set_card_area(G.shop_booster)
                    end
                end
            elseif num_packs == 1 then
                local chosen_b = all_boosters[math.random(#all_boosters)]
                local slot_x = G.shop_booster.T.x + 0.5 * (G.shop_booster.T.w - 1.27 * G.CARD_W)
                local slot_y = G.shop_booster.T.y + G.shop_booster.T.h / 2 - (1.27 * G.CARD_H) / 2

                local card = self:create_random_booster_card(G.shop_booster, chosen_b, slot_x, slot_y, 1)
                if card then
                    table.insert(G.shop_booster.cards, card)
                    card:set_card_area(G.shop_booster)
                end
            end

            G.shop_booster:set_ranks()
            if G.shop_booster.align_cards then
                G.shop_booster:align_cards()
            end
            snap_card_area(G.shop_booster)

            if G.shop_booster.cards then
                for _, c in ipairs(G.shop_booster.cards) do
                    attach_shop_card_ui_sync(c)
                end
            end
        end
    end

    -- Top Joker Bar (0 to 9 jokers)
    self:populate_joker_bar()

    -- Top Consumable Area (0 to 2 cards, or Perkeo mode)
    self:populate_consumables_area()

    self:randomize_background()

    -- Apply card selection in each region (jokers, consumables, shop items)
    self:apply_shop_card_selections()

    -- Snap all cards and areas immediately to eliminate sliding animations
    snap_card_area(G.shop_jokers)
    snap_card_area(G.shop_vouchers)
    snap_card_area(G.shop_booster)
    snap_card_area(G.jokers)
    snap_card_area(G.consumeables)
end

-- =========================================================
-- SETUP PACK SCREEN (BUFFOON, ARCANA, SPECTRAL, CELESTIAL)
-- =========================================================
function G.DATASET_MOD:setup_pack_screen()
    if not G.GAME then return end

    if G.shop then
        G.shop.alignment.offset.y = G.ROOM.T.y + 11
        if G.shop.hard_set_VT then G.shop:hard_set_VT() end
    end
    if G.SHOP_SIGN then
        G.SHOP_SIGN.alignment.offset.y = -15
        if G.SHOP_SIGN.align_to_major then G.SHOP_SIGN:align_to_major() end
        if G.SHOP_SIGN.hard_set_VT then G.SHOP_SIGN:hard_set_VT() end
    end
    if G.pack_cards then
        clear_card_area(G.pack_cards)
    end
    if G.booster_pack then
        G.booster_pack:remove()
        G.booster_pack = nil
        G.pack_cards = nil
        prune_engine_tables()
    end

    -- Randomize deck skin collabs so Standard packs cover vanilla and crossover card skins
    if self.randomize_deck_appearance then
        self:randomize_deck_appearance()
    end

    -- 50% Buffoon pack, 15% Standard pack, 15% Arcana, 10% Celestial, 10% Spectral
    local pack_roll = math.random()
    local pack_kind = 'Buffoon'
    if pack_roll < 0.50 then
        pack_kind = 'Buffoon'
    elseif pack_roll < 0.65 then
        pack_kind = 'Standard'
    elseif pack_roll < 0.80 then
        pack_kind = 'Arcana'
    elseif pack_roll < 0.90 then
        pack_kind = 'Celestial'
    else
        pack_kind = 'Spectral'
    end
    self.current_pack_kind = pack_kind

    -- 10% chance for Showman effect (allows duplicate cards in pack)
    local has_showman = (math.random() < 0.10)
    self.pack_has_showman = has_showman

    local _, base_tarots, base_spectrals, base_planets = self:get_joker_pools()

    -- Legendary jokers NEVER appear in buffoon packs
    local pack_jokers = ShopGen.get_all_jokers(false)

    -- In vanilla:
    -- Arcana packs can spawn 'c_soul' (The Soul) (~0.3% base in vanilla, we give a healthy spawn chance for CV dataset: ~5%)
    -- Spectral packs can spawn any spectral card plus 'c_soul' and 'c_black_hole'
    -- Celestial packs can spawn planet cards plus 'c_black_hole' (Black Hole)
    local soul_center = G.P_CENTERS and G.P_CENTERS['c_soul']
    local black_hole_center = G.P_CENTERS and G.P_CENTERS['c_black_hole']

    local pack_size = 2
    local uibox_fn = nil
    local pack_state = G.STATES.BUFFOON_PACK
    local card_type = 'Joker'
    local pool = pack_jokers

    if pack_kind == 'Buffoon' then
        -- Buffoon Pack: 2 (Normal), 4 (Jumbo/Mega)
        self.pack_size_cycle = (self.pack_size_cycle == 2) and 4 or 2
        pack_size = self.pack_size_cycle
        uibox_fn = create_UIBox_buffoon_pack
        pack_state = G.STATES.BUFFOON_PACK
        card_type = 'Joker'
        pool = pack_jokers
    elseif pack_kind == 'Arcana' then
        -- Arcana Pack: 3 (Normal), 5 (Jumbo/Mega)
        self.arcana_size_cycle = (self.arcana_size_cycle == 3) and 5 or 3
        pack_size = self.arcana_size_cycle
        uibox_fn = create_UIBox_arcana_pack
        pack_state = G.STATES.TAROT_PACK
        card_type = 'Tarot'
        pool = {}
        for _, c in ipairs(base_tarots) do table.insert(pool, c) end
        if soul_center then table.insert(pool, soul_center) end
    elseif pack_kind == 'Spectral' then
        -- Spectral Pack: 2 (Normal), 4 (Jumbo/Mega)
        self.spectral_size_cycle = (self.spectral_size_cycle == 2) and 4 or 2
        pack_size = self.spectral_size_cycle
        uibox_fn = create_UIBox_spectral_pack
        pack_state = G.STATES.SPECTRAL_PACK
        card_type = 'Spectral'
        pool = {}
        for _, c in ipairs(base_spectrals) do table.insert(pool, c) end
        if soul_center then table.insert(pool, soul_center) end
        if black_hole_center then table.insert(pool, black_hole_center) end
    elseif pack_kind == 'Celestial' then
        -- Celestial Pack: 3 (Normal), 5 (Jumbo/Mega)
        self.celestial_size_cycle = (self.celestial_size_cycle == 3) and 5 or 3
        pack_size = self.celestial_size_cycle
        uibox_fn = create_UIBox_celestial_pack
        pack_state = G.STATES.PLANET_PACK
        card_type = 'Planet'
        pool = {}
        for _, c in ipairs(base_planets) do table.insert(pool, c) end
        if black_hole_center then table.insert(pool, black_hole_center) end
    elseif pack_kind == 'Standard' then
        -- Standard Pack: 3 (Normal), 5 (Jumbo/Mega)
        self.standard_size_cycle = (self.standard_size_cycle == 3) and 5 or 3
        pack_size = self.standard_size_cycle
        uibox_fn = create_UIBox_standard_pack
        pack_state = G.STATES.STANDARD_PACK
        card_type = 'PlayingCard'
        pool = nil
    end

    G.GAME.pack_size = pack_size
    G.GAME.pack_choices = (pack_size >= 4 and 2) or 1
    self.current_pack_size = pack_size

    if uibox_fn then
        G.booster_pack = UIBox{
            definition = uibox_fn(),
            config = {align = "tmi", offset = {x = 0, y = -2.2}, major = G.hand, bond = 'Weak'}
        }
        G.booster_pack.alignment.offset.y = -2.2
        if G.booster_pack.hard_set_VT then G.booster_pack:hard_set_VT() end
        G.STATE = pack_state
        G.STATE_COMPLETE = true
    end

    if G.pack_cards then
        clear_card_area(G.pack_cards)
        G.pack_cards.config.card_limit = pack_size
        G.pack_cards.config.temp_limit = pack_size

        local used_pack_keys = {}

        for i = 1, pack_size do
            local chosen_center = nil
            if pool and #pool > 0 then
                if has_showman then
                    chosen_center = pool[math.random(#pool)]
                else
                    for attempt = 1, 30 do
                        local cand = pool[math.random(#pool)]
                        if not used_pack_keys[cand.key] then
                            chosen_center = cand; break
                        end
                    end
                    chosen_center = chosen_center or pool[math.random(#pool)]
                    used_pack_keys[chosen_center.key] = true
                end
            end

            local slot_x = G.pack_cards.T.x + (G.pack_cards.T.w - G.pack_cards.card_w) * ((i - 1) / math.max(pack_size - 1, 1)) + 0.5 * (G.pack_cards.card_w - G.CARD_W)
            local slot_y = G.pack_cards.T.y + G.pack_cards.T.h / 2 - G.CARD_H / 2

            local card
            if card_type == 'Joker' then
                card = self:create_random_joker_card(G.pack_cards, chosen_center, false, slot_x, slot_y)
            elseif card_type == 'PlayingCard' then
                card = self:create_random_playing_card(G.pack_cards, slot_x, slot_y)
            else
                local set_name = (chosen_center and chosen_center.set) or card_type
                card = self:create_random_consumable_card(G.pack_cards, set_name, chosen_center, slot_x, slot_y)
            end

            if card then
                table.insert(G.pack_cards.cards, card)
                card:set_card_area(G.pack_cards)
            end
        end

        G.pack_cards:set_ranks()
        if G.pack_cards.align_cards then
            G.pack_cards:align_cards()
        end
    end

    -- Top Joker Bar (0 to 9 jokers)
    self:populate_joker_bar()

    -- Top Consumable Area (0 to 2 cards, or Perkeo mode)
    self:populate_consumables_area()

    self:randomize_background()

    -- Apply card selection in each region (jokers, consumables, shop items)
    self:apply_shop_card_selections()

    -- Snap all cards and areas immediately to eliminate sliding animations
    snap_card_area(G.pack_cards)
    snap_card_area(G.jokers)
    snap_card_area(G.consumeables)
end

-- Step alternating generator: 100 on shop screen, then 100 on pack screen
function G.DATASET_MOD:generate_shop_step()
    self.shop_phase = self.shop_phase or 'shop'
    self.phase_step = self.phase_step or 0

    if self.shop_phase == 'shop' then
        self:setup_shop_screen()
        self.phase_step = self.phase_step + 1
        if self.phase_step >= (self.config.frames_per_phase or 100) then
            self.shop_phase = 'pack'
            self.phase_step = 0
            print("DATASET: Completed 100 shop screens, switching to PACK phase")
        end
    else
        self:setup_pack_screen()
        self.phase_step = self.phase_step + 1
        if self.phase_step >= (self.config.frames_per_phase or 100) then
            self.shop_phase = 'shop'
            self.phase_step = 0
            print("DATASET: Completed 100 pack screens, switching to SHOP phase")
        end
    end
    ShopGen.cleanup_cycle()
end

-- Toggle screen between Shop and Pack for manual testing/calibration (press 'P')
function G.DATASET_MOD:toggle_shop_screen_mode()
    if self.shop_phase == 'shop' then
        self.shop_phase = 'pack'
        self:setup_pack_screen()
        if self.selected_target == 'roi_shop' or self.selected_target == 'roi_voucher' or self.selected_target == 'roi_booster' then
            self.selected_target = 'roi_pack'
        end
    else
        self.shop_phase = 'shop'
        self:setup_shop_screen()
        if self.selected_target == 'roi_pack' then
            self.selected_target = 'roi_shop'
        end
    end
    self.capture_flash = 1.0
    self.capture_flash_text = "SWITCHED TO " .. string.upper(self.shop_phase) .. " VIEW (" .. (self.current_pack_kind or 'SHOP') .. ")"
end

return ShopGen
