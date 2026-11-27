-- gamestate_dataset_cv/src/capture.lua
-- Handles coordinate transformations, bounding box calculations, screenshot cropping, and dataset label export.

local Capture = {}

local function apply_local_transform(lx, ly, x, y, r, sx, sy, ox, oy, kx, ky)
    x = x or 0; y = y or 0; r = r or 0; sx = sx or 1; sy = sy or 1
    ox = ox or 0; oy = oy or 0; kx = kx or 0; ky = ky or 0
    lx = lx - ox
    ly = ly - oy
    local slx = lx + ky * ly
    local sly = kx * lx + ly
    lx, ly = slx, sly
    lx = lx * sx
    ly = ly * sy
    local c = math.cos(r)
    local s = math.sin(r)
    local rlx = lx * c - ly * s
    local rly = lx * s + ly * c
    lx, ly = rlx, rly
    lx = lx + x
    ly = ly + y
    return lx, ly
end

local function clamp_aabb_to_roi(roi, bx, by, bw, bh)
    local min_x = math.max(bx, roi.x)
    local max_x = math.min(bx + bw, roi.x + roi.w)
    local min_y = math.max(by, roi.y)
    local max_y = math.min(by + bh, roi.y + roi.h)
    
    if min_x >= max_x or min_y >= max_y then return nil end
    return min_x, min_y, max_x - min_x, max_y - min_y
end

local function paste_clamped_region(img, rx, ry, rw, rh)
    local out_w = math.max(1, math.floor(rw))
    local out_h = math.max(1, math.floor(rh))
    local cropped = love.image.newImageData(out_w, out_h)
    if not img then return cropped end
    local img_w, img_h = img:getWidth(), img:getHeight()
    local floor_rx, floor_ry = math.floor(rx), math.floor(ry)
    local src_x1 = math.max(0, floor_rx)
    local src_y1 = math.max(0, floor_ry)
    local src_x2 = math.min(img_w, floor_rx + out_w)
    local src_y2 = math.min(img_h, floor_ry + out_h)
    local copy_w = src_x2 - src_x1
    local copy_h = src_y2 - src_y1
    if copy_w > 0 and copy_h > 0 then
        cropped:paste(img, src_x1 - floor_rx, src_y1 - floor_ry, src_x1, src_y1, copy_w, copy_h)
    end
    return cropped
end

local function to_yolo(roi, bx, by, bw, bh)
    local cx = bx + bw / 2
    local cy = by + bh / 2
    local nx = (cx - roi.x) / roi.w
    local ny = (cy - roi.y) / roi.h
    local nw = bw / roi.w
    local nh = bh / roi.h
    return nx, ny, nw, nh
end

local function json_escape(str)
    if not str then return "" end
    str = tostring(str)
    str = string.gsub(str, "\\", "\\\\")
    str = string.gsub(str, '"', '\\"')
    str = string.gsub(str, "\n", "\\n")
    str = string.gsub(str, "\r", "\\r")
    return str
end

function G.DATASET_MOD:calculate_boxes_for_card(card, draw_args)
    local ox, oy = draw_args[6], draw_args[7]
    local sprite = card.children and (card.children.center or card.children.front)
    local cw = (sprite and sprite.scale and sprite.scale.x) or (ox and ox > 0 and ox * 2) or 71
    local ch = (sprite and sprite.scale and sprite.scale.y) or (oy and oy > 0 and oy * 2) or 95

    -- Select the appropriate bounding box configuration
    local box_def = nil
    local box_name = "card"

    if card.area == G.jokers then
        box_def = self.config.boxes.joker_bar_card
        box_name = "joker_bar_card"
    elseif card.area == G.consumeables then
        box_def = self.config.boxes.consumeable_card or {w = 0.524853, h = 1.052198, ox = -0.230837, oy = -0.024396}
        box_name = "consumeable_card"
    elseif card.area == G.shop_jokers or card.area == G.pack_cards then
        box_def = self.config.boxes.item_card
        box_name = "item_card"
    elseif card.area == G.shop_vouchers then
        box_def = self.config.boxes.voucher_card or {w = 0.524853, h = 1.052198, ox = -0.230837, oy = -0.024396}
        box_name = "voucher_card"
    elseif card.area == G.shop_booster then
        box_def = self.config.boxes.booster_card or {w = 1.0, h = 1.0, ox = 0.0, oy = 0.0}
        box_name = "booster_card"
    elseif card.area and card.area.config and card.area.config.view_deck then
        box_def = self.config.boxes.deck_card or {w = 0.400000, h = 1.040000, ox = -0.290000, oy = 0.000000}
        box_name = "deck_card"
    else
        box_def = self.config.boxes.card
        box_name = "card"
    end

    box_def = box_def or {w = 1.0, h = 1.0, ox = 0.0, oy = 0.0}

    local center_x = cw / 2 + box_def.ox * cw
    local center_y = ch / 2 + box_def.oy * ch
    local hw = (box_def.w * cw) / 2
    local hh = (box_def.h * ch) / 2

    local tex_pts = {
        {center_x - hw, center_y - hh},
        {center_x + hw, center_y - hh},
        {center_x + hw, center_y + hh},
        {center_x - hw, center_y + hh}
    }

    local transformed_pts = {}
    for _, pt in ipairs(tex_pts) do
        local lx, ly = apply_local_transform(pt[1], pt[2], unpack(draw_args))
        local gx, gy = love.graphics.transformPoint(lx, ly)
        table.insert(transformed_pts, gx)
        table.insert(transformed_pts, gy)
    end

    local aabb = {
        math.min(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]),
        math.min(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8]),
        math.max(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]) - math.min(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]),
        math.max(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8]) - math.min(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8])
    }

    self.captured_boxes = self.captured_boxes or setmetatable({}, {__mode = "k"})
    self.captured_boxes[card] = {
        box_name = box_name,
        points = transformed_pts,
        aabb = aabb
    }
end

-- ==========================================
-- CARD CLASSIFIER CAPTURE & EXPORT
-- ==========================================
function G.DATASET_MOD:capture_frame_card()
    local roi = self.config.rois.card_hand or {x = 508, y = 567, w = 1071, h = 322}
    local rw, rh = math.max(1, math.floor(roi.w)), math.max(1, math.floor(roi.h))
    local rx, ry = math.floor(roi.x), math.floor(roi.y)
    local current_frame = self.frame_counts.card

    love.graphics.captureScreenshot(function(img)
        local cropped = paste_clamped_region(img, rx, ry, rw, rh)
        
        local filedata = cropped:encode("png")
        local filename = string.format("Mods/gamestate_dataset_cv/dataset/images/frame_%04d.png", current_frame)
        love.filesystem.write(filename, filedata)
        
        if filedata and filedata.release then filedata:release() end
        if cropped and cropped.release then cropped:release() end
        if img and img.release then img:release() end

        self:save_labels_card(current_frame)
        self.frame_counts.card = self.frame_counts.card + 1
        self.frame_count = self.frame_counts.card
        self.state = 'idle'
        self.capture_flash = 0.5
        self.capture_flash_text = string.format("CARD FRAME %04d SAVED!", current_frame)

        -- Incremental GC step to keep memory flat without freezing
        collectgarbage("step", 200)
    end)
end

function G.DATASET_MOD:save_labels_card(frame_num)
    local roi = self.config.rois.card_hand or {x = 508, y = 567, w = 1071, h = 322}
    local yolo_labels = {}
    local json_cards = {}

    if G.hand and G.hand.cards then
        for idx, card in ipairs(G.hand.cards) do
            local box_data = self.captured_boxes and self.captured_boxes[card]
            if box_data and box_data.aabb then
                local min_x, min_y, bw, bh = box_data.aabb[1], box_data.aabb[2], box_data.aabb[3], box_data.aabb[4]
                local clamped = {clamp_aabb_to_roi(roi, min_x, min_y, bw, bh)}
                if clamped[1] then
                    local nx, ny, nw, nh = to_yolo(roi, clamped[1], clamped[2], clamped[3], clamped[4])
                    
                    -- Single class 0 ("card")
                    table.insert(yolo_labels, string.format("0 %.6f %.6f %.6f %.6f", nx, ny, nw, nh))
                    
                    local is_face_down = (card.facing == 'back' or card.sprite_facing == 'back')
                    local rank = "None"
                    local suit = "None"
                    local edition = "None"
                    local enhancement = "None"
                    local seal = "None"
                    local debuffed = false

                    if not is_face_down then
                        if card.config.center and card.config.center.name == "Stone Card" then
                            rank = "Stone"
                            suit = "None"
                        elseif card.base then
                            rank = tostring(card.base.value or "None")
                            suit = tostring(card.base.suit or "None")
                        end
                        
                        if card.edition then
                            if card.edition.foil then edition = "Foil"
                            elseif card.edition.holo then edition = "Holo"
                            elseif card.edition.polychrome then edition = "Polychrome"
                            elseif card.edition.negative then edition = "Negative"
                            end
                        end
                        
                        if card.config.center and card.config.center.name ~= "Default Base" and card.config.center.name ~= "Stone Card" then
                            enhancement = string.gsub(card.config.center.name, " Card", "")
                            enhancement = string.gsub(enhancement, " ", "")
                        elseif card.config.center and card.config.center.name == "Stone Card" then
                            enhancement = "Stone"
                        end
                        
                        seal = card.seal or "None"
                        debuffed = (card.debuff == true)
                    end
                    
                    local selected = card.highlighted and true or false
                    local card_skin = "None"
                    local is_suit_hc = false
                    if suit ~= "None" and not is_face_down then
                        card_skin = (G.SETTINGS.CUSTOM_DECK and G.SETTINGS.CUSTOM_DECK.Collabs and G.SETTINGS.CUSTOM_DECK.Collabs[suit]) or "default"
                        is_suit_hc = (G.SETTINGS.colour_palettes and G.SETTINGS.colour_palettes[suit] == "hc") or false
                    end
                    
                    table.insert(json_cards, string.format(
                        '  {"card_index": %d, "box_yolo": [%.6f, %.6f, %.6f, %.6f], "box_pixel": [%.1f, %.1f, %.1f, %.1f], "rank": "%s", "suit": "%s", "edition": "%s", "enhancement": "%s", "seal": "%s", "selected": %s, "debuffed": %s, "face_down": %s, "skin": "%s", "high_contrast": %s}',
                        idx, nx, ny, nw, nh, clamped[1] - roi.x, clamped[2] - roi.y, clamped[3], clamped[4],
                        rank, suit, edition, enhancement, seal, selected and "true" or "false", debuffed and "true" or "false", is_face_down and "true" or "false",
                        json_escape(card_skin), is_suit_hc and "true" or "false"
                    ))
                end
            end
        end
    end

    local label_path = string.format("Mods/gamestate_dataset_cv/dataset/labels/frame_%04d.txt", frame_num)
    love.filesystem.write(label_path, table.concat(yolo_labels, "\n"))

    local json_path = string.format("Mods/gamestate_dataset_cv/dataset/labels/frame_%04d.json", frame_num)
    local json_content = "[\n" .. table.concat(json_cards, ",\n") .. "\n]\n"
    love.filesystem.write(json_path, json_content)
end

-- ==========================================
-- SHOP, PACK & CONSUMABLE EXPORT
-- ==========================================
local function get_rarity_str(card)
    local r = card.config and card.config.center and card.config.center.rarity
    if r == 1 then return "Common"
    elseif r == 2 then return "Uncommon"
    elseif r == 3 then return "Rare"
    elseif r == 4 then return "Legendary"
    end
    return tostring(r or "Common")
end

local function get_edition_str(card)
    if not card.edition then return "None" end
    if card.edition.foil then return "Foil"
    elseif card.edition.holo then return "Holo"
    elseif card.edition.polychrome then return "Polychrome"
    elseif card.edition.negative then return "Negative"
    end
    return "None"
end

local function export_roi_dataset(mod, img, roi, card_list, area_name, current_frame, file_suffix)
    if not roi then return end
    local rw, rh = math.max(1, math.floor(roi.w)), math.max(1, math.floor(roi.h))
    local rx, ry = math.floor(roi.x), math.floor(roi.y)

    -- 1. Save Image Crop (write FileData directly to avoid multi-megabyte Lua string heap bloat)
    local cropped = paste_clamped_region(img, rx, ry, rw, rh)
    local filedata = cropped:encode("png")
    local img_path = string.format("Mods/gamestate_dataset_cv/dataset_shop/images/frame_%04d_%s.png", current_frame, file_suffix)
    love.filesystem.write(img_path, filedata)

    if filedata and filedata.release then filedata:release() end
    if cropped and cropped.release then cropped:release() end

    -- 2. Build YOLO and JSON labels
    local yolo_labels = {}
    local json_entries = {}

    local class_id_map = {
        ['Joker'] = 0,
        ['Tarot'] = 1,
        ['Spectral'] = 2,
        ['Planet'] = 3,
        ['Voucher'] = 4,
        ['Booster'] = 5,
        ['PlayingCard'] = 6,
        ['Default'] = 6,
        ['Enhanced'] = 6,
        ['Base'] = 6
    }

    if card_list then
        for idx, card in ipairs(card_list) do
            local box_data = mod.captured_boxes and mod.captured_boxes[card]
            if box_data and box_data.aabb then
                local min_x, min_y, bw, bh = box_data.aabb[1], box_data.aabb[2], box_data.aabb[3], box_data.aabb[4]
                local clamped = {clamp_aabb_to_roi(roi, min_x, min_y, bw, bh)}
                if clamped[1] then
                    local nx, ny, nw, nh = to_yolo(roi, clamped[1], clamped[2], clamped[3], clamped[4])

                    local ability_set = card.ability and card.ability.set
                    local center_set = card.config and card.config.center and card.config.center.set
                    local has_card_base = card.base and type(card.base.suit) == 'string' and card.base.suit ~= '' and type(card.base.value) == 'string' and card.base.value ~= ''
                    local is_playing_card = (center_set == 'Default' or center_set == 'Enhanced')
                        or (ability_set == 'Default' or ability_set == 'Enhanced')
                        or (not ability_set and not center_set and has_card_base)
                    local card_type = is_playing_card and "PlayingCard" or (ability_set or center_set or "Joker")
                    local class_id = class_id_map[card_type] or 0

                    table.insert(yolo_labels, string.format("%d %.6f %.6f %.6f %.6f", class_id, nx, ny, nw, nh))

                    local name = (card.config and card.config.center and card.config.center.name) or "Unknown"
                    if is_playing_card then
                        if card.config and card.config.center and card.config.center.name == "Stone Card" then
                            name = "Stone Card"
                        elseif card.base then
                            name = tostring(card.base.value or "") .. " of " .. tostring(card.base.suit or "")
                        end
                    end
                    local key = (card.config and card.config.center and card.config.center.key) or "unknown"
                    if is_playing_card then
                        if card.config and card.config.center and card.config.center.name == "Stone Card" then
                            key = "m_stone"
                        elseif card.base then
                            key = tostring(card.base.suit or "") .. "_" .. tostring(card.base.value or "")
                        end
                    end
                    local rarity = get_rarity_str(card)
                    local edition = get_edition_str(card)
                    local is_negative = (card.edition and card.edition.negative == true) or (card.ability and card.ability.negative == true)
                    local is_eternal = (card.ability and card.ability.eternal == true)
                    local is_perishable = (card.ability and card.ability.perishable == true)
                    local is_rental = (card.ability and card.ability.rental == true)
                    local is_debuffed = (card.debuff == true)
                    local art_variant = card.custom_art_variant or "default"

                    table.insert(json_entries, string.format(
                        '  {"card_index": %d, "area": "%s", "card_type": "%s", "name": "%s", "key": "%s", "rarity": "%s", "edition": "%s", "negative": %s, "eternal": %s, "perishable": %s, "rental": %s, "debuffed": %s, "art_variant": "%s", "box_yolo": [%.6f, %.6f, %.6f, %.6f], "box_pixel": [%.1f, %.1f, %.1f, %.1f]}',
                        idx, area_name, card_type, json_escape(name), json_escape(key), rarity, edition,
                        is_negative and "true" or "false", is_eternal and "true" or "false", is_perishable and "true" or "false", is_rental and "true" or "false", is_debuffed and "true" or "false",
                        json_escape(art_variant),
                        nx, ny, nw, nh, clamped[1] - roi.x, clamped[2] - roi.y, clamped[3], clamped[4]
                    ))

                    -- If playing card in shop or pack, also export cropped card sub-region to dataset_card_crops
                    if is_playing_card and img then
                        local cx1 = math.max(0, math.floor(min_x))
                        local cy1 = math.max(0, math.floor(min_y))
                        local cx2 = math.min(img:getWidth(), math.ceil(min_x + bw))
                        local cy2 = math.min(img:getHeight(), math.ceil(min_y + bh))
                        local cw = cx2 - cx1
                        local ch = cy2 - cy1

                        if cw > 0 and ch > 0 then
                            local card_crop = paste_clamped_region(img, cx1, cy1, cw, ch)
                            local crop_data = card_crop:encode("png")

                            local crop_filename = string.format("Mods/gamestate_dataset_cv/dataset_card_crops/images/crop_%04d_%s_%d.png", current_frame, file_suffix, idx)
                            love.filesystem.write(crop_filename, crop_data)
                            if crop_data and crop_data.release then crop_data:release() end
                            if card_crop and card_crop.release then card_crop:release() end

                            local p_rank = "None"
                            local p_suit = "None"
                            local p_edition = "None"
                            local p_enhancement = "None"
                            local p_seal = "None"
                            local p_debuffed = (card.debuff == true)
                            local p_face_down = (card.facing == 'back' or card.sprite_facing == 'back')

                            if not p_face_down then
                                if card.config.center and card.config.center.name == "Stone Card" then
                                    p_rank = "Stone"
                                    p_suit = "None"
                                elseif card.base then
                                    p_rank = tostring(card.base.value or "None")
                                    p_suit = tostring(card.base.suit or "None")
                                end

                                if card.edition then
                                    if card.edition.foil then p_edition = "Foil"
                                    elseif card.edition.holo then p_edition = "Holo"
                                    elseif card.edition.polychrome then p_edition = "Polychrome"
                                    elseif card.edition.negative then p_edition = "Negative"
                                    end
                                end

                                if card.config.center and card.config.center.name ~= "Default Base" and card.config.center.name ~= "Stone Card" then
                                    p_enhancement = string.gsub(card.config.center.name, " Card", "")
                                    p_enhancement = string.gsub(p_enhancement, " ", "")
                                elseif card.config.center and card.config.center.name == "Stone Card" then
                                    p_enhancement = "Stone"
                                end

                                p_seal = card.seal or "None"
                            end

                            local p_selected = card.highlighted and true or false
                            local p_card_skin = "None"
                            local p_is_suit_hc = false
                            if p_suit ~= "None" and not p_face_down then
                                p_card_skin = (G.SETTINGS.CUSTOM_DECK and G.SETTINGS.CUSTOM_DECK.Collabs and G.SETTINGS.CUSTOM_DECK.Collabs[p_suit]) or "default"
                                p_is_suit_hc = (G.SETTINGS.colour_palettes and G.SETTINGS.colour_palettes[p_suit] == "hc") or false
                            end

                            local crop_json_content = string.format(
                                '[\n  {"card_index": %d, "box_yolo": [0.5, 0.5, 1.0, 1.0], "box_pixel": [0, 0, %d, %d], "rank": "%s", "suit": "%s", "edition": "%s", "enhancement": "%s", "seal": "%s", "selected": %s, "debuffed": %s, "face_down": %s, "skin": "%s", "high_contrast": %s}\n]\n',
                                idx, cw, ch,
                                p_rank, p_suit, p_edition, p_enhancement, p_seal, p_selected and "true" or "false", p_debuffed and "true" or "false", p_face_down and "true" or "false",
                                json_escape(p_card_skin), p_is_suit_hc and "true" or "false"
                            )
                            local crop_json_path = string.format("Mods/gamestate_dataset_cv/dataset_card_crops/labels/crop_%04d_%s_%d.json", current_frame, file_suffix, idx)
                            love.filesystem.write(crop_json_path, crop_json_content)
                        end
                    end
                end
            end
        end
    end

    -- Write YOLO .txt label
    local txt_path = string.format("Mods/gamestate_dataset_cv/dataset_shop/labels/frame_%04d_%s.txt", current_frame, file_suffix)
    love.filesystem.write(txt_path, table.concat(yolo_labels, "\n"))

    -- Write JSON label
    local json_path = string.format("Mods/gamestate_dataset_cv/dataset_shop/labels/frame_%04d_%s.json", current_frame, file_suffix)
    local json_content = "[\n" .. table.concat(json_entries, ",\n") .. "\n]\n"
    love.filesystem.write(json_path, json_content)
end

function G.DATASET_MOD:capture_frame_shop()
    local current_frame = self.frame_counts.shop
    local current_phase = self.shop_phase or 'shop'

    love.graphics.captureScreenshot(function(img)
        -- 1. Always capture top Joker Bar
        local roi_joker = self.config.rois.joker_bar
        local joker_cards = (G.jokers and G.jokers.cards) or {}
        export_roi_dataset(self, img, roi_joker, joker_cards, "joker_bar", current_frame, "joker")

        -- 2. Always capture top Consumable Area
        local roi_cons = self.config.rois.consumeables
        local cons_cards = (G.consumeables and G.consumeables.cards) or {}
        export_roi_dataset(self, img, roi_cons, cons_cards, "consumeables", current_frame, "consumeable")

        -- 3. Capture active screen ROI (Shop or Pack)
        if current_phase == 'shop' then
            -- 3a. Shop Items (Jokers, Consumables)
            local roi_shop = self.config.rois.shop
            local shop_cards = {}
            if G.shop_jokers and G.shop_jokers.cards then
                for _, c in ipairs(G.shop_jokers.cards) do table.insert(shop_cards, c) end
            end
            export_roi_dataset(self, img, roi_shop, shop_cards, "shop", current_frame, "shop")

            -- 3b. Dedicated Voucher Shelf
            local roi_voucher = self.config.rois.voucher
            local voucher_cards = {}
            if G.shop_vouchers and G.shop_vouchers.cards then
                for _, c in ipairs(G.shop_vouchers.cards) do table.insert(voucher_cards, c) end
            end
            export_roi_dataset(self, img, roi_voucher, voucher_cards, "shop_voucher", current_frame, "voucher")

            -- 3c. Dedicated Booster Pack Area (Shelf)
            local roi_booster = self.config.rois.booster
            local booster_cards = {}
            if G.shop_booster and G.shop_booster.cards then
                for _, c in ipairs(G.shop_booster.cards) do table.insert(booster_cards, c) end
            end
            export_roi_dataset(self, img, roi_booster, booster_cards, "shop_booster", current_frame, "booster")
        else
            local roi_pack = self.config.rois.pack
            local pack_cards = (G.pack_cards and G.pack_cards.cards) or {}
            export_roi_dataset(self, img, roi_pack, pack_cards, "pack", current_frame, "pack")
        end

        if img and img.release then img:release() end

        self.frame_counts.shop = self.frame_counts.shop + 1
        self.frame_count = self.frame_counts.shop
        self.state = 'idle'
        self.capture_flash = 0.5
        self.capture_flash_text = string.format("SHOP FRAME %04d SAVED (%s)!", current_frame, string.upper(current_phase))

        -- Run full cleanup and collection cycle to keep memory completely flat
        if self.ShopGen and self.ShopGen.cleanup_cycle then
            self.ShopGen.cleanup_cycle()
        else
            collectgarbage("collect")
        end
    end)
end

-- ==========================================
-- HUD OCR CAPTURE & EXPORT
-- ==========================================
local HUD_CAPTURE_FIELDS = {
    "hud_round_score",
    "hud_money",
    "hud_hands",
    "hud_discards",
    "hud_ante",
    "hud_round",
    "hud_reroll_cost",
    "hud_blind_target",
}

function G.DATASET_MOD:capture_frame_hud()
    if not G or not G.GAME then
        self.state = 'idle'
        self.capture_flash = 1.5
        self.capture_flash_text = "START A RUN FIRST!"
        return
    end

    self.frame_counts = self.frame_counts or {card = 0, shop = 0, end_round = 0, hud_ocr = 0}
    local current_frame = self.frame_counts.hud_ocr or 0
    local phase = self.hud_phase or ((G.STATE == G.STATES.SHOP) and 'shop' or 'blind')

    local fields_snapshot = {
        hud_round_score = tostring(G.GAME.chips_text or (number_format and number_format(G.GAME.chips or 0)) or (G.GAME.chips or 0)),
        hud_money = tostring(G.GAME.dollars or 0),
        hud_hands = tostring((G.GAME.current_round and G.GAME.current_round.hands_left) or 0),
        hud_discards = tostring((G.GAME.current_round and G.GAME.current_round.discards_left) or 0),
        hud_ante = tostring((G.GAME.round_resets and G.GAME.round_resets.ante) or 1),
        hud_round = tostring(G.GAME.round or 1),
        hud_reroll_cost = (phase == 'shop') and tostring((G.GAME.current_round and G.GAME.current_round.reroll_cost) or 0) or "",
        hud_blind_target = (phase == 'blind' and self.hud_blind_visible ~= false and G.GAME.blind and G.GAME.blind.blind_set)
            and tostring(G.GAME.blind.chip_text or (number_format and number_format(G.GAME.blind.chips or 0)) or "0")
            or "",
    }

    if self.hud_corrupted_fields then
        for fname, is_corrupted in pairs(self.hud_corrupted_fields) do
            if is_corrupted and fields_snapshot[fname] ~= nil then
                fields_snapshot[fname] = ""
            end
        end
    end

    local jitter_px = (self.config and self.config.hud_capture_jitter_px) or 5
    local rois_copy = {}
    for _, fname in ipairs(HUD_CAPTURE_FIELDS) do
        local r = self.config.rois[fname]
        if r then
            local jx = (jitter_px > 0) and math.random(-jitter_px, jitter_px) or 0
            local jy = (jitter_px > 0) and math.random(-jitter_px, jitter_px) or 0
            rois_copy[fname] = {x = r.x + jx, y = r.y + jy, w = r.w, h = r.h}
        end
    end

    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset_hud_ocr/images")
    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset_hud_ocr/labels")

    love.graphics.captureScreenshot(function(img)
        for _, fname in ipairs(HUD_CAPTURE_FIELDS) do
            local r = rois_copy[fname]
            if r then
                local rw, rh = math.max(1, math.floor(r.w)), math.max(1, math.floor(r.h))
                local rx, ry = math.floor(r.x), math.floor(r.y)
                local cropped = paste_clamped_region(img, rx, ry, rw, rh)
                local filedata = cropped:encode("png")
                local img_path = string.format("Mods/gamestate_dataset_cv/dataset_hud_ocr/images/frame_%05d_%s.png", current_frame, fname)
                love.filesystem.write(img_path, filedata)
                if filedata and filedata.release then filedata:release() end
                if cropped and cropped.release then cropped:release() end
            end
        end

        if img and img.release then img:release() end

        local field_lines = {}
        for _, fname in ipairs(HUD_CAPTURE_FIELDS) do
            table.insert(field_lines, string.format('    "%s": "%s"', fname, json_escape(fields_snapshot[fname] or "")))
        end
        local json_content = string.format(
            '{\n  "frame_index": %d,\n  "phase": "%s",\n  "fields": {\n%s\n  }\n}\n',
            current_frame,
            json_escape(phase),
            table.concat(field_lines, ",\n")
        )
        local json_path = string.format("Mods/gamestate_dataset_cv/dataset_hud_ocr/labels/frame_%05d.json", current_frame)
        love.filesystem.write(json_path, json_content)

        self.frame_counts.hud_ocr = current_frame + 1
        self.frame_count = self.frame_counts.hud_ocr
        self.state = 'idle'
        self.capture_flash = 0.5
        self.capture_flash_text = string.format("HUD FRAME %05d SAVED (%s)!", current_frame, string.upper(phase))

        if self.ShopGen and self.ShopGen.cleanup_cycle then
            self.ShopGen.cleanup_cycle()
        else
            collectgarbage("collect")
        end
    end)
end

-- ==========================================
-- SEED OCR CAPTURE & EXPORT
-- ==========================================
function G.DATASET_MOD:capture_frame_seed()
    if not G or not G.GAME then
        self.state = 'idle'
        self.capture_flash = 1.5
        self.capture_flash_text = "START A RUN FIRST!"
        return
    end

    self.frame_counts = self.frame_counts or {card = 0, shop = 0, end_round = 0, hud_ocr = 0, seed_ocr = 0}
    local current_frame = self.frame_counts.seed_ocr or 0
    local phase = self.seed_phase or 'win'
    local roi_key = (phase == 'game_over') and 'seed_game_over' or 'seed_win'
    local base_roi = (self.config and self.config.rois and self.config.rois[roi_key])
        or (self.config and self.config.rois and self.config.rois.seed_win)
    if not base_roi then
        self.state = 'idle'
        return
    end

    local jitter_px = (self.config and self.config.seed_capture_jitter_px) or 5
    local jx = (jitter_px > 0) and math.random(-jitter_px, jitter_px) or 0
    local jy = (jitter_px > 0) and math.random(-jitter_px, jitter_px) or 0
    local crop_roi = {x = base_roi.x + jx, y = base_roi.y + jy, w = base_roi.w, h = base_roi.h}

    local seed_label = ""
    local is_seeded = not not G.GAME.seeded
    if not self.seed_corrupted then
        seed_label = tostring((G.GAME.pseudorandom and G.GAME.pseudorandom.seed) or "")
    end

    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset_seed_ocr/images")
    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset_seed_ocr/labels")

    love.graphics.captureScreenshot(function(img)
        local rw, rh = math.max(1, math.floor(crop_roi.w)), math.max(1, math.floor(crop_roi.h))
        local rx, ry = math.floor(crop_roi.x), math.floor(crop_roi.y)
        local cropped = paste_clamped_region(img, rx, ry, rw, rh)
        local filedata = cropped:encode("png")
        local img_path = string.format("Mods/gamestate_dataset_cv/dataset_seed_ocr/images/frame_%05d_seed.png", current_frame)
        love.filesystem.write(img_path, filedata)
        if filedata and filedata.release then filedata:release() end
        if cropped and cropped.release then cropped:release() end
        if img and img.release then img:release() end

        local json_content = string.format(
            '{\n  "frame_index": %d,\n  "phase": "%s",\n  "seeded": %s,\n  "seed": "%s"\n}\n',
            current_frame,
            json_escape(phase),
            is_seeded and "true" or "false",
            json_escape(seed_label)
        )
        local json_path = string.format("Mods/gamestate_dataset_cv/dataset_seed_ocr/labels/frame_%05d.json", current_frame)
        love.filesystem.write(json_path, json_content)

        self.frame_counts.seed_ocr = current_frame + 1
        self.frame_count = self.frame_counts.seed_ocr
        self.state = 'idle'
        self.capture_flash = 0.5
        self.capture_flash_text = string.format("SEED FRAME %05d SAVED (%s | %s)!", current_frame, string.upper(phase), seed_label ~= "" and seed_label or "<EMPTY>")
        if self.ShopGen and self.ShopGen.cleanup_cycle then
            self.ShopGen.cleanup_cycle()
        else
            collectgarbage("collect")
        end
    end)
end

-- ==========================================
-- VIEW DECK CLASSIFIER CAPTURE & EXPORT
-- ==========================================
local function get_view_deck_areas(mod)
    if mod and mod.DeckGen and mod.DeckGen.get_active_deck_areas then
        return mod.DeckGen.get_active_deck_areas(mod)
    end
    local areas = {}
    if G and G.I and G.I.CARDAREA then
        for _, area in ipairs(G.I.CARDAREA) do
            if area and area.config and area.config.view_deck and not area.REMOVED and not area.removed and area.cards then
                table.insert(areas, area)
            end
        end
    end
    table.sort(areas, function(a, b)
        local ay = (a.VT and a.VT.y) or (a.T and a.T.y) or 0
        local by = (b.VT and b.VT.y) or (b.T and b.T.y) or 0
        return ay < by
    end)
    return areas
end

function G.DATASET_MOD:save_labels_deck(frame_num)
    local roi = (self.config and self.config.rois and self.config.rois.view_deck) or {x = 600, y = 200, w = 1140, h = 660}
    local areas = get_view_deck_areas(self)
    local yolo_labels = {}
    local json_cards = {}
    local card_idx = 0

    for _, area in ipairs(areas) do
        local row_cards = {}
        for _, card in ipairs(area.cards) do
            local box_data = self.captured_boxes and self.captured_boxes[card]
            if box_data and box_data.aabb then
                local min_x, min_y, bw, bh = box_data.aabb[1], box_data.aabb[2], box_data.aabb[3], box_data.aabb[4]
                local clamped = {clamp_aabb_to_roi(roi, min_x, min_y, bw, bh)}
                if clamped[1] then
                    table.insert(row_cards, {
                        card = card,
                        aabb_x = min_x,
                        clamped = clamped,
                    })
                end
            end
        end

        table.sort(row_cards, function(a, b)
            if a.aabb_x ~= b.aabb_x then
                return a.aabb_x < b.aabb_x
            end
            return a.clamped[1] < b.clamped[1]
        end)

        for _, entry in ipairs(row_cards) do
            card_idx = card_idx + 1
            local card = entry.card
            local clamped = entry.clamped
            local nx, ny, nw, nh = to_yolo(roi, clamped[1], clamped[2], clamped[3], clamped[4])

            table.insert(yolo_labels, string.format("0 %.6f %.6f %.6f %.6f", nx, ny, nw, nh))

            local is_face_down = (card.facing == 'back' or card.sprite_facing == 'back')
            local row_suit = (card.base and tostring(card.base.suit or "None")) or "None"
            local rank = "None"
            local suit = "None"
            local edition = "None"
            local enhancement = "None"
            local seal = "None"
            local debuffed = false
            local greyed = (card.greyed == true)

            if not is_face_down then
                if card.config and card.config.center and card.config.center.name == "Stone Card" then
                    rank = "Stone"
                    suit = "None"
                    enhancement = "Stone"
                else
                    if card.base then
                        rank = tostring(card.base.value or "None")
                        suit = tostring(card.base.suit or "None")
                    end
                    if card.config and card.config.center and card.config.center.name and card.config.center.name ~= "Default Base" then
                        enhancement = string.gsub(card.config.center.name, " Card", "")
                        enhancement = string.gsub(enhancement, " ", "")
                    end
                end

                if card.edition then
                    if card.edition.foil then edition = "Foil"
                    elseif card.edition.holo then edition = "Holo"
                    elseif card.edition.polychrome then edition = "Polychrome"
                    elseif card.edition.negative then edition = "Negative"
                    end
                end

                seal = card.seal or "None"
                debuffed = (card.debuff == true)
            end

            local card_skin = "None"
            local is_suit_hc = false
            local skin_suit = (suit ~= "None") and suit or row_suit
            if skin_suit ~= "None" and not is_face_down then
                card_skin = (G.SETTINGS.CUSTOM_DECK and G.SETTINGS.CUSTOM_DECK.Collabs and G.SETTINGS.CUSTOM_DECK.Collabs[skin_suit]) or "default"
                is_suit_hc = (G.SETTINGS.colour_palettes and G.SETTINGS.colour_palettes[skin_suit] == "hc") or false
            end

            table.insert(json_cards, string.format(
                '  {"card_index": %d, "row_suit": "%s", "box_yolo": [%.6f, %.6f, %.6f, %.6f], "box_pixel": [%.1f, %.1f, %.1f, %.1f], "rank": "%s", "suit": "%s", "edition": "%s", "enhancement": "%s", "seal": "%s", "selected": false, "debuffed": %s, "face_down": %s, "greyed": %s, "skin": "%s", "high_contrast": %s}',
                card_idx, json_escape(row_suit), nx, ny, nw, nh, clamped[1] - roi.x, clamped[2] - roi.y, clamped[3], clamped[4],
                json_escape(rank), json_escape(suit), json_escape(edition), json_escape(enhancement), json_escape(seal),
                debuffed and "true" or "false", is_face_down and "true" or "false", greyed and "true" or "false",
                json_escape(card_skin), is_suit_hc and "true" or "false"
            ))
        end
    end

    local label_path = string.format("Mods/gamestate_dataset_cv/dataset_deck/labels/frame_%05d.txt", frame_num)
    love.filesystem.write(label_path, table.concat(yolo_labels, "\n"))

    local json_path = string.format("Mods/gamestate_dataset_cv/dataset_deck/labels/frame_%05d.json", frame_num)
    local json_content = "[\n" .. table.concat(json_cards, ",\n") .. "\n]\n"
    love.filesystem.write(json_path, json_content)

    return card_idx
end

function G.DATASET_MOD:capture_frame_deck()
    self.frame_counts = self.frame_counts or {card = 0, shop = 0, end_round = 0, hud_ocr = 0, seed_ocr = 0, deck = 0}
    local roi = (self.config and self.config.rois and self.config.rois.view_deck) or {x = 600, y = 200, w = 1140, h = 660}
    local areas = get_view_deck_areas(self)
    local box_count = 0
    for _, area in ipairs(areas) do
        for _, card in ipairs(area.cards) do
            local box_data = self.captured_boxes and self.captured_boxes[card]
            if box_data and box_data.aabb then
                local min_x, min_y, bw, bh = box_data.aabb[1], box_data.aabb[2], box_data.aabb[3], box_data.aabb[4]
                local cx = clamp_aabb_to_roi(roi, min_x, min_y, bw, bh)
                if cx then
                    box_count = box_count + 1
                end
            end
        end
    end

    if box_count == 0 then
        if self.config and self.config.calibration_mode == 'generation' then
            self.state = 'waiting'
            self.timer = 0.05
        else
            self.state = 'idle'
            self.capture_flash = 1.5
            self.capture_flash_text = "OPEN VIEW DECK OR PRESS [R] FIRST!"
        end
        return
    end

    local rw, rh = math.max(1, math.floor(roi.w)), math.max(1, math.floor(roi.h))
    local rx, ry = math.floor(roi.x), math.floor(roi.y)
    local current_frame = self.frame_counts.deck or 0

    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset_deck/images")
    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset_deck/labels")

    love.graphics.captureScreenshot(function(img)
        local cropped = paste_clamped_region(img, rx, ry, rw, rh)
        local filedata = cropped:encode("png")
        local filename = string.format("Mods/gamestate_dataset_cv/dataset_deck/images/frame_%05d.png", current_frame)
        love.filesystem.write(filename, filedata)

        if filedata and filedata.release then filedata:release() end
        if cropped and cropped.release then cropped:release() end
        if img and img.release then img:release() end

        local saved_count = self:save_labels_deck(current_frame)
        self.frame_counts.deck = current_frame + 1
        self.frame_count = self.frame_counts.deck
        self.state = 'idle'
        self.capture_flash = 0.5
        self.capture_flash_text = string.format("DECK FRAME %05d SAVED (%d cards)!", current_frame, saved_count or 0)

        collectgarbage("step", 200)
    end)
end

-- Top-level capture dispatcher
function G.DATASET_MOD:capture_frame()
    if self.config.mode == 'card_classifier' then
        self:capture_frame_card()
    elseif self.config.mode == 'shop_classifier' then
        self:capture_frame_shop()
    elseif self.config.mode == 'hud_ocr' then
        self:capture_frame_hud()
    elseif self.config.mode == 'seed_ocr' then
        self:capture_frame_seed()
    elseif self.config.mode == 'deck_classifier' then
        self:capture_frame_deck()
    end
end

return Capture
