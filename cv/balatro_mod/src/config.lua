-- gamestate_dataset_cv/src/config.lua

local Config = {}

local DEFAULT_CONFIG = {
    mode = 'off', -- 'off', 'card_classifier', 'shop_classifier', 'end_round_classifier', 'hud_ocr', 'seed_ocr', 'deck_classifier'
    calibration_mode = 'setup', -- 'setup' or 'generation'
    max_frames = 20000,
    max_clips = 1000, -- Target clip count cap for end_round_classifier
    end_round_frames_per_tick = 2, -- 15fps clip frames simulated per LÖVE tick (2 ~= 8x realtime @ 60Hz)
    frames_per_phase = 100, -- 100 shop screens, then 100 pack screens
    perkeo_max_consumables = 10, -- Max consumables in perkeo mode (configurable)
    show_shop_prices = false, -- Synchronously render price labels below shop items (disabled for fast animation-free generation)
    card_selection_chance = '50%', -- '0%', '50%', '100%'
    frame_wait_time = 0.08, -- Seconds to wait between generation and capture
    rois = {
        card_hand    = {x = 508.319934,  y = 567.839093, w = 1071.056521, h = 322.734447},
        pack_hand    = {x = 490.000000,  y = 300.000000, w = 1100.000000, h = 340.000000},
        play_area    = {x = 518.702717,  y = 421.037276, w = 1034.463019, h = 287.461606},
        joker_bar    = {x = 512.701385,  y = 20.409878,  w = 868.473724,  h = 259.967609},
        shop         = {x = 818.386186,  y = 369.181989, w = 720.772049,  h = 280.979925},
        voucher      = {x = 656.059547,  y = 714.459732, w = 360.743773,  h = 284.420985},
        booster      = {x = 1071.358597, y = 724.761198, w = 427.707911,  h = 300.000000},
        pack         = {x = 565.612571,  y = 588.005101, w = 960.654887,  h = 307.239491},
        consumeables = {x = 1369.783812, y = 30.836806,  w = 425.206356,  h = 270.406328},
        hud_round_score  = {x = 220.599731, y = 389.166317, w = 310.609893, h = 91.386205},
        hud_money        = {x = 235.579213, y = 810.532096, w = 258.355712, h = 109.781902},
        hud_hands        = {x = 245.551355, y = 721.540544, w = 103.609385, h = 84.604535},
        hud_discards     = {x = 376.543146, y = 722.718153, w = 104.654328, h = 79.453563},
        hud_ante         = {x = 237.682514, y = 955.549886, w = 118.302089, h = 75.922924},
        hud_round        = {x = 376.753403, y = 955.475887, w = 105.791305, h = 74.350088},
        hud_reroll_cost  = {x = 629.031965, y = 575.298322, w = 122.196835, h = 65.468198},
        hud_blind_target = {x = 245.537867, y = 233.218735, w = 253.485902, h = 88.705984},
        seed_win         = {x = 1066.000000, y = 700.000000, w = 173.000000, h = 58.000000},
        seed_game_over   = {x = 1066.000000, y = 660.000000, w = 173.000000, h = 58.000000},
        view_deck        = {x = 600.000000,  y = 200.000000, w = 1140.000000, h = 660.000000},
    },
    boxes = {
        card             = {w = 0.465890, h = 1.040970, ox = -0.240265, oy = 0.020829},
        joker_bar_card   = {w = 0.524853, h = 1.052198, ox = -0.230837, oy = -0.024396},
        item_card        = {w = 0.524853, h = 1.052198, ox = -0.230837, oy = -0.024396},
        consumeable_card = {w = 0.524853, h = 1.052198, ox = -0.230837, oy = -0.024396},
        voucher_card     = {w = 0.524853, h = 1.052198, ox = -0.230837, oy = -0.024396},
        booster_card     = {w = 1.000000, h = 1.000000, ox = 0.000000,  oy = 0.000000},
        deck_card        = {w = 0.400000, h = 1.040000, ox = -0.290000, oy = 0.000000},
    }
}

function Config.init(mod)
    mod.config = {
        mode = DEFAULT_CONFIG.mode,
        calibration_mode = DEFAULT_CONFIG.calibration_mode,
        max_frames = DEFAULT_CONFIG.max_frames,
        max_clips = DEFAULT_CONFIG.max_clips,
        end_round_frames_per_tick = DEFAULT_CONFIG.end_round_frames_per_tick,
        frames_per_phase = DEFAULT_CONFIG.frames_per_phase,
        perkeo_max_consumables = DEFAULT_CONFIG.perkeo_max_consumables,
        show_shop_prices = DEFAULT_CONFIG.show_shop_prices,
        card_selection_chance = DEFAULT_CONFIG.card_selection_chance,
        frame_wait_time = DEFAULT_CONFIG.frame_wait_time,
        rois = {},
        boxes = {}
    }

    for k, v in pairs(DEFAULT_CONFIG.rois) do
        mod.config.rois[k] = {x = v.x, y = v.y, w = v.w, h = v.h}
    end
    for k, v in pairs(DEFAULT_CONFIG.boxes) do
        mod.config.boxes[k] = {w = v.w, h = v.h, ox = v.ox, oy = v.oy}
    end

    mod.selected_target = 'roi_joker_bar'
    mod.frame_counts = {card = 0, shop = 0, end_round = 0, hud_ocr = 0, seed_ocr = 0, deck = 0}

    Config.init_directories()
    mod:load_config()
    Config.init_frame_counts(mod)
    Config.init_classes(mod)
end

function Config.init_directories()
    local dirs = {
        "Mods/gamestate_dataset_cv/dataset",
        "Mods/gamestate_dataset_cv/dataset/images",
        "Mods/gamestate_dataset_cv/dataset/labels",
        "Mods/gamestate_dataset_cv/dataset_shop",
        "Mods/gamestate_dataset_cv/dataset_shop/images",
        "Mods/gamestate_dataset_cv/dataset_shop/labels",
        "Mods/gamestate_dataset_cv/dataset_card_crops",
        "Mods/gamestate_dataset_cv/dataset_card_crops/images",
        "Mods/gamestate_dataset_cv/dataset_card_crops/labels",
        "Mods/gamestate_dataset_cv/dataset_end_round",
        "Mods/gamestate_dataset_cv/dataset_hud_ocr",
        "Mods/gamestate_dataset_cv/dataset_hud_ocr/images",
        "Mods/gamestate_dataset_cv/dataset_hud_ocr/labels",
        "Mods/gamestate_dataset_cv/dataset_seed_ocr",
        "Mods/gamestate_dataset_cv/dataset_seed_ocr/images",
        "Mods/gamestate_dataset_cv/dataset_seed_ocr/labels",
        "Mods/gamestate_dataset_cv/dataset_deck",
        "Mods/gamestate_dataset_cv/dataset_deck/images",
        "Mods/gamestate_dataset_cv/dataset_deck/labels",
    }
    for _, dir in ipairs(dirs) do
        love.filesystem.createDirectory(dir)
    end
end

function Config.init_frame_counts(mod)
    -- 1. Card dataset frame counter
    local max_card_idx = -1
    local card_dirs = {"Mods/gamestate_dataset_cv/dataset/images", "Mods/gamestate_dataset_cv/dataset/labels"}
    for _, dir in ipairs(card_dirs) do
        if love.filesystem.getInfo(dir) then
            local files = love.filesystem.getDirectoryItems(dir)
            for _, f in ipairs(files) do
                local num = f:match("^frame_(%d+)%.%w+$")
                if num then
                    local n = tonumber(num)
                    if n and n > max_card_idx then max_card_idx = n end
                end
            end
        end
    end
    mod.frame_counts.card = max_card_idx + 1

    -- 2. Shop dataset frame counter
    local max_shop_idx = -1
    local shop_dirs = {"Mods/gamestate_dataset_cv/dataset_shop/images", "Mods/gamestate_dataset_cv/dataset_shop/labels"}
    for _, dir in ipairs(shop_dirs) do
        if love.filesystem.getInfo(dir) then
            local files = love.filesystem.getDirectoryItems(dir)
            for _, f in ipairs(files) do
                local num = f:match("^frame_(%d+)_%w+%.%w+$") or f:match("^frame_(%d+)%.%w+$")
                if num then
                    local n = tonumber(num)
                    if n and n > max_shop_idx then max_shop_idx = n end
                end
            end
        end
    end
    mod.frame_counts.shop = max_shop_idx + 1

    -- 3. End-round dataset clip counter (ignore empty orphan clip directories)
    local max_er_idx = -1
    local er_dir = "Mods/gamestate_dataset_cv/dataset_end_round"
    if love.filesystem.getInfo(er_dir) then
        local items = love.filesystem.getDirectoryItems(er_dir)
        for _, item in ipairs(items) do
            local num = item:match("^clip_(%d+)$")
            if num then
                local clip_path = er_dir .. "/" .. item
                local info = love.filesystem.getInfo(clip_path)
                if info and info.type == "directory" and love.filesystem.getInfo(clip_path .. "/meta.json") then
                    local n = tonumber(num)
                    if n and n > max_er_idx then max_er_idx = n end
                end
            end
        end
    end
    mod.frame_counts.end_round = max_er_idx + 1

    -- 4. HUD OCR dataset frame counter
    local max_hud_idx = -1
    local hud_dir = "Mods/gamestate_dataset_cv/dataset_hud_ocr/labels"
    if love.filesystem.getInfo(hud_dir) then
        for _, f in ipairs(love.filesystem.getDirectoryItems(hud_dir)) do
            local num = f:match("^frame_(%d+)%.json$")
            if num then
                local n = tonumber(num)
                if n and n > max_hud_idx then max_hud_idx = n end
            end
        end
    end
    mod.frame_counts.hud_ocr = max_hud_idx + 1

    -- 5. Seed OCR dataset frame counter
    local max_seed_idx = -1
    local seed_dir = "Mods/gamestate_dataset_cv/dataset_seed_ocr/labels"
    if love.filesystem.getInfo(seed_dir) then
        for _, f in ipairs(love.filesystem.getDirectoryItems(seed_dir)) do
            local num = f:match("^frame_(%d+)%.json$")
            if num then
                local n = tonumber(num)
                if n and n > max_seed_idx then max_seed_idx = n end
            end
        end
    end
    mod.frame_counts.seed_ocr = max_seed_idx + 1

    -- 6. View Deck dataset frame counter
    local max_deck_idx = -1
    local deck_dirs = {"Mods/gamestate_dataset_cv/dataset_deck/images", "Mods/gamestate_dataset_cv/dataset_deck/labels"}
    for _, dir in ipairs(deck_dirs) do
        if love.filesystem.getInfo(dir) then
            for _, f in ipairs(love.filesystem.getDirectoryItems(dir)) do
                local num = f:match("^frame_(%d+)%.%w+$")
                if num then
                    local n = tonumber(num)
                    if n and n > max_deck_idx then max_deck_idx = n end
                end
            end
        end
    end
    mod.frame_counts.deck = max_deck_idx + 1

    if mod.config.mode == 'shop_classifier' then
        mod.frame_count = mod.frame_counts.shop
    elseif mod.config.mode == 'end_round_classifier' then
        mod.frame_count = mod.frame_counts.end_round
    elseif mod.config.mode == 'hud_ocr' then
        mod.frame_count = mod.frame_counts.hud_ocr
    elseif mod.config.mode == 'seed_ocr' then
        mod.frame_count = mod.frame_counts.seed_ocr
    elseif mod.config.mode == 'deck_classifier' then
        mod.frame_count = mod.frame_counts.deck
    else
        mod.frame_count = mod.frame_counts.card
    end
    print(string.format("DATASET: Initialized counters - Card: %d, Shop: %d, EndRound Clips: %d, HUD OCR: %d, Seed OCR: %d, Deck: %d", mod.frame_counts.card, mod.frame_counts.shop, mod.frame_counts.end_round, mod.frame_counts.hud_ocr, mod.frame_counts.seed_ocr, mod.frame_counts.deck))
end

function Config.init_classes(mod)
    local shop_classes_path = "Mods/gamestate_dataset_cv/dataset_shop/classes.txt"
    local shop_classes = "Joker\nTarot\nSpectral\nPlanet\nVoucher\nBooster\nPlayingCard\n"
    love.filesystem.write(shop_classes_path, shop_classes)
    love.filesystem.write("Mods/gamestate_dataset_cv/dataset_deck/classes.txt", "card\n")
end

function Config.clear_all_dataset_data(mod)
    local dirs = {
        "Mods/gamestate_dataset_cv/dataset/images",
        "Mods/gamestate_dataset_cv/dataset/labels",
        "Mods/gamestate_dataset_cv/dataset_shop/images",
        "Mods/gamestate_dataset_cv/dataset_shop/labels",
        "Mods/gamestate_dataset_cv/dataset_card_crops/images",
        "Mods/gamestate_dataset_cv/dataset_card_crops/labels",
        "Mods/gamestate_dataset_cv/dataset_hud_ocr/images",
        "Mods/gamestate_dataset_cv/dataset_hud_ocr/labels",
        "Mods/gamestate_dataset_cv/dataset_seed_ocr/images",
        "Mods/gamestate_dataset_cv/dataset_seed_ocr/labels",
        "Mods/gamestate_dataset_cv/dataset_deck/images",
        "Mods/gamestate_dataset_cv/dataset_deck/labels",
    }
    local removed_count = 0
    for _, dir in ipairs(dirs) do
        if love.filesystem.getInfo(dir) then
            local files = love.filesystem.getDirectoryItems(dir)
            for _, f in ipairs(files) do
                local file_path = dir .. "/" .. f
                if love.filesystem.remove(file_path) then
                    removed_count = removed_count + 1
                end
            end
        end
    end

    local er_root = "Mods/gamestate_dataset_cv/dataset_end_round"
    if love.filesystem.getInfo(er_root) then
        for _, sub in ipairs(love.filesystem.getDirectoryItems(er_root)) do
            local sub_path = er_root .. "/" .. sub
            local info = love.filesystem.getInfo(sub_path)
            if info and info.type == "directory" then
                for _, f in ipairs(love.filesystem.getDirectoryItems(sub_path)) do
                    if love.filesystem.remove(sub_path .. "/" .. f) then
                        removed_count = removed_count + 1
                    end
                end
                love.filesystem.remove(sub_path)
            else
                if love.filesystem.remove(sub_path) then
                    removed_count = removed_count + 1
                end
            end
        end
    end

    mod.frame_counts = mod.frame_counts or {}
    mod.frame_counts.card = 0
    mod.frame_counts.shop = 0
    mod.frame_counts.end_round = 0
    mod.frame_counts.hud_ocr = 0
    mod.frame_counts.seed_ocr = 0
    mod.frame_counts.deck = 0
    mod.frame_count = 0
    mod.phase_step = 0
    mod.stats_text = string.format("Card: 0  |  Shop: 0  |  Deck: 0  |  EndRound Clips: 0  |  Phase: %s",
        string.upper(mod.shop_phase or 'shop'))

    print(string.format("DATASET: Cleared %d files from dataset directories.", removed_count))
    return removed_count
end

function G.DATASET_MOD:clear_all_dataset_data()
    return Config.clear_all_dataset_data(self)
end

function G.DATASET_MOD:save_config()
    local content = "return {\n"
    content = content .. string.format("  mode = %q,\n", self.config.mode or 'off')
    content = content .. string.format("  calibration_mode = %q,\n", self.config.calibration_mode or 'setup')
    content = content .. string.format("  max_frames = %d,\n", self.config.max_frames or 20000)
    content = content .. string.format("  max_clips = %d,\n", self.config.max_clips or 1000)
    content = content .. string.format("  end_round_frames_per_tick = %d,\n", self.config.end_round_frames_per_tick or 2)
    content = content .. string.format("  frames_per_phase = %d,\n", self.config.frames_per_phase or 100)
    content = content .. string.format("  perkeo_max_consumables = %d,\n", self.config.perkeo_max_consumables or 10)
    content = content .. string.format("  show_shop_prices = %s,\n", self.config.show_shop_prices ~= false and "true" or "false")
    content = content .. string.format("  card_selection_chance = %q,\n", self.config.card_selection_chance or '50%')
    content = content .. string.format("  frame_wait_time = %.3f,\n", tonumber(self.config.frame_wait_time) or 0.08)
    
    content = content .. "  rois = {\n"
    for k, v in pairs(self.config.rois) do
        content = content .. string.format("    %s = {x=%f, y=%f, w=%f, h=%f},\n", k, v.x, v.y, v.w, v.h)
    end
    content = content .. "  },\n"

    content = content .. "  boxes = {\n"
    for k, v in pairs(self.config.boxes) do
        content = content .. string.format("    %s = {w=%f, h=%f, ox=%f, oy=%f},\n", k, v.w, v.h, v.ox, v.oy)
    end
    content = content .. "  }\n}\n"

    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset")
    love.filesystem.write("Mods/gamestate_dataset_cv/dataset/boxes.lua", content)
    print("DATASET: Config saved to boxes.lua!")
end

function G.DATASET_MOD:load_config(preserve_mode)
    local path = "Mods/gamestate_dataset_cv/dataset/boxes.lua"
    if love.filesystem.getInfo(path) then
        local content = love.filesystem.read(path)
        local chunk = (loadstring or load)(content)
        if chunk then
            local t = chunk()
            if type(t) == "table" then
                if not preserve_mode then
                    if t.mode then self.config.mode = t.mode end
                    if t.calibration_mode then self.config.calibration_mode = t.calibration_mode end
                end
                if t.max_frames then self.config.max_frames = tonumber(t.max_frames) or 20000 end
                if t.max_clips then self.config.max_clips = tonumber(t.max_clips) or 1000 end
                if t.end_round_frames_per_tick then self.config.end_round_frames_per_tick = tonumber(t.end_round_frames_per_tick) or 2 end
                if t.frames_per_phase then self.config.frames_per_phase = tonumber(t.frames_per_phase) or 100 end
                if t.perkeo_max_consumables then self.config.perkeo_max_consumables = tonumber(t.perkeo_max_consumables) or 10 end
                if t.show_shop_prices ~= nil then self.config.show_shop_prices = (t.show_shop_prices == true) end
                if t.card_selection_chance then self.config.card_selection_chance = tostring(t.card_selection_chance) end
                if t.frame_wait_time then self.config.frame_wait_time = tonumber(t.frame_wait_time) or 0.08 end

                -- Backwards compatibility: single 'roi' table
                if t.roi and type(t.roi) == "table" and not t.rois then
                    self.config.rois.card_hand = {
                        x = tonumber(t.roi.x) or DEFAULT_CONFIG.rois.card_hand.x,
                        y = tonumber(t.roi.y) or DEFAULT_CONFIG.rois.card_hand.y,
                        w = tonumber(t.roi.w) or DEFAULT_CONFIG.rois.card_hand.w,
                        h = tonumber(t.roi.h) or DEFAULT_CONFIG.rois.card_hand.h
                    }
                end

                if t.rois and type(t.rois) == "table" then
                    for k, v in pairs(t.rois) do
                        if type(v) == "table" then
                            self.config.rois[k] = {
                                x = tonumber(v.x) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].x or 0),
                                y = tonumber(v.y) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].y or 0),
                                w = tonumber(v.w) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].w or 100),
                                h = tonumber(v.h) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].h or 100)
                            }
                        end
                    end
                end
                for k, v in pairs(DEFAULT_CONFIG.rois) do
                    if not self.config.rois[k] then
                        self.config.rois[k] = {x = v.x, y = v.y, w = v.w, h = v.h}
                    end
                end

                if t.boxes and type(t.boxes) == "table" then
                    for k, v in pairs(t.boxes) do
                        if type(v) == "table" then
                            self.config.boxes[k] = {
                                w = tonumber(v.w) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].w or 1),
                                h = tonumber(v.h) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].h or 1),
                                ox = tonumber(v.ox) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].ox or 0),
                                oy = tonumber(v.oy) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].oy or 0)
                            }
                        end
                    end
                end
                for k, v in pairs(DEFAULT_CONFIG.boxes) do
                    if not self.config.boxes[k] then
                        self.config.boxes[k] = {w = v.w, h = v.h, ox = v.ox, oy = v.oy}
                    end
                end
                print("DATASET: Config loaded from boxes.lua!")
            end
        end
    end
end

function G.DATASET_MOD:set_mode(mode)
    if mode == 'card_classifier' or mode == 'shop_classifier' or mode == 'end_round_classifier' or mode == 'hud_ocr' or mode == 'seed_ocr' or mode == 'deck_classifier' or mode == 'off' then
        if mode ~= self.config.mode then
            self.config.calibration_mode = 'setup'
            self.state = 'idle'
        end
        if mode ~= 'shop_classifier' and self.restore_joker_display_config then
            self:restore_joker_display_config()
        end
        if mode ~= 'seed_ocr' and self.cleanup_seed_overlay then
            self:cleanup_seed_overlay()
        end
        if mode ~= 'deck_classifier' and self.cleanup_deck_overlay then
            self:cleanup_deck_overlay(true)
        end
        if mode ~= 'end_round_classifier' and self.EndRoundCapture then
            if self._active_end_round_clip then
                self.EndRoundCapture.stop(self)
            end
            self.EndRoundCapture.release_gpu_resources()
        end
        self.config.mode = mode
        if mode == 'card_classifier' then
            self.selected_target = 'roi_card_hand'
            self.frame_count = self.frame_counts.card
        elseif mode == 'shop_classifier' then
            self.selected_target = 'roi_joker_bar'
            self.frame_count = self.frame_counts.shop
        elseif mode == 'end_round_classifier' then
            self.selected_target = 'roi_play_area'
            self.frame_count = self.frame_counts.end_round
        elseif mode == 'hud_ocr' then
            self.selected_target = 'roi_hud_round_score'
            self.frame_count = self.frame_counts.hud_ocr
        elseif mode == 'seed_ocr' then
            self.selected_target = 'roi_seed_win'
            self.frame_count = self.frame_counts.seed_ocr
        elseif mode == 'deck_classifier' then
            self.selected_target = 'roi_view_deck'
            self.frame_count = self.frame_counts.deck
        end
        print("DATASET: Mode set to " .. tostring(mode))
    end
end

return Config
