-- gamestate_dataset_cv/main.lua
-- Root orchestrator for GameState Dataset CV mod.

G.DATASET_MOD = {
    state = 'idle',
    timer = 0,
    capture_flash = 0,
    capture_flash_text = "",
    captured_boxes = setmetatable({}, {__mode = "k"}),
    classes = {},
    class_mapping = {},
    next_class_id = 0,
    mod_path = "",
    current_card = nil
}

-- Determine mod root directory
local function get_mod_path()
    local p = love.filesystem.getSaveDirectory() .. "/Mods/gamestate_dataset_cv/"
    if SMODS and SMODS.current_mod and SMODS.current_mod.path then
        local smods_path = SMODS.current_mod.path
        if string.sub(smods_path, 2, 2) == ":" or string.sub(smods_path, 1, 1) == "/" then
            p = smods_path
        end
    end
    if string.sub(p, -1) ~= "/" and string.sub(p, -1) ~= "\\" then
        p = p .. "/"
    end
    return p
end
G.DATASET_MOD.mod_path = get_mod_path()

-- Submodule Loader
local function load_submodule(rel_path)
    local last_err = nil
    -- 1. Try SMODS.load_file
    if SMODS and SMODS.load_file then
        local chunk, err = SMODS.load_file(rel_path, "gamestate_dataset_cv")
        if chunk then return chunk() end
        if err then last_err = err end
        local chunk2, err2 = SMODS.load_file(rel_path)
        if chunk2 then return chunk2() end
        if err2 then last_err = err2 end
    end
    -- 2. Try NFS.read
    if NFS and NFS.read and G.DATASET_MOD and G.DATASET_MOD.mod_path then
        local full_p = G.DATASET_MOD.mod_path .. rel_path
        local content = NFS.read(full_p)
        if content then
            local chunk, err = (loadstring or load)(content, "=[GDCV " .. rel_path .. "]")
            if chunk then return chunk() end
            if err then last_err = err end
        end
    end
    -- 3. Fallback to love.filesystem
    local candidates = {
        "Mods/gamestate_dataset_cv/" .. rel_path,
        rel_path
    }
    for _, path in ipairs(candidates) do
        if love.filesystem.getInfo(path) then
            local content = love.filesystem.read(path)
            local chunk, err = (loadstring or load)(content, "=[GDCV " .. path .. "]")
            if chunk then return chunk() end
            if err then last_err = err end
        end
    end
    error("GDCV: Failed to load module " .. tostring(rel_path) .. (last_err and (": " .. tostring(last_err)) or ""))
end

-- Load Modular Components
local ConfigModule          = load_submodule("src/config.lua")
local CardsGenModule        = load_submodule("src/cards_gen.lua")
local ShopGenModule         = load_submodule("src/shop_gen.lua")
local CaptureModule         = load_submodule("src/capture.lua")
local EndRoundGenModule     = load_submodule("src/end_round_gen.lua")
local EndRoundCaptureModule = load_submodule("src/end_round_capture.lua")
local HudGenModule          = load_submodule("src/hud_gen.lua")
local SeedGenModule         = load_submodule("src/seed_gen.lua")
local DeckGenModule         = load_submodule("src/deck_gen.lua")
local UIModule              = load_submodule("src/ui.lua")
local RunTracker            = load_submodule("src/run_tracker.lua")

G.DATASET_MOD.ShopGen = ShopGenModule
G.DATASET_MOD.EndRoundGen = EndRoundGenModule
G.DATASET_MOD.EndRoundCapture = EndRoundCaptureModule
G.DATASET_MOD.HudGen = HudGenModule
G.DATASET_MOD.SeedGen = SeedGenModule
G.DATASET_MOD.DeckGen = DeckGenModule
G.DATASET_MOD.RunTracker = RunTracker

-- Initialize configuration, counters, and directories
ConfigModule.init(G.DATASET_MOD)

-- Initialize RunTracker
RunTracker.init()
RunTracker.hook_actions()

-- Initialize SMODS callbacks and config tab
UIModule.init_callbacks(G.DATASET_MOD)
if SMODS and SMODS.current_mod then
    SMODS.current_mod.config_tab = UIModule.create_config_tab
end
if SMODS and SMODS.Mods and SMODS.Mods["gamestate_dataset_cv"] then
    SMODS.Mods["gamestate_dataset_cv"].config_tab = UIModule.create_config_tab
end

-- Backward compatibility helpers for classes.txt
function G.DATASET_MOD:load_classes()
    local path = "Mods/gamestate_dataset_cv/dataset/classes.txt"
    if love.filesystem.getInfo(path) then
        local content = love.filesystem.read(path)
        if content then
            local id = 0
            for line in content:gmatch("[^\r\n]+") do
                local trimmed = line:match("^%s*(.-)%s*$")
                if trimmed and trimmed ~= "" then
                    self.class_mapping[trimmed] = id
                    id = id + 1
                end
            end
            self.next_class_id = id
        end
    end
end

function G.DATASET_MOD:save_classes()
    local ordered = {}
    for name, id in pairs(self.class_mapping) do
        ordered[id] = name
    end
    local lines = {}
    for i = 0, self.next_class_id - 1 do
        table.insert(lines, ordered[i])
    end
    love.filesystem.write("Mods/gamestate_dataset_cv/dataset/classes.txt", table.concat(lines, "\n"))
end

function G.DATASET_MOD:get_class_id(class_name)
    if not self.class_mapping[class_name] then
        self.class_mapping[class_name] = self.next_class_id
        self.next_class_id = self.next_class_id + 1
        self:save_classes()
    end
    return self.class_mapping[class_name]
end

G.DATASET_MOD:load_classes()

-- ==========================================
-- ENGINE HOOKS
-- ==========================================

-- 1. love.graphics.draw hook: compute bounding boxes on card front draw
local orig_lg_draw = love.graphics.draw
function love.graphics.draw(drawable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    local is_front = false
    local card = nil
    if G.DATASET_MOD and G.DATASET_MOD.config and G.DATASET_MOD.config.mode ~= 'off' and G.DATASET_MOD.current_card then
        card = G.DATASET_MOD.current_card
        -- Only capture bounding box if drawing the card's center or front sprite (not price tag UI or DynaText)
        local is_card_sprite = false
        if card.children then
            if card.children.center and card.children.center.atlas and card.children.center.atlas.image == drawable then
                is_card_sprite = true
            elseif card.children.front and card.children.front.atlas and card.children.front.atlas.image == drawable then
                is_card_sprite = true
            elseif card.children.back and card.children.back.atlas and card.children.back.atlas.image == drawable then
                is_card_sprite = true
            elseif not card.children.center and not card.children.front then
                is_card_sprite = true
            end
        else
            is_card_sprite = true
        end

        local cr, cg, cb, ca = love.graphics.getColor()
        if is_card_sprite and ca > 0.9 and (cr > 0.1 or cg > 0.1 or cb > 0.1) then
            is_front = true
            G.DATASET_MOD.captured_boxes = G.DATASET_MOD.captured_boxes or setmetatable({}, {__mode = "k"})
            local current_frame = (G.FRAMES and G.FRAMES.DRAW) or 0
            if card._last_box_frame ~= current_frame then
                card._last_box_frame = current_frame
                local draw_args = {}
                if type(arg1) == "userdata" and arg1.typeOf and arg1:typeOf("Quad") then
                    draw_args = {arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10}
                else
                    draw_args = {arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9}
                end
                G.DATASET_MOD:calculate_boxes_for_card(card, draw_args)
            end
        end
    end

    orig_lg_draw(drawable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
end

-- 2. Card:draw hook: mark card as tracked if inside active areas
local orig_card_draw = Card.draw
function Card:draw(layer)
    if G.DATASET_MOD and G.DATASET_MOD.config and G.DATASET_MOD.config.mode ~= 'off' then
        local tracked = false
        local mode = G.DATASET_MOD.config.mode

        if mode == 'card_classifier' then
            if G.hand and G.hand.cards then
                for _, c in ipairs(G.hand.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.DATASET_MOD.config.calibration_mode == 'setup' and G.play and G.play.cards then
                for _, c in ipairs(G.play.cards) do
                    if c == self then tracked = true; break end
                end
            end
        elseif mode == 'shop_classifier' then
            if G.jokers and G.jokers.cards then
                for _, c in ipairs(G.jokers.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.shop_jokers and G.shop_jokers.cards then
                for _, c in ipairs(G.shop_jokers.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.consumeables and G.consumeables.cards then
                for _, c in ipairs(G.consumeables.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.pack_cards and G.pack_cards.cards then
                for _, c in ipairs(G.pack_cards.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.shop_vouchers and G.shop_vouchers.cards then
                for _, c in ipairs(G.shop_vouchers.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.shop_booster and G.shop_booster.cards then
                for _, c in ipairs(G.shop_booster.cards) do
                    if c == self then tracked = true; break end
                end
            end
        elseif mode == 'deck_classifier' then
            if self.area and self.area.config and self.area.config.view_deck then
                tracked = true
            end
        end

        if tracked then
            G.DATASET_MOD.current_card = self
            G.DATASET_MOD.captured_boxes = G.DATASET_MOD.captured_boxes or setmetatable({}, {__mode = "k"})
        else
            G.DATASET_MOD.current_card = nil
        end
    end

    orig_card_draw(self, layer)

    if G.DATASET_MOD then
        G.DATASET_MOD.current_card = nil
    end
end

-- 3. Card:remove hook: clean up captured box when card is deleted
local orig_card_remove = Card.remove
function Card:remove()
    if G.DATASET_MOD and G.DATASET_MOD.captured_boxes then
        G.DATASET_MOD.captured_boxes[self] = nil
    end
    return orig_card_remove(self)
end

-- Hook nuGC globally to prevent Balatro from disabling automatic GC via collectgarbage('stop')
local orig_nuGC = nuGC
function nuGC(time_budget, memory_ceiling, disable_otherwise)
    if G.DATASET_MOD and G.DATASET_MOD.config and G.DATASET_MOD.config.mode ~= 'off' then
        collectgarbage("step", 200)
        return
    end
    if orig_nuGC then
        return orig_nuGC(time_budget, memory_ceiling, disable_otherwise)
    end
end

-- 4. love.update & love.draw hooks
local orig_love_update = love.update
local orig_love_draw = love.draw

function love.update(dt)
    local handled = false
    if G.DATASET_MOD and G.DATASET_MOD.EndRoundCapture then
        handled = G.DATASET_MOD.EndRoundCapture.step_love_update(G.DATASET_MOD, dt, orig_love_update, orig_love_draw)
    end
    if not handled and orig_love_update then
        orig_love_update(dt)
    end
    if G.DATASET_MOD then
        G.DATASET_MOD:update(dt)
        if G.DATASET_MOD.config and G.DATASET_MOD.config.mode ~= 'off' then
            collectgarbage("step", 200)
        end
    end
end

function love.draw()
    if orig_love_draw then orig_love_draw() end
    if G.DATASET_MOD and G.DATASET_MOD.EndRoundCapture then
        G.DATASET_MOD.EndRoundCapture.capture_after_draw(G.DATASET_MOD)
    end
    if G.DATASET_MOD and not (G.DATASET_MOD.EndRoundCapture and G.DATASET_MOD.EndRoundCapture.in_internal_draw) then
        G.DATASET_MOD:draw_debug()
    end
end

-- 5. love.keypressed hook
local orig_love_keypressed = love.keypressed
function love.keypressed(key)
    if orig_love_keypressed then orig_love_keypressed(key) end
    if G.DATASET_MOD then G.DATASET_MOD:on_keypressed(key) end
end

print("DATASET: GameState Dataset CV loaded successfully!")
