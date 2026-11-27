-- gamestate_dataset_cv/src/hud_gen.lua
-- Randomizes HUD values across Blind and Shop states for HUD OCR dataset generation.

local HudGen = {}

local function sample_score_value()
    local r = math.random()
    if r < 0.15 then
        return 0
    elseif r < 0.45 then
        return math.random(1, 999)
    elseif r < 0.80 then
        return math.random(1000, 99999999)
    else
        local exp = math.random(11, 28)
        local mantissa = 1.0 + math.random() * 8.99
        return mantissa * (10 ^ exp)
    end
end

local function sample_weighted_small_int(low_max, high_max, low_prob)
    if math.random() < (low_prob or 0.75) then
        return math.random(0, low_max)
    end
    return math.random(low_max + 1, high_max)
end

local function snap_dynatext(dt_obj)
    if not dt_obj then return end
    if dt_obj.config then
        dt_obj.config.pop_in = nil
        dt_obj.config.pop_out = nil
        dt_obj.config.quiver = nil
        dt_obj.config.bump = nil
    end
    if dt_obj.update_text then
        pcall(dt_obj.update_text, dt_obj, true)
    end
    if dt_obj.strings then
        for _, s in ipairs(dt_obj.strings) do
            if s.letters then
                for _, let in ipairs(s.letters) do
                    let.pop_in = 1
                    let.offset = {x = 0, y = 0}
                end
            end
        end
    end
    if dt_obj.align_letters then
        pcall(dt_obj.align_letters, dt_obj)
    end
    if dt_obj.hard_set_VT then
        pcall(dt_obj.hard_set_VT, dt_obj)
    end
end

local function snap_ui_tree(node, visited)
    if not node or type(node) ~= "table" then return end
    visited = visited or {}
    if visited[node] then return end
    visited[node] = true

    if node.velocity then
        node.velocity.x = 0
        node.velocity.y = 0
        node.velocity.r = 0
        node.velocity.scale = 0
    end
    if node.align_to_major then
        pcall(node.align_to_major, node)
    end
    if node.hard_set_VT then
        pcall(node.hard_set_VT, node)
    end
    if node.VT and node.T then
        node.VT.x = node.T.x
        node.VT.y = node.T.y
        node.VT.w = node.T.w
        node.VT.h = node.T.h
        node.VT.scale = node.T.scale
        node.VT.r = node.T.r
    end
    if node.config and node.config.object then
        local obj = node.config.object
        if obj.update_text then
            snap_dynatext(obj)
        elseif obj.hard_set_VT then
            pcall(obj.hard_set_VT, obj)
        end
    end
    if node.UIRoot then
        snap_ui_tree(node.UIRoot, visited)
    end
    if node.children then
        for _, child in pairs(node.children) do
            snap_ui_tree(child, visited)
        end
    end
end

function G.DATASET_MOD:randomize_hud_common_values()
    if not G or not G.GAME then return end
    G.GAME.current_round = G.GAME.current_round or {}
    G.GAME.round_resets = G.GAME.round_resets or {}

    local chips = sample_score_value()
    G.GAME.chips = chips
    if number_format then
        G.GAME.chips_text = number_format(chips)
    else
        G.GAME.chips_text = tostring(chips)
    end
    if G.hand_text_area and G.hand_text_area.game_chips and G.FUNCS and G.FUNCS.chip_UI_set then
        pcall(G.FUNCS.chip_UI_set, G.hand_text_area.game_chips)
    end

    local dr = math.random()
    if dr < 0.12 then
        G.GAME.dollars = -math.random(1, 35)
    elseif dr < 0.75 then
        G.GAME.dollars = math.random(0, 250)
    else
        G.GAME.dollars = math.random(251, 9999)
    end

    G.GAME.current_round.hands_left = sample_weighted_small_int(8, 25, 0.85)
    G.GAME.current_round.discards_left = sample_weighted_small_int(6, 25, 0.85)

    if math.random() < 0.80 then
        G.GAME.round_resets.ante = math.random(1, 12)
    else
        G.GAME.round_resets.ante = math.random(13, 39)
    end

    if math.random() < 0.80 then
        G.GAME.round = math.random(1, 24)
    else
        G.GAME.round = math.random(25, 80)
    end

    if self.randomize_background then
        self:randomize_background()
    end

    if G.HUD then
        if G.HUD.recalculate then pcall(G.HUD.recalculate, G.HUD, true) end
        snap_ui_tree(G.HUD)
    end
    if G.HUD_blind then
        if G.HUD_blind.recalculate then pcall(G.HUD_blind.recalculate, G.HUD_blind, true) end
        snap_ui_tree(G.HUD_blind)
    end
    if G.shop then
        snap_ui_tree(G.shop)
    end
    if G.SHOP_SIGN then
        snap_ui_tree(G.SHOP_SIGN)
    end
end

local function cleanup_hud_transitions(self)
    self.hud_corrupted_fields = {}
    if self.hud_temp_uiboxes then
        for _, box in ipairs(self.hud_temp_uiboxes) do
            if box and box.remove then
                pcall(box.remove, box)
            end
        end
    end
    self.hud_temp_uiboxes = {}
end

local function corrupt_dynatext_letters(uie)
    if not uie or not uie.config or not uie.config.object then return false end
    local dt_obj = uie.config.object
    if not dt_obj.strings or #dt_obj.strings == 0 then return false end
    local s = dt_obj.strings[dt_obj.focused_string or 1] or dt_obj.strings[1]
    if not s or not s.letters or #s.letters == 0 then return false end

    if dt_obj.config then
        dt_obj.config.pop_in = nil
        dt_obj.config.pop_out = nil
    end

    local n = #s.letters
    local touched = 0
    for i = 1, n do
        local let = s.letters[i]
        if let then
            if i == n or math.random() < 0.70 then
                let.pop_in = math.random() * 0.30
                let.offset = {
                    x = (math.random() - 0.5) * 18,
                    y = (math.random() - 0.5) * 18,
                }
                touched = touched + 1
            end
        end
    end
    return touched > 0
end

local function create_attention_cover(self, uie, delta_text, cover_col)
    if not uie or not uie.parent or not UIBox or not G or not G.UIT then return false end
    local cover = uie.parent
    local box_w = (cover.T and cover.T.w or 0.5) + 0.06
    local box_h = (cover.T and cover.T.h or 0.4) + 0.06
    local ok, box = pcall(function()
        return UIBox{
            T = {0, 0, 0, 0},
            definition = {
                n = G.UIT.ROOT,
                config = {
                    align = 'cm',
                    minw = box_w,
                    minh = box_h,
                    padding = 0.03,
                    r = 0.1,
                    colour = cover_col or (G.C and G.C.MONEY) or {0.95, 0.72, 0.34, 1},
                },
                nodes = {
                    {
                        n = G.UIT.T,
                        config = {
                            text = delta_text,
                            scale = 0.55,
                            colour = (G.C and G.C.WHITE) or {1, 1, 1, 1},
                            shadow = true,
                        }
                    }
                }
            },
            config = {
                align = 'cm',
                offset = {
                    x = (math.random() - 0.5) * 0.12,
                    y = (math.random() - 0.5) * 0.10,
                },
                major = cover,
                bond = 'Weak',
            }
        }
    end)
    if not ok or not box then return false end
    box.attention_text = true
    snap_ui_tree(box)
    self.hud_temp_uiboxes = self.hud_temp_uiboxes or {}
    table.insert(self.hud_temp_uiboxes, box)
    return true
end

local function apply_random_hud_transitions(self, phase)
    self.hud_corrupted_fields = {}
    if math.random() >= 0.30 then return end

    local count = (math.random() < 0.35) and 2 or 1
    for _ = 1, count do
        local kind = math.random(1, 3)
        if kind == 1 then
            -- 1. Mid-slide partial off-screen panel
            if phase == 'shop' and G.shop and G.shop.alignment and G.shop.alignment.offset then
                G.shop.alignment.offset.y = -2.0 + math.random() * 1.5
                if G.shop.recalculate then pcall(G.shop.recalculate, G.shop, true) end
                snap_ui_tree(G.shop)
                self.hud_corrupted_fields.hud_reroll_cost = true
            elseif phase == 'blind' and self.hud_blind_visible ~= false and G.HUD_blind and G.HUD_blind.alignment and G.HUD_blind.alignment.offset then
                G.HUD_blind.alignment.offset.y = -5.8 + math.random() * 2.2
                if G.HUD_blind.recalculate then pcall(G.HUD_blind.recalculate, G.HUD_blind, true) end
                snap_ui_tree(G.HUD_blind)
                self.hud_corrupted_fields.hud_blind_target = true
            end
        elseif kind == 2 then
            -- 2. DynaText mid-pop_in / shrunk / jittered letters
            local candidates = {
                {field = "hud_round_score", uie = G.HUD and G.HUD:get_UIE_by_ID('chip_UI_count')},
                {field = "hud_money",       uie = G.HUD and G.HUD:get_UIE_by_ID('dollar_text_UI')},
                {field = "hud_hands",       uie = G.HUD and G.HUD:get_UIE_by_ID('hand_UI_count')},
                {field = "hud_discards",    uie = G.HUD and G.HUD:get_UIE_by_ID('discard_UI_count')},
                {field = "hud_ante",        uie = G.HUD and G.HUD:get_UIE_by_ID('ante_UI_count')},
                {field = "hud_round",       uie = G.HUD and G.HUD:get_UIE_by_ID('round_UI_count')},
            }
            local pick = candidates[math.random(#candidates)]
            if pick and pick.uie and corrupt_dynatext_letters(pick.uie) then
                self.hud_corrupted_fields[pick.field] = true
            end
        else
            -- 3. attention_text delta popup covering the counter
            local pop_choices = {
                {field = "hud_money",       uie = G.HUD and G.HUD:get_UIE_by_ID('dollar_text_UI'),  texts = {"+$8", "-$4", "+$20", "+$5"}, col = G.C and G.C.MONEY},
                {field = "hud_hands",       uie = G.HUD and G.HUD:get_UIE_by_ID('hand_UI_count'),   texts = {"-1", "+1", "+3"},            col = G.C and G.C.BLUE},
                {field = "hud_discards",    uie = G.HUD and G.HUD:get_UIE_by_ID('discard_UI_count'),texts = {"-1", "+1"},                  col = G.C and G.C.RED},
                {field = "hud_round_score", uie = G.HUD and G.HUD:get_UIE_by_ID('chip_UI_count'),   texts = {"+300", "+1,200", "+45"},       col = G.C and G.C.IMPORTANT},
            }
            local pick = pop_choices[math.random(#pop_choices)]
            if pick and pick.uie then
                local txt = pick.texts[math.random(#pick.texts)]
                if create_attention_cover(self, pick.uie, txt, pick.col) then
                    self.hud_corrupted_fields[pick.field] = true
                end
            end
        end
    end
end

function G.DATASET_MOD:setup_hud_blind_screen()
    if not G or not G.GAME then
        self.capture_flash = 2.0
        self.capture_flash_text = "START A RUN FIRST BEFORE GENERATING HUD OCR DATA!"
        return false
    end

    cleanup_hud_transitions(self)

    if G.shop then
        G.shop.alignment.offset.y = (G.ROOM and G.ROOM.T and G.ROOM.T.y or 0) + 11
        snap_ui_tree(G.shop)
    end
    if G.SHOP_SIGN then
        G.SHOP_SIGN.alignment.offset.y = -15
        snap_ui_tree(G.SHOP_SIGN)
    end

    G.STATE = G.STATES.SELECTING_HAND
    G.STATE_COMPLETE = true
    self.hud_phase = 'blind'

    -- Pick a random blind theme for diverse HUD colors and boss blind banners
    if G.GAME.blind and G.P_BLINDS then
        local keys = {}
        for k, b in pairs(G.P_BLINDS) do
            if type(b) == 'table' and b.name then
                table.insert(keys, k)
            end
        end
        if #keys > 0 then
            local pick_key = keys[math.random(#keys)]
            local b_cfg = G.P_BLINDS[pick_key]
            G.GAME.blind.config = G.GAME.blind.config or {}
            G.GAME.blind.config.blind = b_cfg
            G.GAME.blind.name = b_cfg.name or ''
            G.GAME.blind.boss = not not b_cfg.boss
            G.GAME.blind.pos = b_cfg.pos
            if G.GAME.blind.set_text then pcall(G.GAME.blind.set_text, G.GAME.blind) end
            if G.GAME.blind.change_colour then pcall(G.GAME.blind.change_colour, G.GAME.blind) end
            if G.GAME.blind.children and G.GAME.blind.children.animatedSprite and b_cfg.pos then
                pcall(G.GAME.blind.children.animatedSprite.set_sprite_pos, G.GAME.blind.children.animatedSprite, b_cfg.pos)
            end
        end
    end

    local show_blind = (math.random() < 0.90)
    self.hud_blind_visible = show_blind

    if G.HUD_blind and G.GAME.blind then
        if show_blind then
            local b_chips = sample_score_value()
            if b_chips == 0 then b_chips = math.random(300, 50000) end
            G.GAME.blind.chips = b_chips
            G.GAME.blind.chip_text = number_format and number_format(b_chips) or tostring(b_chips)
            G.GAME.blind.blind_set = true
            G.GAME.blind.states = G.GAME.blind.states or {}
            G.GAME.blind.states.visible = true
            G.HUD_blind.alignment.offset.y = 0
            if G.hand_text_area and G.hand_text_area.blind_chips and G.FUNCS and G.FUNCS.blind_chip_UI_scale then
                pcall(G.FUNCS.blind_chip_UI_scale, G.hand_text_area.blind_chips)
            end
            if G.HUD_blind.recalculate then pcall(G.HUD_blind.recalculate, G.HUD_blind, true) end
            snap_ui_tree(G.HUD_blind)
        else
            G.GAME.blind.blind_set = false
            G.GAME.blind.states = G.GAME.blind.states or {}
            G.GAME.blind.states.visible = false
            G.HUD_blind.alignment.offset.y = -10
            snap_ui_tree(G.HUD_blind)
        end
    end

    self:randomize_hud_common_values()
    apply_random_hud_transitions(self, 'blind')
    return true
end

function G.DATASET_MOD:setup_hud_shop_screen()
    if not G or not G.GAME then
        self.capture_flash = 2.0
        self.capture_flash_text = "START A RUN FIRST BEFORE GENERATING HUD OCR DATA!"
        return false
    end

    cleanup_hud_transitions(self)

    self.hud_phase = 'shop'
    self.hud_blind_visible = false

    if self.setup_shop_screen then
        self:setup_shop_screen()
    end

    if G.HUD_blind and G.GAME.blind then
        G.GAME.blind.blind_set = false
        G.GAME.blind.states = G.GAME.blind.states or {}
        G.GAME.blind.states.visible = false
        G.HUD_blind.alignment.offset.y = -10
        snap_ui_tree(G.HUD_blind)
    end

    G.GAME.current_round = G.GAME.current_round or {}
    local rr = math.random()
    if rr < 0.15 then
        G.GAME.current_round.reroll_cost = 0
    elseif rr < 0.75 then
        G.GAME.current_round.reroll_cost = math.random(1, 15)
    else
        G.GAME.current_round.reroll_cost = math.random(16, 99)
    end

    if G.shop and G.shop.recalculate then
        pcall(G.shop.recalculate, G.shop, true)
    end

    self:randomize_hud_common_values()
    apply_random_hud_transitions(self, 'shop')
    return true
end

function G.DATASET_MOD:generate_hud_step()
    self.hud_phase = self.hud_phase or 'blind'
    self.hud_phase_step = self.hud_phase_step or 0

    local ok = false
    if self.hud_phase == 'blind' then
        ok = self:setup_hud_blind_screen()
    else
        ok = self:setup_hud_shop_screen()
    end
    if not ok then return false end

    self.hud_phase_step = self.hud_phase_step + 1
    local per_phase = self.config.frames_per_phase or 100
    if self.hud_phase_step >= per_phase then
        self.hud_phase = (self.hud_phase == 'blind') and 'shop' or 'blind'
        self.hud_phase_step = 0
    end

    if self.ShopGen and self.ShopGen.cleanup_cycle then
        self.ShopGen.cleanup_cycle()
    end
    return true
end

function G.DATASET_MOD:toggle_hud_screen_mode()
    if (self.hud_phase or 'blind') == 'blind' then
        self:setup_hud_shop_screen()
    else
        self:setup_hud_blind_screen()
    end
    self.capture_flash = 1.2
    self.capture_flash_text = "HUD PHASE: " .. string.upper(self.hud_phase or 'blind')
end

return HudGen
