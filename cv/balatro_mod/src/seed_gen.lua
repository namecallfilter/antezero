-- gamestate_dataset_cv/src/seed_gen.lua
-- Spawns genuine Win / Game Over overlays with randomized 8-char Balatro seeds for Seed OCR data collection.

local SeedGen = {}

-- Exact 34-character Balatro random seed alphabet (excludes '0' and 'O').
local SEED_CHARS = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
local CONFUSABLE_CHARS = "12568BILMSUVZWGC"

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

local function snap_overlay_tree(node, visited)
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
        snap_overlay_tree(node.UIRoot, visited)
    end
    if node.children then
        for _, child in pairs(node.children) do
            snap_overlay_tree(child, visited)
        end
    end
end

function G.DATASET_MOD:sample_random_seed()
    local pool = (math.random() < 0.35) and CONFUSABLE_CHARS or SEED_CHARS
    local plen = #pool
    local chars = {}
    for i = 1, 8 do
        local idx = math.random(1, plen)
        chars[i] = string.sub(pool, idx, idx)
    end
    if math.random() < 0.15 then
        local p = math.random(1, 7)
        chars[p + 1] = chars[p]
    end
    return table.concat(chars)
end

function G.DATASET_MOD:cleanup_seed_overlay()
    self.seed_corrupted = false
    if G then
        if G.OVERLAY_MENU and G.OVERLAY_MENU.remove then
            pcall(G.OVERLAY_MENU.remove, G.OVERLAY_MENU)
        end
        G.OVERLAY_MENU = nil
        if G.SETTINGS then
            G.SETTINGS.paused = false
        end
        if G.ROOM then
            G.ROOM.jiggle = 0
        end
    end
    if self.ShopGen and self.ShopGen.cleanup_cycle then
        self.ShopGen.cleanup_cycle()
    else
        collectgarbage("collect")
    end
end

local function randomize_round_scores_for_modal()
    if not G or not G.GAME then return end
    G.GAME.round_scores = G.GAME.round_scores or {}
    local numeric_keys = {
        "cards_played", "cards_discarded", "cards_purchased",
        "times_rerolled", "new_collection", "hand",
    }
    for _, k in ipairs(numeric_keys) do
        if type(G.GAME.round_scores[k]) ~= "table" then
            G.GAME.round_scores[k] = {amt = 0}
        end
        if k == "hand" then
            G.GAME.round_scores[k].amt = math.random(250, 5000000)
        elseif k == "new_collection" then
            G.GAME.round_scores[k].amt = math.random(0, 12)
        else
            G.GAME.round_scores[k].amt = math.random(1, 350)
        end
    end
    G.GAME.round_scores.seed = G.GAME.round_scores.seed or {amt = 0}
    G.GAME.round_scores.poker_hand = G.GAME.round_scores.poker_hand or {amt = 0}
    G.GAME.round_scores.furthest_ante = G.GAME.round_scores.furthest_ante or {amt = 0}
    G.GAME.round_scores.furthest_round = G.GAME.round_scores.furthest_round or {amt = 0}
    G.GAME.round_scores.defeated_by = G.GAME.round_scores.defeated_by or {amt = 0}

    G.GAME.round_resets = G.GAME.round_resets or {}
    G.GAME.round_resets.ante = math.random(1, 8)
    G.GAME.round = math.random(1, 27)
    G.GAME.pseudorandom = G.GAME.pseudorandom or {}
    G.GAME.pseudorandom.seed = G.DATASET_MOD:sample_random_seed()
    G.GAME.seeded = (math.random() < 0.12)

    if G.GAME.blind and G.P_BLINDS then
        local keys = {}
        for k, b in pairs(G.P_BLINDS) do
            if type(b) == "table" and b.pos then
                table.insert(keys, k)
            end
        end
        if #keys > 0 then
            local b_cfg = G.P_BLINDS[keys[math.random(#keys)]]
            G.GAME.blind.config = G.GAME.blind.config or {}
            G.GAME.blind.config.blind = b_cfg
        end
    end
end

function G.DATASET_MOD:setup_seed_screen(phase_override)
    if not G or not G.GAME then
        self.capture_flash = 2.0
        self.capture_flash_text = "START A RUN FIRST BEFORE GENERATING SEED OCR DATA!"
        return false
    end

    self:cleanup_seed_overlay()
    local phase = phase_override or self.seed_phase or 'win'
    self.seed_phase = phase

    if self.randomize_background then
        self:randomize_background()
    end

    randomize_round_scores_for_modal()

    -- ~10% negative/empty examples: background only (no modal overlay)
    local negative_roll = math.random()
    if negative_roll < 0.10 then
        self.seed_corrupted = true
        return true
    end

    local ok_def, def = false, nil
    if phase == 'game_over' and create_UIBox_game_over then
        ok_def, def = pcall(create_UIBox_game_over)
    elseif create_UIBox_win then
        ok_def, def = pcall(create_UIBox_win)
    end
    if not ok_def or not def or not UIBox then
        self.capture_flash = 2.0
        self.capture_flash_text = "FAILED TO BUILD SEED OVERLAY MODAL!"
        return false
    end

    local ok_box, box = pcall(function()
        return UIBox{
            definition = def,
            config = {
                align = "cm",
                offset = {x = 0, y = 0},
                major = G.ROOM_ATTACH,
                bond = 'Weak',
                no_esc = true,
            }
        }
    end)
    if not ok_box or not box then
        return false
    end

    G.OVERLAY_MENU = box
    G.SETTINGS.paused = true
    self.seed_corrupted = false
    G.OVERLAY_MENU.alignment.offset.y = 0
    if G.OVERLAY_MENU.align_to_major then
        pcall(G.OVERLAY_MENU.align_to_major, G.OVERLAY_MENU)
    end
    snap_overlay_tree(G.OVERLAY_MENU)
    return true
end

function G.DATASET_MOD:generate_seed_step()
    self.seed_phase = self.seed_phase or 'win'
    self.seed_phase_step = self.seed_phase_step or 0

    local ok = self:setup_seed_screen(self.seed_phase)
    if not ok then return false end

    self.seed_phase_step = self.seed_phase_step + 1
    local per_phase = self.config.frames_per_phase or 100
    if self.seed_phase_step >= per_phase then
        self.seed_phase = (self.seed_phase == 'win') and 'game_over' or 'win'
        self.seed_phase_step = 0
    end
    return true
end

function G.DATASET_MOD:toggle_seed_screen_mode()
    if (self.seed_phase or 'win') == 'win' then
        self:setup_seed_screen('game_over')
    else
        self:setup_seed_screen('win')
    end
    self.capture_flash = 1.2
    self.capture_flash_text = "SEED MODAL: " .. string.upper(self.seed_phase or 'win')
end

return SeedGen
