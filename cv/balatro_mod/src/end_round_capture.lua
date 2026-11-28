-- gamestate_dataset_cv/src/end_round_capture.lua
-- Fixed-timestep video capture, event-hook ground truth logger, GPU ROI downscaler, and binary exporter.

local EndRoundCapture = {}

EndRoundCapture.OUTPUT_W = 256
EndRoundCapture.OUTPUT_H = 72
EndRoundCapture.TARGET_FPS = 15.0
EndRoundCapture.SUBSTEPS_PER_FRAME = 4
EndRoundCapture.FIXED_DT = 1.0 / 60.0
EndRoundCapture.BASE_W = 1920.0
EndRoundCapture.BASE_H = 1080.0
EndRoundCapture.DATASET_DIR = "Mods/gamestate_dataset_cv/dataset_end_round"

EndRoundCapture.MAX_CLIP_FRAMES = 750
EndRoundCapture.SETTLE_HOLD_MIN = 8
EndRoundCapture.SETTLE_HOLD_MAX = 22

local BYTES_PER_FRAME = EndRoundCapture.OUTPUT_W * EndRoundCapture.OUTPUT_H * 4

-- Persistent GPU resources allocated lazily and reused across all clips.
local gpu_state = {
    crop_canvas = nil,
    crt_canvas = nil,
    crt_w = 0,
    crt_h = 0,
    quad = nil,
}

--- Escape a Lua string cleanly for JSON output.
local function json_escape(s)
    if s == nil then return "" end
    return tostring(s)
        :gsub("\\", "\\\\")
        :gsub('"', '\\"')
        :gsub("\n", "\\n")
        :gsub("\r", "\\r")
        :gsub("\t", "\\t")
end

--- Encode a flat or nested Lua table of primitives into deterministic JSON.
local function encode_json_value(val, indent)
    local t = type(val)
    if val == nil then
        return "null"
    elseif t == "boolean" then
        return val and "true" or "false"
    elseif t == "number" then
        if val ~= val or val == math.huge or val == -math.huge then
            return "null"
        elseif math.floor(val) == val and math.abs(val) < 1e15 then
            return string.format("%.0f", val)
        end
        return string.format("%.6f", val)
    elseif t == "string" then
        return '"' .. json_escape(val) .. '"'
    elseif t == "table" then
        local keys = {}
        for k in pairs(val) do keys[#keys + 1] = k end
        table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
        if #keys == 0 then return "{}" end
        local pad = string.rep(" ", indent or 0)
        local inner_pad = string.rep(" ", (indent or 0) + 2)
        local parts = {}
        for _, k in ipairs(keys) do
            parts[#parts + 1] = string.format('%s"%s": %s', inner_pad, json_escape(k), encode_json_value(val[k], (indent or 0) + 2))
        end
        return "{\n" .. table.concat(parts, ",\n") .. "\n" .. pad .. "}"
    end
    return '"' .. json_escape(tostring(val)) .. '"'
end

--- Test whether a Moveable's projected visual bounding box overlaps `roi` (in 1920x1080 base pixels).
local function node_overlaps_roi(node, roi)
    if not node or node.REMOVED or not node.VT or not roi then return false end
    local win_w, win_h = love.graphics.getDimensions()
    if win_w <= 0 or win_h <= 0 then return false end

    local tile_px = (G.TILESCALE or 3.65) * (G.TILESIZE or 20)
    local cont = node.container
    local cont_x = (cont and cont ~= node and cont.T and cont.T.x) or (G.ROOM and G.ROOM.T and G.ROOM.T.x) or 0
    local cont_y = (cont and cont ~= node and cont.T and cont.T.y) or (G.ROOM and G.ROOM.T and G.ROOM.T.y) or 0

    local to_base_x = EndRoundCapture.BASE_W / win_w
    local to_base_y = EndRoundCapture.BASE_H / win_h
    local x1 = (cont_x + node.VT.x) * tile_px * to_base_x
    local y1 = (cont_y + node.VT.y) * tile_px * to_base_y
    local x2 = x1 + (node.VT.w or 0) * tile_px * to_base_x
    local y2 = y1 + (node.VT.h or 0) * tile_px * to_base_y

    return x1 < (roi.x + roi.w) and x2 > roi.x and y1 < (roi.y + roi.h) and y2 > roi.y
end

--- Lazily initialize and resize offscreen GPU canvases used for cropping.
local function ensure_gpu_canvases()
    if not gpu_state.crop_canvas then
        gpu_state.crop_canvas = love.graphics.newCanvas(
            EndRoundCapture.OUTPUT_W,
            EndRoundCapture.OUTPUT_H,
            { type = '2d', readable = true, dpiscale = 1 }
        )
        gpu_state.crop_canvas:setFilter('linear', 'linear')
    end

    if not G.AA_CANVAS and G.CANVAS then
        local cw, ch = G.CANVAS:getDimensions()
        local dpi = G.CANVAS.getDPIScale and G.CANVAS:getDPIScale() or 1
        if (not gpu_state.crt_canvas) or gpu_state.crt_w ~= cw or gpu_state.crt_h ~= ch then
            if gpu_state.crt_canvas then
                pcall(function() gpu_state.crt_canvas:release() end)
            end
            gpu_state.crt_canvas = love.graphics.newCanvas(cw, ch, { type = '2d', readable = true, dpiscale = dpi })
            gpu_state.crt_w = cw
            gpu_state.crt_h = ch
        end
    end
end

--- Release GPU canvases when switching away from end_round_classifier to free VRAM.
function EndRoundCapture.release_gpu_resources()
    if gpu_state.crop_canvas then
        pcall(function() gpu_state.crop_canvas:release() end)
        gpu_state.crop_canvas = nil
    end
    if gpu_state.crt_canvas then
        pcall(function() gpu_state.crt_canvas:release() end)
        gpu_state.crt_canvas = nil
    end
    gpu_state.crt_w = 0
    gpu_state.crt_h = 0
    gpu_state.quad = nil
end

--- Grab the current post-CRT frame cropped to `play_area` and resized to 256x72 RGBA bytes.
local function capture_play_area_rgba(mod)
    if not (G and G.CANVAS and mod and mod.config and mod.config.rois and mod.config.rois.play_area) then
        return nil
    end

    ensure_gpu_canvases()
    local src = G.AA_CANVAS
    if not src then
        -- Match Balatro's final screen output (applying G.SHADERS['CRT'] with the user's Graphics settings).
        src = gpu_state.crt_canvas
        if not src then return nil end
        love.graphics.push('all')
        love.graphics.origin()
        love.graphics.setCanvas(src)
        love.graphics.clear(0, 0, 0, 1)
        love.graphics.setColor(1, 1, 1, 1)
        if G.SHADERS and G.SHADERS['CRT'] then
            love.graphics.setShader(G.SHADERS['CRT'])
        else
            love.graphics.setShader()
        end
        -- CRT shader outputs premultiplied-alpha RGB attenuated by vignette mask; replace preserves exact shading.
        love.graphics.setBlendMode('replace', 'premultiplied')
        love.graphics.draw(G.CANVAS, 0, 0)
        love.graphics.setShader()
        love.graphics.setCanvas()
        love.graphics.pop()
    end

    local src_w, src_h = src:getDimensions()
    if src_w <= 0 or src_h <= 0 then return nil end

    local roi = mod.config.rois.play_area
    local scale_x = src_w / EndRoundCapture.BASE_W
    local scale_y = src_h / EndRoundCapture.BASE_H
    local rx = math.max(0, math.min(src_w - 1, roi.x * scale_x))
    local ry = math.max(0, math.min(src_h - 1, roi.y * scale_y))
    local rw = math.max(1, math.min(src_w - rx, roi.w * scale_x))
    local rh = math.max(1, math.min(src_h - ry, roi.h * scale_y))

    if not gpu_state.quad then
        gpu_state.quad = love.graphics.newQuad(rx, ry, rw, rh, src_w, src_h)
    else
        gpu_state.quad:setViewport(rx, ry, rw, rh, src_w, src_h)
    end

    love.graphics.push('all')
    love.graphics.origin()
    love.graphics.setCanvas(gpu_state.crop_canvas)
    love.graphics.setShader()
    love.graphics.clear(0, 0, 0, 1)
    -- Src canvas is opaque after CRT shader; replace mode avoids premultiplication darkening.
    love.graphics.setBlendMode('replace', 'premultiplied')
    src:setFilter('linear', 'linear')
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(src, gpu_state.quad, 0, 0, 0, EndRoundCapture.OUTPUT_W / rw, EndRoundCapture.OUTPUT_H / rh)
    love.graphics.setCanvas()
    love.graphics.pop()

    local img_data = gpu_state.crop_canvas:newImageData()
    local bytes = img_data:getString()
    img_data:release()
    return bytes
end

--- Install one-time idempotent Balatro function hooks forwarding to G.DATASET_MOD.EndRoundCapture.
function EndRoundCapture.install_hooks(mod)
    if mod._er_hooks_installed then return end
    mod._er_hooks_installed = true
    mod._er_orig = mod._er_orig or {}
    local orig = mod._er_orig

    local function dispatch(method, ...)
        local rec = G.DATASET_MOD and G.DATASET_MOD.EndRoundCapture
        if rec and rec[method] then
            return rec[method](G.DATASET_MOD, ...)
        end
    end

    if G.FUNCS and G.FUNCS.play_cards_from_highlighted and not orig.play_cards_from_highlighted then
        orig.play_cards_from_highlighted = G.FUNCS.play_cards_from_highlighted
        G.FUNCS.play_cards_from_highlighted = function(e)
            dispatch('on_play_cards_from_highlighted')
            return orig.play_cards_from_highlighted(e)
        end
    end

    if G.FUNCS and G.FUNCS.draw_from_play_to_discard and not orig.draw_from_play_to_discard then
        orig.draw_from_play_to_discard = G.FUNCS.draw_from_play_to_discard
        G.FUNCS.draw_from_play_to_discard = function(e)
            dispatch('on_draw_from_play_to_discard')
            return orig.draw_from_play_to_discard(e)
        end
    end

    if end_round and not orig.end_round then
        orig.end_round = end_round
        function end_round()
            dispatch('on_end_round_scheduled')
            return orig.end_round()
        end
    end

    if Card and Card.calculate_joker and not orig.calculate_joker then
        orig.calculate_joker = Card.calculate_joker
        function Card:calculate_joker(context)
            dispatch('on_card_calculate_joker', context)
            return orig.calculate_joker(self, context)
        end
    end

    if SMODS and SMODS.calculate_context and not orig.smods_calculate_context then
        orig.smods_calculate_context = SMODS.calculate_context
        function SMODS.calculate_context(context, return_table, no_resolve)
            dispatch('on_smods_calculate_context', context)
            return orig.smods_calculate_context(context, return_table, no_resolve)
        end
    end

    if Card and Card.get_end_of_round_effect and not orig.get_end_of_round_effect then
        orig.get_end_of_round_effect = Card.get_end_of_round_effect
        function Card:get_end_of_round_effect(context)
            dispatch('on_card_get_end_of_round_effect')
            return orig.get_end_of_round_effect(self, context)
        end
    end

    if G.FUNCS and G.FUNCS.draw_from_hand_to_discard and not orig.draw_from_hand_to_discard then
        orig.draw_from_hand_to_discard = G.FUNCS.draw_from_hand_to_discard
        G.FUNCS.draw_from_hand_to_discard = function(e)
            dispatch('on_draw_from_hand_to_discard')
            return orig.draw_from_hand_to_discard(e)
        end
    end

    if G.FUNCS and G.FUNCS.evaluate_round and not orig.evaluate_round then
        orig.evaluate_round = G.FUNCS.evaluate_round
        G.FUNCS.evaluate_round = function()
            dispatch('on_evaluate_round')
            return orig.evaluate_round()
        end
    end

    --- Return the currently scheduled Joker card to hover at `clip.current_frame_idx`, or nil.
    local function resolve_clip_hover_target(clip)
        if not (clip and clip.hover_schedule) then return nil end
        local idx = clip.current_frame_idx or 0
        for _, seg in ipairs(clip.hover_schedule) do
            if idx >= seg.start_frame and idx < seg.end_frame then
                local card = seg.card
                if card and not card.REMOVED and card.area == G.jokers and card.put_focused_cursor then
                    return card
                end
                return nil
            end
        end
        return nil
    end

    if Controller and Controller.set_cursor_position and not orig.set_cursor_position then
        orig.set_cursor_position = Controller.set_cursor_position
        function Controller:set_cursor_position()
            local mod_ref = G and G.DATASET_MOD
            local clip = mod_ref and mod_ref._active_end_round_clip
            if clip then
                self.interrupt.focus = false
                if self.focused and self.focused.target then
                    if self.focused.target.states and self.focused.target.states.focus then
                        self.focused.target.states.focus.is = false
                    end
                    self.focused.target = nil
                end
                local cx, cy = -100, -100
                local hover_card = resolve_clip_hover_target(clip)
                if hover_card then
                    local ok, tx, ty = pcall(function() return hover_card:put_focused_cursor() end)
                    if ok and tx and ty then
                        cx, cy = tx, ty
                    end
                end
                self.cursor_position.x, self.cursor_position.y = cx, cy
                local scale = (G.TILESCALE or 1) * (G.TILESIZE or 1)
                if G.CURSOR and G.CURSOR.T and G.CURSOR.VT then
                    G.CURSOR.T.x = cx / scale
                    G.CURSOR.T.y = cy / scale
                    G.CURSOR.VT.x = G.CURSOR.T.x
                    G.CURSOR.VT.y = G.CURSOR.T.y
                end
                return
            end
            return orig.set_cursor_position(self)
        end
    end
end

-- ==========================================================================
-- Hook Event Receivers
-- ==========================================================================

local function record_end_round_once(mod, source_tag)
    local clip = mod._active_end_round_clip
    if not clip or not clip.end_round_window then return end
    if clip.events.end_round_frame == nil then
        clip.events.end_round_frame = clip.current_frame_idx
        clip.end_round_source = source_tag
    end
end

function EndRoundCapture.on_play_cards_from_highlighted(mod)
    local clip = mod._active_end_round_clip
    if not clip then return end
    if clip.events.play_hand_frame == nil then
        clip.events.play_hand_frame = clip.current_frame_idx
    end
end

function EndRoundCapture.on_draw_from_play_to_discard(mod)
    local clip = mod._active_end_round_clip
    if not clip then return end
    if clip.events.draw_to_discard_frame == nil then
        clip.events.draw_to_discard_frame = clip.current_frame_idx
    end
end

function EndRoundCapture.on_end_round_scheduled(mod)
    local clip = mod._active_end_round_clip
    if not clip then return end
    clip.end_round_window = true
    if clip.events.end_round_scheduled_frame == nil then
        clip.events.end_round_scheduled_frame = clip.current_frame_idx
    end
end

function EndRoundCapture.on_smods_calculate_context(mod, context)
    if context and context.end_of_round and context.main_eval and not context.individual and not context.repetition and not context.other_card then
        record_end_round_once(mod, 'joker_end_of_round')
    end
end

function EndRoundCapture.on_card_calculate_joker(mod, context)
    if context and context.end_of_round and not context.individual and not context.repetition and not context.other_card
        and (context.cardarea == nil or context.cardarea == G.jokers) then
        record_end_round_once(mod, 'joker_end_of_round')
    end
end

function EndRoundCapture.on_card_get_end_of_round_effect(mod)
    record_end_round_once(mod, 'hand_end_of_round')
end

function EndRoundCapture.on_draw_from_hand_to_discard(mod)
    record_end_round_once(mod, 'hand_to_discard')
end

function EndRoundCapture.on_evaluate_round(mod)
    record_end_round_once(mod, 'evaluate_round_fallback')
end

-- ==========================================================================
-- Visual Boundary Tracking & Serialization
-- ==========================================================================

--- Return true once `add_round_eval_row({name='bottom'})` has spawned the Cash Out button.
local function has_cash_out_button()
    if not (G and G.I and G.I.UIBOX and G.round_eval) then return false end
    for _, uib in ipairs(G.I.UIBOX) do
        if uib and not uib.REMOVED and uib.get_UIE_by_ID then
            local maj = (uib.role and uib.role.major) or (uib.config and uib.config.major)
            if maj == G.round_eval and uib:get_UIE_by_ID('cash_out_button') then
                return true
            end
        end
    end
    return false
end

local function update_visual_boundaries(mod, clip, idx)
    local roi = mod.config.rois.play_area
    local ev = clip.events

    -- 1. Check whether any played card visually overlaps play_area yet.
    local any_play_visible = false
    if G.play and G.play.cards then
        for _, card in ipairs(G.play.cards) do
            if not card.REMOVED and (card.dissolve or 0) < 0.98 and node_overlaps_roi(card, roi) then
                any_play_visible = true
                break
            end
        end
    end
    if not any_play_visible then
        for _, card in ipairs(clip.tracked_play_cards) do
            if not card.REMOVED and (card.dissolve or 0) < 0.98 and card.area ~= G.hand and node_overlaps_roi(card, roi) then
                any_play_visible = true
                break
            end
        end
    end

    if any_play_visible then
        clip.saw_cards_in_play = true
        if ev.first_card_in_play_frame == nil then
            ev.first_card_in_play_frame = idx
        end
    elseif clip.saw_cards_in_play and ev.cards_cleared_frame == nil then
        ev.cards_cleared_frame = idx
    end

    -- 2. Track G.round_eval entry into play_area and settled panel state.
    if G.STATE == G.STATES.ROUND_EVAL then
        record_end_round_once(mod, 'state_round_eval_fallback')
        if G.round_eval and not G.round_eval.REMOVED then
            local entered = node_overlaps_roi(G.round_eval, roi)
                or (G.round_eval.UIRoot and node_overlaps_roi(G.round_eval.UIRoot, roi))
            if ev.round_eval_enter_frame == nil and entered then
                ev.round_eval_enter_frame = idx
            end
            if ev.round_eval_enter_frame ~= nil and ev.round_eval_settled_frame == nil then
                local ty = G.round_eval.T and G.round_eval.T.y or 0
                local vy = G.round_eval.VT and G.round_eval.VT.y or 999
                local panel_aligned = math.abs(ty - vy) < 0.05 and (idx - ev.round_eval_enter_frame) >= 2
                local cash_out_ready = panel_aligned and has_cash_out_button()
                local gs = (clip.scenario and clip.scenario.gamespeed) or 1
                local settle_timeout = (gs < 1) and 220 or 140
                if cash_out_ready then
                    ev.round_eval_settled_frame = idx
                    clip.settle_hold_remaining = math.random(
                        EndRoundCapture.SETTLE_HOLD_MIN,
                        EndRoundCapture.SETTLE_HOLD_MAX
                    )
                elseif (idx - ev.round_eval_enter_frame) >= settle_timeout then
                    -- Safety settle threshold in case reduced motion, slow gamespeed, or UI changes delay Cash Out spawning.
                    ev.round_eval_settled_frame = idx
                    clip.settle_hold_remaining = math.random(
                        EndRoundCapture.SETTLE_HOLD_MIN,
                        EndRoundCapture.SETTLE_HOLD_MAX
                    )
                end
            end
        end
    end
end

--- Validate and write `frames.bin` + `meta.json` for a completed clip.
local function finalize_and_write_clip(mod, clip)
    local num_frames = #clip.frame_bytes
    local ev = clip.events

    if num_frames < 8 or ev.end_round_frame == nil or ev.end_round_frame < 0 or ev.end_round_frame >= num_frames then
        print(string.format(
            "DATASET: Skipping incomplete clip %s (frames=%d, end_round=%s, state=%s)",
            tostring(clip.id), num_frames, tostring(ev.end_round_frame), tostring(G.STATE)
        ))
        return false
    end

    for i = 1, num_frames do
        if #clip.frame_bytes[i] ~= BYTES_PER_FRAME then
            print(string.format("DATASET: Skipping corrupt clip %s (frame %d bytes=%d)", clip.id, i, #clip.frame_bytes[i]))
            return false
        end
    end

    local glass_shattered = 0
    for _, c in ipairs(clip.tracked_play_cards) do
        if c.shattered then
            glass_shattered = glass_shattered + 1
        end
    end

    local clip_dir = string.format("%s/%s", EndRoundCapture.DATASET_DIR, clip.id)
    love.filesystem.createDirectory(clip_dir)

    local bin_payload = table.concat(clip.frame_bytes)
    local ok_bin, err_bin = love.filesystem.write(clip_dir .. "/frames.bin", bin_payload)
    if not ok_bin then
        print("DATASET: Failed writing frames.bin: " .. tostring(err_bin))
        return false
    end

    local roi = mod.config.rois.play_area
    local win_w, win_h = love.graphics.getDimensions()
    local meta = {
        schema_version = 1,
        clip_id = clip.id,
        source = "balatro_mod",
        fps = EndRoundCapture.TARGET_FPS,
        num_frames = num_frames,
        width = EndRoundCapture.OUTPUT_W,
        height = EndRoundCapture.OUTPUT_H,
        frame_format = "raw_rgba_u8",
        already_cropped = true,
        payload_file = "frames.bin",
        roi_base_1080p = {
            x = roi.x,
            y = roi.y,
            w = roi.w,
            h = roi.h,
        },
        events = {
            play_hand_frame = ev.play_hand_frame or 0,
            first_card_in_play_frame = ev.first_card_in_play_frame,
            draw_to_discard_frame = ev.draw_to_discard_frame,
            end_round_frame = ev.end_round_frame,
            cards_cleared_frame = ev.cards_cleared_frame,
            round_eval_enter_frame = ev.round_eval_enter_frame,
            round_eval_settled_frame = ev.round_eval_settled_frame,
        },
        metadata = {
            archetype = clip.scenario.archetype,
            gamespeed = clip.scenario.gamespeed,
            crt = clip.scenario.crt,
            bloom = clip.scenario.bloom,
            reduced_motion = clip.scenario.reduced_motion,
            joker_display_enabled = clip.scenario.joker_display_enabled,
            simulate_hover = clip.scenario.simulate_hover,
            num_jokers_hovered = clip.scenario.num_jokers_hovered or 0,
            hover_frames_total = clip.scenario.hover_frames_total or 0,
            blind_key = clip.scenario.blind_key,
            blind_name = clip.scenario.blind_name,
            blind_chips = clip.scenario.blind_chips,
            initial_chips = clip.scenario.initial_chips,
            num_jokers = clip.scenario.num_jokers,
            num_played_cards = clip.scenario.num_played_cards,
            num_held_cards = clip.scenario.num_held_cards,
            num_glass_shattered = glass_shattered,
            end_round_source = clip.end_round_source or "unknown",
            window_w = win_w,
            window_h = win_h,
            canvas_scale = G.CANV_SCALE or 1,
        },
    }

    local ok_json, err_json = love.filesystem.write(clip_dir .. "/meta.json", encode_json_value(meta, 0) .. "\n")
    if not ok_json then
        print("DATASET: Failed writing meta.json: " .. tostring(err_json))
        pcall(function() love.filesystem.remove(clip_dir .. "/frames.bin") end)
        pcall(function() love.filesystem.remove(clip_dir) end)
        return false
    end

    mod.frame_counts.end_round = (mod.frame_counts.end_round or 0) + 1
    mod.frame_count = mod.frame_counts.end_round
    mod.capture_flash = 2.0
    mod.capture_flash_text = string.format(
        "CLIP %s SAVED (%d frames, END_ROUND@%d, %s)",
        clip.id, num_frames, ev.end_round_frame, clip.scenario.archetype
    )
    return true
end

-- ==========================================================================
-- Public Clip Lifecycle & Main Loop Stepping
-- ==========================================================================

--- Start a randomized clip recording right away (requiring SELECTING_HAND on initial entry).
function EndRoundCapture.begin_next_clip(mod)
    -- Only the initial clip must enter from SELECTING_HAND; continuous Tab batch clips naturally transition
    -- from the previous clip's ROUND_EVAL once user state has already been backed up.
    local continuing_batch = (mod._end_round_backup ~= nil) and (not mod._er_stop_after_one)
    local valid_run = G and G.STAGE == G.STAGES.RUN and G.hand and G.play and G.GAME
    local valid_state = continuing_batch or (G and G.STATE == G.STATES.SELECTING_HAND)
    if not (valid_run and valid_state) then
        EndRoundCapture.stop(mod)
        mod.config.calibration_mode = 'setup'
        mod.capture_flash = 3.0
        mod.capture_flash_text = "ENTER AN ACTIVE BLIND HAND (SELECTING_HAND) BEFORE GENERATING!"
        return false
    end

    EndRoundCapture.install_hooks(mod)

    local scenario = mod.EndRoundGen.setup_clip_scenario(mod)
    local clip_num = mod.frame_counts and mod.frame_counts.end_round or 0
    local tracked = {}
    if G.hand and G.hand.highlighted then
        for _, card in ipairs(G.hand.highlighted) do
            tracked[#tracked + 1] = card
        end
    end

    local clip = {
        id = string.format("clip_%06d", clip_num),
        current_frame_idx = 0,
        frame_bytes = {},
        scenario = scenario,
        hover_schedule = scenario.hover_schedule or {},
        tracked_play_cards = tracked,
        events = {
            play_hand_frame = 0,
        },
        end_round_window = false,
        saw_cards_in_play = false,
        settle_hold_remaining = nil,
        awaiting_capture = false,
    }
    mod._active_end_round_clip = clip
    mod.state = 'recording'

    -- Move cursor to frame-0 position (-100, -100) before playing hand so no residual hover persists.
    if G.CONTROLLER and G.CONTROLLER.set_cursor_position then
        pcall(function() G.CONTROLLER:set_cursor_position() end)
    end

    -- Trigger Play Hand synchronously on clip frame 0.
    if G.FUNCS and G.FUNCS.play_cards_from_highlighted then
        G.FUNCS.play_cards_from_highlighted()
    end
    return true
end

--- Abort any active clip cleanly, leave a playable board in RUN, and restore user settings.
function EndRoundCapture.stop(mod)
    local had_session = (mod._active_end_round_clip ~= nil)
        or (mod._er_stop_after_one ~= nil)
        or (mod._end_round_backup ~= nil)
    mod._active_end_round_clip = nil
    mod._er_stop_after_one = nil
    mod.state = 'idle'
    if mod.EndRoundGen then
        if had_session then
            if G and G.STAGE == G.STAGES.RUN then
                -- Repopulate a playable hand/deck before restoring settings; an empty board in
                -- SELECTING_HAND triggers game.lua:3056 end_round() on the next frame and softlocks ROUND_EVAL.
                mod.EndRoundGen.setup_clip_scenario(mod, { keep_blind_cycle = true })
            else
                mod.EndRoundGen.cleanup_clip_state(mod)
            end
        end
        mod.EndRoundGen.restore_user_state(mod)
    end
end

--- Abort the active clip on error/timeout, cleaning board state and exiting single-clip mode if active.
local function abort_active_clip(mod, reason)
    print(reason)
    local was_single = mod._er_stop_after_one == true
    EndRoundCapture.stop(mod)
    if was_single then
        mod.config.calibration_mode = 'setup'
        mod.capture_flash = 2.0
        mod.capture_flash_text = "SINGLE CLIP ABORTED (SEE LOG)"
    end
end

--- Step one simulated 15fps clip frame (4 fixed 1/60s physics/event substeps) and mark for capture.
local function step_one_clip_frame(mod, orig_love_update)
    local clip = mod._active_end_round_clip
    if not clip then return false end

    local idx = #clip.frame_bytes
    clip.current_frame_idx = idx

    EndRoundCapture.in_internal_step = true
    for _ = 1, EndRoundCapture.SUBSTEPS_PER_FRAME do
        orig_love_update(EndRoundCapture.FIXED_DT)
    end
    EndRoundCapture.in_internal_step = false

    clip.awaiting_capture = true
    return true
end

--- Finish evaluating the just-captured frame and advance or finalize the clip.
local function post_capture_advance(mod)
    local clip = mod._active_end_round_clip
    if not clip then return false end

    local idx = #clip.frame_bytes - 1
    update_visual_boundaries(mod, clip, idx)

    if clip.settle_hold_remaining ~= nil then
        if clip.settle_hold_remaining <= 0 then
            finalize_and_write_clip(mod, clip)
            mod._active_end_round_clip = nil
            mod.state = 'idle'
            if mod._er_stop_after_one then
                mod.config.calibration_mode = 'setup'
                EndRoundCapture.stop(mod)
            end
            return false
        end
        clip.settle_hold_remaining = clip.settle_hold_remaining - 1
    end

    -- Safety aborts if a clip somehow returns to hand selection, dies, or hangs.
    local timed_out = (#clip.frame_bytes >= EndRoundCapture.MAX_CLIP_FRAMES)
    local lost_state = (G.STATE == G.STATES.GAME_OVER)
        or (G.STATE == G.STATES.SELECTING_HAND and idx >= 10 and not clip.end_round_window)

    if timed_out or lost_state then
        abort_active_clip(mod, string.format(
            "DATASET: Aborting clip %s (frames=%d, state=%s, timed_out=%s)",
            clip.id, #clip.frame_bytes, tostring(G.STATE), tostring(timed_out)
        ))
        return false
    end

    return true
end

--- Drive batch fixed-timestep updates inside hooked `love.update(dt)`.
--- Returns true when this update handled end-round fixed stepping (caller skips normal mod:update).
function EndRoundCapture.step_love_update(mod, dt, orig_love_update, orig_love_draw)
    if EndRoundCapture.in_internal_step then
        return false
    end
    if not (mod and mod.config and mod.config.mode == 'end_round_classifier') then
        return false
    end

    if mod.config.calibration_mode ~= 'generation' then
        if mod._active_end_round_clip then
            EndRoundCapture.stop(mod)
        end
        return false
    end

    local max_clips = mod.config.max_clips or 1000
    local cur_clips = (mod.frame_counts and mod.frame_counts.end_round) or 0
    if (not mod._er_stop_after_one) and cur_clips >= max_clips then
        EndRoundCapture.stop(mod)
        mod.config.calibration_mode = 'setup'
        mod.capture_flash = 3.0
        mod.capture_flash_text = string.format("REACHED TARGET CAP (%d CLIPS)! GENERATION STOPPED.", max_clips)
        print(string.format("DATASET: Reached cap of %d clips. Stopped generation.", max_clips))
        return true
    end

    local batch = math.max(1, tonumber(mod.config.end_round_frames_per_tick) or 2)
    for step_i = 1, batch do
        if (not mod._er_stop_after_one) and (mod.frame_counts.end_round or 0) >= max_clips then break end

        if not mod._active_end_round_clip then
            if not EndRoundCapture.begin_next_clip(mod) then
                return true
            end
        end

        if step_one_clip_frame(mod, orig_love_update) then
            if step_i < batch and orig_love_draw then
                -- Render intermediate frames offscreen immediately; final frame in batch waits for LÖVE's love.draw().
                EndRoundCapture.in_internal_draw = true
                orig_love_draw()
                EndRoundCapture.capture_after_draw(mod)
                EndRoundCapture.in_internal_draw = false
                if not mod._active_end_round_clip then
                    break
                end
            end
        end
    end
    return true
end

--- Called right after `orig_love_draw()` inside `love.draw()` (before `draw_debug()`).
function EndRoundCapture.capture_after_draw(mod)
    local clip = mod and mod._active_end_round_clip
    if not (clip and clip.awaiting_capture) then return end
    clip.awaiting_capture = false

    local rgba = capture_play_area_rgba(mod)
    if rgba and #rgba == BYTES_PER_FRAME then
        clip.frame_bytes[#clip.frame_bytes + 1] = rgba
        post_capture_advance(mod)
    else
        abort_active_clip(mod, string.format(
            "DATASET: Frame capture returned invalid size (%s)", tostring(rgba and #rgba)
        ))
    end
end

--- Capture or trigger a single manual test clip in Setup mode (via C key).
function EndRoundCapture.run_single_test_clip(mod)
    if mod.config.mode ~= 'end_round_classifier' then return end
    mod._er_stop_after_one = true
    mod.config.calibration_mode = 'generation'
end

return EndRoundCapture
