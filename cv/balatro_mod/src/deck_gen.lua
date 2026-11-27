-- gamestate_dataset_cv/src/deck_gen.lua
-- Randomizes G.playing_cards with realistic Balatro deck patterns and opens the
-- View Deck overlay (G.UIDEF.deck_info) for deck_classifier dataset generation.

local DeckGen = {}

local STANDARD_SUITS = {'Spades', 'Hearts', 'Clubs', 'Diamonds'}
local SUIT_PREFIX = {
    Spades = 'S',
    Hearts = 'H',
    Clubs = 'C',
    Diamonds = 'D',
}
local RANK_CODES = {'2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'}
local HIGH_VALUE_RANKS = {'2', 'T', 'J', 'Q', 'K', 'A'}
local FACE_RANK_CODES = {J = true, Q = true, K = true}

-- Playing cards only receive Foil / Holo / Polychrome (never Negative).
local EDITIONS = {'foil', 'holo', 'polychrome'}
local SEALS = {'Gold', 'Red', 'Blue', 'Purple'}
local ENHANCEMENTS = {'m_bonus', 'm_mult', 'm_wild', 'm_glass', 'm_steel', 'm_gold', 'm_lucky', 'm_stone'}

local MIN_DECK_SIZE = 15
local MAX_DECK_SIZE = 100
local MAX_PER_SUIT = 30

-- Base single-card trait rates (duplicated/copied cards use 2x these probabilities).
local BASE_EDITION_PROB = 1.0 / 20.0
local BASE_SEAL_PROB = 1.0 / 10.0
local BASE_ENHANCEMENT_PROB = 1.0 / 3.0

local function clamp(val, lo, hi)
    if val < lo then return lo end
    if val > hi then return hi end
    return val
end

local function shuffle_array(tbl)
    local out = {}
    for i = 1, #tbl do
        out[i] = tbl[i]
    end
    for i = #out, 2, -1 do
        local j = math.random(i)
        out[i], out[j] = out[j], out[i]
    end
    return out
end

local function safe_normal(mean, std_dev)
    local u1 = math.max(1e-7, math.random())
    local u2 = math.random()
    local z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0 * std_dev + mean
end

function DeckGen.sample_deck_size()
    local r = math.random()
    local n
    if r < 0.70 then
        -- Concentrated near standard ~52-card deck size.
        n = clamp(math.floor(safe_normal(52.0, 7.5) + 0.5), 32, 72)
    elseif r < 0.85 then
        -- Thinned / aggressive Hanged Man decks down to 15 cards.
        n = clamp(math.floor(safe_normal(29.0, 7.5) + 0.5), 15, 44)
    else
        -- Bloated / Hologram / DNA / Cryptid decks up to 100 cards.
        n = clamp(math.floor(safe_normal(76.0, 12.0) + 0.5), 56, 100)
    end
    return clamp(n, MIN_DECK_SIZE, MAX_DECK_SIZE)
end

function DeckGen.roll_card_traits(is_copy)
    local mult = is_copy and 2.0 or 1.0
    local edition = nil
    local seal = nil
    local enhancement = nil

    if math.random() < (BASE_EDITION_PROB * mult) then
        edition = EDITIONS[math.random(#EDITIONS)]
    end
    if math.random() < (BASE_SEAL_PROB * mult) then
        seal = SEALS[math.random(#SEALS)]
    end
    if math.random() < (BASE_ENHANCEMENT_PROB * mult) then
        enhancement = ENHANCEMENTS[math.random(#ENHANCEMENTS)]
    end

    return {
        edition = edition,
        seal = seal,
        enhancement = enhancement,
    }
end

local function pick_weighted_suit(eligible_suits, suit_weights)
    local total_w = 0.0
    for _, s in ipairs(eligible_suits) do
        total_w = total_w + math.max(0.0, suit_weights[s] or 0.0)
    end
    if total_w <= 0.0 then
        return eligible_suits[math.random(#eligible_suits)]
    end
    local r = math.random() * total_w
    local acc = 0.0
    for _, s in ipairs(eligible_suits) do
        acc = acc + math.max(0.0, suit_weights[s] or 0.0)
        if r <= acc then
            return s
        end
    end
    return eligible_suits[#eligible_suits]
end

local function pick_rank_code(prefer_high_value)
    if prefer_high_value and math.random() < 0.50 then
        return HIGH_VALUE_RANKS[math.random(#HIGH_VALUE_RANKS)]
    end
    return RANK_CODES[math.random(#RANK_CODES)]
end

function DeckGen.build_deck_plan()
    local total_n = DeckGen.sample_deck_size()
    local perm = shuffle_array(STANDARD_SUITS)

    local roll = math.random()
    local archetype
    local suit_weights = {}
    if roll < 0.22 then
        archetype = 'balanced'
        for _, s in ipairs(STANDARD_SUITS) do
            suit_weights[s] = 0.8 + math.random() * 0.4
        end
    elseif roll < 0.57 then
        archetype = 'single_suit_heavy'
        suit_weights[perm[1]] = 3.5 + math.random() * 2.5
        suit_weights[perm[2]] = 0.3 + math.random() * 0.7
        suit_weights[perm[3]] = 0.3 + math.random() * 0.7
        if total_n <= 90 and math.random() < 0.25 then
            suit_weights[perm[4]] = 0.0
        else
            suit_weights[perm[4]] = 0.3 + math.random() * 0.7
        end
    elseif roll < 0.77 then
        archetype = 'two_suit_heavy'
        suit_weights[perm[1]] = 3.0 + math.random() * 2.0
        suit_weights[perm[2]] = 3.0 + math.random() * 2.0
        if total_n <= 60 and math.random() < 0.50 then
            suit_weights[perm[3]] = 0.0
            suit_weights[perm[4]] = 0.0
        else
            suit_weights[perm[3]] = 0.15 + math.random() * 0.45
            suit_weights[perm[4]] = 0.15 + math.random() * 0.45
        end
    else
        archetype = 'copy_heavy'
        suit_weights[perm[1]] = 2.5 + math.random() * 2.0
        suit_weights[perm[2]] = 1.0 + math.random() * 1.5
        suit_weights[perm[3]] = 0.4 + math.random() * 0.8
        suit_weights[perm[4]] = 0.4 + math.random() * 0.8
    end

    local suit_counts = {Spades = 0, Hearts = 0, Clubs = 0, Diamonds = 0}
    local specs = {}

    -- Step A: Copy/duplicate groups (2..20 copies sharing identical rank/suit/traits, 2x trait odds)
    local num_groups = 0
    if archetype == 'copy_heavy' then
        num_groups = math.random(1, 4)
    elseif archetype == 'single_suit_heavy' or archetype == 'two_suit_heavy' then
        if math.random() < 0.75 then
            num_groups = math.random(1, 3)
        end
    else
        if math.random() < 0.50 then
            num_groups = math.random(1, 2)
        end
    end

    for g_idx = 1, num_groups do
        local rem_total = total_n - #specs
        if rem_total < 2 then break end

        local eligible_suits = {}
        for _, s in ipairs(STANDARD_SUITS) do
            if suit_counts[s] <= MAX_PER_SUIT - 2 then
                table.insert(eligible_suits, s)
            end
        end
        if #eligible_suits == 0 then break end

        local chosen_suit = pick_weighted_suit(eligible_suits, suit_weights)
        local desired_k
        if archetype == 'copy_heavy' then
            desired_k = (g_idx == 1) and math.random(5, 20) or math.random(2, 10)
        elseif archetype == 'single_suit_heavy' or archetype == 'two_suit_heavy' then
            desired_k = (math.random() < 0.25) and math.random(10, 20) or math.random(2, 14)
        else
            desired_k = math.random(2, 8)
        end

        local max_for_suit = MAX_PER_SUIT - suit_counts[chosen_suit]
        local k = math.min(desired_k, max_for_suit, rem_total)
        if k >= 2 then
            local rank_code = pick_rank_code(true)
            local traits = DeckGen.roll_card_traits(true)
            for _ = 1, k do
                table.insert(specs, {
                    suit = chosen_suit,
                    rank_code = rank_code,
                    edition = traits.edition,
                    enhancement = traits.enhancement,
                    seal = traits.seal,
                    is_copy = true,
                })
            end
            suit_counts[chosen_suit] = suit_counts[chosen_suit] + k
        end
    end

    -- Step B: Fill remaining slots proportionally to suit weights while enforcing <= 30 per suit
    local rem = total_n - #specs
    local add_counts = {Spades = 0, Hearts = 0, Clubs = 0, Diamonds = 0}
    for _ = 1, rem do
        local eligible_weighted = {}
        local eligible_any = {}
        for _, s in ipairs(STANDARD_SUITS) do
            local cap = MAX_PER_SUIT - (suit_counts[s] + add_counts[s])
            if cap > 0 then
                table.insert(eligible_any, s)
                if (suit_weights[s] or 0) > 0 then
                    table.insert(eligible_weighted, s)
                end
            end
        end
        local pool = (#eligible_weighted > 0) and eligible_weighted or eligible_any
        if #pool == 0 then break end
        local chosen_suit = pick_weighted_suit(pool, suit_weights)
        add_counts[chosen_suit] = add_counts[chosen_suit] + 1
    end

    for _, s in ipairs(STANDARD_SUITS) do
        local count_s = add_counts[s]
        if count_s > 0 then
            local shuffled_ranks = shuffle_array(RANK_CODES)
            for idx = 1, count_s do
                local rank_code
                if idx <= #shuffled_ranks then
                    rank_code = shuffled_ranks[idx]
                else
                    rank_code = pick_rank_code(false)
                end
                local traits = DeckGen.roll_card_traits(false)
                table.insert(specs, {
                    suit = s,
                    rank_code = rank_code,
                    edition = traits.edition,
                    enhancement = traits.enhancement,
                    seal = traits.seal,
                    is_copy = false,
                })
            end
            suit_counts[s] = suit_counts[s] + count_s
        end
    end

    return specs, archetype, suit_counts
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

local function snap_card_area_and_cards(area)
    if not area then return end
    if area.align_to_major then pcall(area.align_to_major, area) end
    if area.align_cards then pcall(area.align_cards, area) end
    if area.hard_set_T then pcall(area.hard_set_T, area) end
    if area.hard_set_VT then pcall(area.hard_set_VT, area) end
    if area.hard_set_cards then pcall(area.hard_set_cards, area) end
    if area.VT and area.T then
        area.VT.x = area.T.x
        area.VT.y = area.T.y
        area.VT.w = area.T.w
        area.VT.h = area.T.h
        area.VT.scale = area.T.scale
        area.VT.r = area.T.r
    end
    if area.cards then
        for _, c in ipairs(area.cards) do
            if c.velocity then
                c.velocity.x = 0
                c.velocity.y = 0
                c.velocity.r = 0
                c.velocity.scale = 0
            end
            if c.hard_set_T then pcall(c.hard_set_T, c) end
            if c.hard_set_VT then pcall(c.hard_set_VT, c) end
            if c.VT and c.T then
                c.VT.x = c.T.x
                c.VT.y = c.T.y
                c.VT.w = c.T.w
                c.VT.h = c.T.h
                c.VT.scale = c.T.scale
                c.VT.r = c.T.r
            end
            if c.children then
                for _, child in pairs(c.children) do
                    if child and child.hard_set_VT then
                        pcall(child.hard_set_VT, child)
                    end
                    if child and child.VT and child.T then
                        child.VT.x = child.T.x
                        child.VT.y = child.T.y
                        child.VT.w = child.T.w
                        child.VT.h = child.T.h
                        child.VT.scale = child.T.scale
                        child.VT.r = child.T.r
                    end
                end
            end
        end
    end
end

local function snap_overlay_tree(node, visited, found_areas)
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
        if obj.config and obj.config.view_deck then
            snap_card_area_and_cards(obj)
            if found_areas then
                table.insert(found_areas, obj)
            end
        elseif obj.update_text then
            snap_dynatext(obj)
        elseif obj.hard_set_VT then
            pcall(obj.hard_set_VT, obj)
        end
        if type(obj) == "table" and (obj.UIRoot or obj.children) then
            snap_overlay_tree(obj, visited, found_areas)
        end
    end

    if node.UIRoot then
        snap_overlay_tree(node.UIRoot, visited, found_areas)
    end
    if node.children then
        for _, child in pairs(node.children) do
            snap_overlay_tree(child, visited, found_areas)
        end
    end
end

function DeckGen.get_active_deck_areas(mod)
    local areas = {}
    if mod and mod._active_deck_areas and #mod._active_deck_areas > 0 then
        for _, area in ipairs(mod._active_deck_areas) do
            if area and not area.REMOVED and not area.removed and area.cards then
                table.insert(areas, area)
            end
        end
    end

    if #areas == 0 and G and G.I and G.I.CARDAREA then
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

function G.DATASET_MOD:cleanup_deck_overlay(restore_original_deck)
    local ShopGen = self.ShopGen

    -- 1. Clear all cards inside active view_deck CardAreas first
    local active_areas = DeckGen.get_active_deck_areas(self)
    for _, area in ipairs(active_areas) do
        if ShopGen and ShopGen.clear_card_area then
            ShopGen.clear_card_area(area)
        elseif area.cards then
            for i = #area.cards, 1, -1 do
                local c = area.cards[i]
                if c and c.remove then pcall(c.remove, c) end
            end
            area.cards = {}
        end
    end
    self._active_deck_areas = nil

    -- 2. Remove the overlay menu UIBox
    if G then
        if G.OVERLAY_MENU and G.OVERLAY_MENU.remove then
            pcall(G.OVERLAY_MENU.remove, G.OVERLAY_MENU)
        end
        G.OVERLAY_MENU = nil
        G.VIEWING_DECK = nil
        if G.SETTINGS then
            G.SETTINGS.paused = false
        end
        if G.ROOM then
            G.ROOM.jiggle = 0
        end
    end

    self.captured_boxes = setmetatable({}, {__mode = "k"})

    -- 3. Restore or clear G.playing_cards BEFORE removing synthetic cards to prevent O(N^2) scans
    if G then
        if restore_original_deck and self._saved_playing_cards then
            G.playing_cards = self._saved_playing_cards
            self._saved_playing_cards = nil
        elseif not restore_original_deck and self._deck_gen_cards then
            G.playing_cards = {}
        end
    end

    -- 4. Remove synthetic G.playing_cards created during setup_deck_screen
    if self._deck_gen_cards then
        for i = #self._deck_gen_cards, 1, -1 do
            local c = self._deck_gen_cards[i]
            if c then
                -- Never allow Card:remove / remove_from_area to mutate real G.deck or G.discard
                c.area = nil
                if ShopGen and ShopGen.fully_remove_card then
                    ShopGen.fully_remove_card(c)
                elseif c.remove then
                    pcall(c.remove, c)
                end
            end
        end
        self._deck_gen_cards = nil
    end

    if ShopGen and ShopGen.cleanup_cycle then
        ShopGen.cleanup_cycle()
    else
        collectgarbage("collect")
    end
end

function G.DATASET_MOD:setup_deck_screen(phase_override)
    if not G or not G.GAME or not G.UIDEF or not G.UIDEF.deck_info then
        self.capture_flash = 2.0
        self.capture_flash_text = "START A RUN FIRST BEFORE GENERATING DECK DATA!"
        return false
    end

    -- Preserve player's actual deck on first entry so leaving deck_classifier restores it cleanly
    if self._saved_playing_cards == nil then
        self._saved_playing_cards = G.playing_cards
    end

    self:cleanup_deck_overlay(false)

    if self.randomize_deck_appearance then
        self:randomize_deck_appearance()
    end
    if self.randomize_background then
        self:randomize_background()
    end

    local specs, archetype, suit_counts = DeckGen.build_deck_plan()
    local phase = phase_override
    if not phase then
        phase = (math.random() < 0.75) and 'full' or 'remaining'
    end
    self.deck_phase = phase
    local show_remaining = (phase == 'remaining')

    -- Optional Boss Blind suit / face debuff simulation (10% of frames)
    local debuff_suit = nil
    local debuff_faces = false
    local debuff_roll = math.random()
    if debuff_roll < 0.05 then
        debuff_suit = STANDARD_SUITS[math.random(#STANDARD_SUITS)]
    elseif debuff_roll < 0.10 then
        debuff_faces = true
    end

    local new_playing_cards = {}
    local tracked_cards = {}
    local remaining_discard_ratio = show_remaining and (0.10 + math.random() * 0.35) or 0.0

    for _, spec in ipairs(specs) do
        local prefix = SUIT_PREFIX[spec.suit] or 'S'
        local card_key = prefix .. '_' .. spec.rank_code
        local base_card = (G.P_CARDS and G.P_CARDS[card_key]) or self:get_random_card_center()
        local center = (spec.enhancement and G.P_CENTERS and G.P_CENTERS[spec.enhancement])
            or (G.P_CENTERS and G.P_CENTERS.c_base)

        local card = Card(0, 0, G.CARD_W, G.CARD_H, base_card, center, {
            bypass_discovery_center = true,
            bypass_discovery_ui = true,
        })
        if base_card and card.set_base then
            card:set_base(base_card, true)
        end

        local is_debuffed = false
        if debuff_suit and spec.suit == debuff_suit then
            is_debuffed = true
        elseif debuff_faces and FACE_RANK_CODES[spec.rank_code] then
            is_debuffed = true
        end
        card.debuff = is_debuffed

        if spec.edition then
            card:set_edition({[spec.edition] = true}, true, true)
        end
        if spec.seal then
            card:set_seal(spec.seal, true, true)
        end

        -- G.UIDEF.view_deck(true) greys out cards whose area is not G.deck
        if show_remaining and G.discard and math.random() < remaining_discard_ratio then
            card.area = G.discard
        else
            card.area = G.deck
        end

        table.insert(new_playing_cards, card)
        table.insert(tracked_cards, card)
    end

    self._deck_gen_cards = tracked_cards
    G.playing_cards = new_playing_cards

    local ok_def, def = pcall(G.UIDEF.deck_info, show_remaining)
    if not ok_def or not def or not UIBox then
        self:cleanup_deck_overlay(true)
        self.capture_flash = 2.0
        self.capture_flash_text = "FAILED TO BUILD VIEW DECK OVERLAY!"
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
        self:cleanup_deck_overlay(true)
        return false
    end

    G.OVERLAY_MENU = box
    G.SETTINGS.paused = true
    G.VIEWING_DECK = true
    G.OVERLAY_MENU.alignment.offset.y = 0
    if G.ROOM then
        G.ROOM.jiggle = 0
    end
    if G.OVERLAY_MENU.align_to_major then
        pcall(G.OVERLAY_MENU.align_to_major, G.OVERLAY_MENU)
    end
    if G.OVERLAY_MENU.recalculate then
        pcall(G.OVERLAY_MENU.recalculate, G.OVERLAY_MENU)
    end

    local found_areas = {}
    snap_overlay_tree(G.OVERLAY_MENU, {}, found_areas)
    for _, area in ipairs(found_areas) do
        snap_card_area_and_cards(area)
    end
    table.sort(found_areas, function(a, b)
        local ay = (a.VT and a.VT.y) or (a.T and a.T.y) or 0
        local by = (b.VT and b.VT.y) or (b.T and b.T.y) or 0
        return ay < by
    end)
    self._active_deck_areas = found_areas
    self.captured_boxes = setmetatable({}, {__mode = "k"})

    self.last_deck_summary = string.format(
        "%d cards (%s | S:%d H:%d C:%d D:%d | %s)",
        #new_playing_cards,
        archetype,
        suit_counts.Spades or 0,
        suit_counts.Hearts or 0,
        suit_counts.Clubs or 0,
        suit_counts.Diamonds or 0,
        show_remaining and "REM" or "FULL"
    )
    return true
end

function G.DATASET_MOD:generate_deck_step()
    return self:setup_deck_screen()
end

function G.DATASET_MOD:toggle_deck_screen_mode()
    local next_phase = ((self.deck_phase or 'full') == 'full') and 'remaining' or 'full'
    local ok = self:setup_deck_screen(next_phase)
    if ok then
        self.capture_flash = 1.2
        self.capture_flash_text = "VIEW DECK PHASE: " .. string.upper(next_phase)
    end
end

return DeckGen
