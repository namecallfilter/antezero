-- gamestate_dataset_cv/src/cards_gen.lua
-- Contains all hand generation, deck appearance, and card attribute randomization for the card classifier.

local CardsGen = {}

-- Utility for Normal Distribution
function G.DATASET_MOD:normal_dist(mean, std_dev)
    local u1 = math.random()
    local u2 = math.random()
    local z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0 * std_dev + mean
end

function G.DATASET_MOD:get_random_card_center()
    local cards = {}
    if G.P_CARDS then
        for k, v in pairs(G.P_CARDS) do
            if k ~= 'empty' and type(v) == 'table' and v.suit and v.value then
                table.insert(cards, v)
            end
        end
    end
    if #cards > 0 then
        return cards[math.random(#cards)]
    end
    return nil
end

function G.DATASET_MOD:get_random_edition()
    local r = math.random()
    if r < 0.2 then return 'foil'
    elseif r < 0.4 then return 'holo'
    elseif r < 0.6 then return 'polychrome'
    end
    return nil
end

function G.DATASET_MOD:get_random_enhancement()
    if math.random() < 0.45 then return nil end
    local enhs = {
        'm_bonus', 'm_mult', 'm_wild',
        'm_glass', 'm_glass',
        'm_steel', 'm_steel',
        'm_gold', 'm_lucky',
    }
    return enhs[math.random(#enhs)]
end

function G.DATASET_MOD:get_random_seal()
    if math.random() < 0.5 then return nil end
    local seals = {'Gold', 'Red', 'Blue', 'Purple'}
    return seals[math.random(#seals)]
end

function G.DATASET_MOD:randomize_background()
    if G.C and G.C.BACKGROUND then
        local realistic_bgs = {
            -- Green (Default / Main Menu)
            {L = {0.15, 0.35, 0.25, 1}, D = {0.08, 0.25, 0.15, 1}},
            -- Blue (Small Blind)
            {L = {0.15, 0.45, 0.75, 1}, D = {0.05, 0.25, 0.45, 1}},
            -- Orange/Yellow (Big Blind)
            {L = {0.85, 0.55, 0.15, 1}, D = {0.55, 0.25, 0.05, 1}},
            -- Red (Boss Blind)
            {L = {0.75, 0.25, 0.25, 1}, D = {0.45, 0.05, 0.05, 1}},
            -- Purple (Tarot / Magic)
            {L = {0.55, 0.25, 0.65, 1}, D = {0.25, 0.05, 0.35, 1}},
            -- Teal (Planet)
            {L = {0.15, 0.65, 0.55, 1}, D = {0.05, 0.35, 0.25, 1}},
            -- Ghostly Blue (Spectral)
            {L = {0.35, 0.65, 0.75, 1}, D = {0.15, 0.35, 0.45, 1}},
            -- Dark / Boss Special
            {L = {0.2, 0.2, 0.2, 1}, D = {0.05, 0.05, 0.05, 1}}
        }
        
        local bg = realistic_bgs[math.random(#realistic_bgs)]
        local noise = (math.random() - 0.5) * 0.05
        
        G.C.BACKGROUND.L = {math.max(0, math.min(1, bg.L[1] + noise)), math.max(0, math.min(1, bg.L[2] + noise)), math.max(0, math.min(1, bg.L[3] + noise)), 1}
        G.C.BACKGROUND.D = {math.max(0, math.min(1, bg.D[1] + noise)), math.max(0, math.min(1, bg.D[2] + noise)), math.max(0, math.min(1, bg.D[3] + noise)), 1}
        G.C.BACKGROUND.C = G.C.BACKGROUND.L
    end
end

-- Deck Appearance Randomization (High-Contrast & Crossover Arts)
function G.DATASET_MOD:get_all_suits()
    local suits = {}
    local seen = {}
    
    -- 1. Query SMODS suits if registered
    if SMODS and SMODS.Suit and SMODS.Suit.obj_buffer then
        for _, s in ipairs(SMODS.Suit.obj_buffer) do
            if not seen[s] then
                seen[s] = true
                table.insert(suits, s)
            end
        end
    end
    
    -- 2. Query suits registered in G.COLLABS.options
    if G.COLLABS and G.COLLABS.options then
        for s, _ in pairs(G.COLLABS.options) do
            if not seen[s] then
                seen[s] = true
                table.insert(suits, s)
            end
        end
    end
    
    -- 3. Fallback standard suits
    local standard = {'Spades', 'Hearts', 'Clubs', 'Diamonds'}
    for _, s in ipairs(standard) do
        if not seen[s] then
            seen[s] = true
            table.insert(suits, s)
        end
    end
    
    return suits
end

local function is_default_skin(key, suit)
    if not key or type(key) ~= "string" then return true end
    if key == "default" or key == ("default_" .. suit) then return true end
    if string.sub(key, 1, 8) == "default_" then return true end
    return false
end

function G.DATASET_MOD:get_available_crossover_arts(suit)
    local crossover_arts = {}
    local seen = {}
    
    -- 1. Query G.COLLABS.options[suit] dynamically (captures vanilla and any mod-added collabs)
    if G.COLLABS and G.COLLABS.options and G.COLLABS.options[suit] then
        for _, skin_key in ipairs(G.COLLABS.options[suit]) do
            if not is_default_skin(skin_key, suit) and not seen[skin_key] then
                seen[skin_key] = true
                table.insert(crossover_arts, skin_key)
            end
        end
    end
    
    -- 2. Query SMODS.DeckSkins / SMODS.DeckSkin.obj_table directly
    local deck_skins = (SMODS and SMODS.DeckSkins) or (SMODS and SMODS.DeckSkin and SMODS.DeckSkin.obj_table)
    if deck_skins then
        for key, skin_obj in pairs(deck_skins) do
            if skin_obj and skin_obj.suit == suit then
                local k = skin_obj.key or key
                if not is_default_skin(k, suit) and not seen[k] then
                    seen[k] = true
                    table.insert(crossover_arts, k)
                end
            end
        end
    end
    
    return crossover_arts
end

function G.DATASET_MOD:randomize_deck_appearance()
    G.SETTINGS = G.SETTINGS or {}
    G.SETTINGS.CUSTOM_DECK = G.SETTINGS.CUSTOM_DECK or {}
    G.SETTINGS.CUSTOM_DECK.Collabs = G.SETTINGS.CUSTOM_DECK.Collabs or {}
    G.SETTINGS.colour_palettes = G.SETTINGS.colour_palettes or {}
    
    local suits = self:get_all_suits()
    for _, suit in ipairs(suits) do
        -- 1. Randomly toggle high contrast colors on/off for each suit
        local is_hc = (math.random() > 0.5)
        G.SETTINGS.colour_palettes[suit] = is_hc and 'hc' or 'lc'
        
        -- 2. Dynamically pick between available custom face card arts (crossover arts)
        local crossover_arts = self:get_available_crossover_arts(suit)
        local default_skin = (G.COLLABS and G.COLLABS.options and G.COLLABS.options[suit] and G.COLLABS.options[suit][1]) or ('default_' .. suit)
        local chosen_skin = default_skin
        if #crossover_arts > 0 and math.random() >= 0.35 then
            chosen_skin = crossover_arts[math.random(#crossover_arts)]
        end
        G.SETTINGS.CUSTOM_DECK.Collabs[suit] = chosen_skin
        
        -- 3. Update suit color palette
        if G.FUNCS and G.FUNCS.update_suit_colours then
            G.FUNCS.update_suit_colours(suit, chosen_skin)
        end
        if G.C and G.C.SUITS then
            local proto = is_hc and (G.C.SO_2 or G.C.SUITS_HC) or (G.C.SO_1 or G.C.SUITS_LC)
            if proto and proto[suit] then
                G.C.SUITS[suit] = proto[suit]
            end
        end
    end
end

-- Deck back randomization helpers
function G.DATASET_MOD:get_all_deck_keys()
    local decks = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == 'Back' and k ~= 'b_undiscovered' then
                table.insert(decks, k)
            end
        end
    end
    if #decks == 0 then
        decks = {
            'b_red', 'b_blue', 'b_yellow', 'b_green', 'b_black', 'b_magic',
            'b_nebula', 'b_ghost', 'b_abandoned', 'b_checkered', 'b_zodiac',
            'b_painted', 'b_anaglyph', 'b_plasma', 'b_erratic', 'b_challenge'
        }
    end
    table.sort(decks)
    return decks
end

function G.DATASET_MOD:set_random_deck()
    local decks = self:get_all_deck_keys()
    local chosen_deck_key = decks[math.random(#decks)]
    local chosen_center = (G.P_CENTERS and G.P_CENTERS[chosen_deck_key])
    local chosen_pos = chosen_center and chosen_center.pos or {x = 0, y = 0}

    if G.GAME then
        if Back and chosen_center then
            G.GAME.selected_back = Back(chosen_center)
        else
            G.GAME.selected_back = {pos = chosen_pos, effect = {center = chosen_center}}
        end
        G.GAME.selected_back_key = chosen_deck_key
    end
    return chosen_deck_key, chosen_pos
end

function G.DATASET_MOD:generate_random_hand()
    if not G.hand then return end
    
    -- Randomly toggle high contrast colors and pick available crossover face card arts for each suit
    self:randomize_deck_appearance()
    
    -- Reset captured boxes to release all old Card references
    self.captured_boxes = setmetatable({}, {__mode = "k"})
    
    -- Clear current hand
    local ShopGen = G.DATASET_MOD and G.DATASET_MOD.ShopGen
    for i = #G.hand.cards, 1, -1 do
        local c = G.hand.cards[i]
        if ShopGen and ShopGen.fully_remove_card then
            ShopGen.fully_remove_card(c)
        else
            c:remove()
        end
    end
    G.hand.cards = {}
    G.hand.highlighted = {}
    if ShopGen and ShopGen.cleanup_cycle then
        ShopGen.cleanup_cycle()
    end
    
    local num_cards = math.max(1, math.min(15, math.floor(self:normal_dist(8, 2))))
    G.hand.config.card_limit = num_cards
    
    for i=1, num_cards do
        local base_card = self:get_random_card_center()
        local center = G.P_CENTERS.c_base
        
        if math.random() < 0.1 and G.P_CENTERS['m_stone'] then
            center = G.P_CENTERS['m_stone']
        else
            local enhancement = self:get_random_enhancement()
            if enhancement and G.P_CENTERS[enhancement] then
                center = G.P_CENTERS[enhancement]
            end
        end
        
        local card = Card(G.play.T.x + G.play.T.w/2, G.play.T.y, G.CARD_W, G.CARD_H, base_card, center, {bypass_discovery_center = true, bypass_discovery_ui = true})
        
        -- Force re-application of base card to guarantee sprite atlas is loaded (vanilla blank art fix)
        -- Pass initial=true so active blind doesn't auto-debuff cards (debuff is controlled below)
        if base_card and card.set_base then
            card:set_base(base_card, true)
        end
        card.debuff = false
        
        local edition = self:get_random_edition()
        if edition then card:set_edition({[edition] = true}, true, true) end
        
        local seal = self:get_random_seal()
        if seal then card:set_seal(seal, true, true) end
        
        if G.hand.emplace then
            G.hand:emplace(card)
        else
            card.area = G.hand
            table.insert(G.hand.cards, card)
        end
    end
    
    -- Hand modifier event: 10% Debuffed hand, 10% Face-down hand, 80% Normal hand (strictly mutually exclusive)
    local hand_event = math.random()
    if hand_event < 0.10 and #G.hand.cards > 0 then
        -- 1. DEBUFFED HAND (10% of hands): between 1 card and all cards debuffed
        local num_debuffed = math.random(1, #G.hand.cards)
        local indices = {}
        for idx = 1, #G.hand.cards do indices[idx] = idx end
        for idx = #indices, 2, -1 do
            local j = math.random(idx)
            indices[idx], indices[j] = indices[j], indices[idx]
        end
        for i = 1, num_debuffed do
            local c = G.hand.cards[indices[i]]
            if c.set_debuff then
                c:set_debuff(true)
            else
                c.debuff = true
            end
        end
    elseif hand_event < 0.20 and #G.hand.cards > 0 then
        -- 2. FACE-DOWN HAND (10% of hands): between 1 card and all cards face-down
        local chosen_deck_key, chosen_pos = self:set_random_deck()
        for _, c in ipairs(G.hand.cards) do
            if c.children and c.children.back then
                c.children.back:set_sprite_pos(chosen_pos)
            end
        end

        local num_face_down = math.random(1, #G.hand.cards)
        local indices = {}
        for idx = 1, #G.hand.cards do indices[idx] = idx end
        for idx = #indices, 2, -1 do
            local j = math.random(idx)
            indices[idx], indices[j] = indices[j], indices[idx]
        end
        for i = 1, num_face_down do
            local c = G.hand.cards[indices[i]]
            c.facing = 'back'
            c.sprite_facing = 'back'
            c.ability = c.ability or {}
            c.ability.wheel_flipped = true
        end
    end

    -- Randomly select
    local num_selected = math.random(0, math.min(5, num_cards))
    for i=1, num_selected do
        local idx = math.random(1, #G.hand.cards)
        local c = G.hand.cards[idx]
        if not c.highlighted then
            c.highlighted = true
            G.hand.highlighted[#G.hand.highlighted + 1] = c
        end
    end
    
    G.hand:set_ranks()
    if G.hand.sort then G.hand:sort() end
end

function G.DATASET_MOD:randomize_hand_selection()
    if G.hand and G.hand.cards then
        for _, c in ipairs(G.hand.cards) do
            c.highlighted = false
        end
        G.hand.highlighted = {}
        
        local num_selected = math.random(0, math.min(5, #G.hand.cards))
        for i=1, num_selected do
            local idx = math.random(1, #G.hand.cards)
            local c = G.hand.cards[idx]
            if not c.highlighted then
                c.highlighted = true
                G.hand.highlighted[#G.hand.highlighted + 1] = c
            end
        end
    end
end

return CardsGen
