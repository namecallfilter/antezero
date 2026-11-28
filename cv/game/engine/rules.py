import copy

from .context import TriggerContext, TriggerEvent
from .jokers import (
    JOKER_EFFECTS,
    dispatch_jokers,
    get_card_repetitions,
    get_oops_multiplier,
    on_joker_added,
    on_joker_removed,
    resolve_copy_target,
    sync_dynamic_joker_states,
)
from .poker_hands import evaluate_poker_hand, get_scoring_cards

try:
    from game.engine.rng_sim.events import (
        evaluate_lucky_hit,
        get_ancient_joker_suit,
        get_blocked_card_names,
        get_castle_suit,
        get_idol_card,
        get_mail_in_rebate_target,
        get_next_voucher,
        poll_edition,
        poll_joker_pool,
        poll_planet_pool,
        poll_tarot_pool,
        predict_ankh_choice,
        predict_familiar_spawns,
        predict_grim_spawns,
        predict_hook_discards,
        predict_immolate_destructions,
        predict_incantation_spawns,
        predict_ouija_rank,
        predict_purple_seal_tarot,
        predict_random_destroy,
        state_has_showman,
    )
    from game.engine.rng_sim.state import SimDeck
except ImportError:
    from engine.rng_sim.events import (
        evaluate_lucky_hit,
        get_ancient_joker_suit,
        get_blocked_card_names,
        get_castle_suit,
        get_idol_card,
        get_mail_in_rebate_target,
        get_next_voucher,
        poll_edition,
        poll_joker_pool,
        poll_planet_pool,
        poll_tarot_pool,
        predict_ankh_choice,
        predict_familiar_spawns,
        predict_grim_spawns,
        predict_hook_discards,
        predict_immolate_destructions,
        predict_incantation_spawns,
        predict_ouija_rank,
        predict_purple_seal_tarot,
        predict_random_destroy,
        state_has_showman,
    )
    from engine.rng_sim.state import SimDeck

try:
    from game.balatro_models import (
        Action,
        ActionType,
        BlindName,
        BoosterPack,
        BoosterSize,
        BoosterType,
        CardType,
        Deck,
        Edition,
        Enhancement,
        Joker,
        JokerName,
        Location,
        Planet,
        PlanetName,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        Spectral,
        SpectralName,
        Stake,
        Suit,
        Tarot,
        TarotName,
        Voucher,
        VoucherName,
    )
except ImportError:
    from balatro_models import (
        Action,
        ActionType,
        BlindName,
        BoosterPack,
        BoosterSize,
        BoosterType,
        CardType,
        Deck,
        Edition,
        Enhancement,
        Joker,
        JokerName,
        Location,
        Planet,
        PlanetName,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        Spectral,
        SpectralName,
        Stake,
        Suit,
        Tarot,
        TarotName,
        Voucher,
        VoucherName,
    )

POKER_HAND_TO_PLANET = {
    PokerHand.PAIR: PlanetName.MERCURY,
    PokerHand.THREE_OF_A_KIND: PlanetName.VENUS,
    PokerHand.TWO_PAIR: PlanetName.URANUS,
    PokerHand.FULL_HOUSE: PlanetName.EARTH,
    PokerHand.FLUSH: PlanetName.JUPITER,
    PokerHand.STRAIGHT: PlanetName.SATURN,
    PokerHand.FOUR_OF_A_KIND: PlanetName.MARS,
    PokerHand.STRAIGHT_FLUSH: PlanetName.NEPTUNE,
    PokerHand.FIVE_OF_A_KIND: PlanetName.PLANET_X,
    PokerHand.FLUSH_HOUSE: PlanetName.CERES,
    PokerHand.FLUSH_FIVE: PlanetName.ERIS,
    PokerHand.HIGH_CARD: PlanetName.PLUTO,
}
PLANET_TO_POKER_HAND = {v: k for k, v in POKER_HAND_TO_PLANET.items()}

_TAG_PACK_SPECS: dict[SkipTag, tuple[BoosterType, BoosterSize, int, int]] = {
    SkipTag.CHARM_TAG: (BoosterType.ARCANA, BoosterSize.MEGA, 2, 5),
    SkipTag.METEOR_TAG: (BoosterType.CELESTIAL, BoosterSize.MEGA, 2, 5),
    SkipTag.ETHEREAL_TAG: (BoosterType.SPECTRAL, BoosterSize.REGULAR, 1, 2),
    SkipTag.STANDARD_TAG: (BoosterType.STANDARD, BoosterSize.MEGA, 2, 5),
    SkipTag.BUFFOON_TAG: (BoosterType.BUFFOON, BoosterSize.MEGA, 2, 4),
}

_EDITION_TAG_MAP: dict[SkipTag, Edition] = {
    SkipTag.FOIL_TAG: Edition.FOIL,
    SkipTag.HOLOGRAPHIC_TAG: Edition.HOLOGRAPHIC,
    SkipTag.POLYCHROME_TAG: Edition.POLYCHROME,
    SkipTag.NEGATIVE_TAG: Edition.NEGATIVE,
}

SHOWDOWN_BOSSES = {
    BlindName.AMBER_ACORN,
    BlindName.VERDANT_LEAF,
    BlindName.VIOLET_VESSEL,
    BlindName.CRIMSON_HEART,
    BlindName.CERULEAN_BELL,
}


def calculate_base_round_payout(state) -> int:
    """Computes deterministic end-of-round base blind reward + remaining hands bonus + interest."""
    dollars = 0

    # 1. Base blind reward
    if state.current_blind == BlindName.SMALL:
        if state.stake == Stake.WHITE:
            dollars += 3
    elif state.current_blind == BlindName.BIG:
        dollars += 4
    elif state.current_blind in SHOWDOWN_BOSSES:
        dollars += 8
    else:
        dollars += 5

    # 2. Remaining hands & discards bonus
    if state.deck == Deck.GREEN_DECK:
        dollars += state.hands_left * 2 + state.discards_left * 1
    else:
        dollars += state.hands_left * 1

    # 3. Interest (calculated on state.money after rental deductions, before adding round payout)
    if state.deck != Deck.GREEN_DECK and state.money >= 5:
        if VoucherName.MONEY_TREE in state.vouchers_redeemed:
            interest_cap = 100
        elif VoucherName.SEED_MONEY in state.vouchers_redeemed:
            interest_cap = 50
        else:
            interest_cap = 25

        to_the_moon_count = sum(
            1
            for j in state.inventory.jokers
            if j.name == JokerName.TO_THE_MOON and not j.debuffed
        )
        interest_amount = 1 + to_the_moon_count
        dollars += interest_amount * min(state.money // 5, interest_cap // 5)

    return dollars


def trigger_end_of_round_effects(state, action: Action | None = None) -> None:
    """Executes Balatro's end_round() effects when a blind is beaten (or on Cash Out if not yet triggered)."""
    if getattr(state, "round_won", False):
        return
    state.round_won = True

    # 1. Held-in-hand end-of-round card effects (Gold cards + Red Seal + Mime retriggers)
    jokers = state.inventory.jokers if state.inventory else []
    mime_count = 0
    for j in jokers:
        if j.debuffed:
            continue
        if j.name == JokerName.MIME:
            mime_count += 1
        elif j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
            target = resolve_copy_target(j, jokers)
            if target and not target.debuffed and target.name == JokerName.MIME:
                mime_count += 1

    for hc in state.hand.cards:
        if hc.selected:
            continue
        c = hc.card
        if getattr(c, "debuffed", False):
            continue
        reps = 1 + (1 if c.seal == Seal.RED else 0) + mime_count
        if c.enhancement == Enhancement.GOLD:
            state.money += 3 * reps
        if c.seal == Seal.BLUE and state.inventory:
            for _ in range(reps):
                if state.inventory.can_add_consumable:
                    p_name = POKER_HAND_TO_PLANET.get(
                        getattr(state, "last_hand_played", PokerHand.HIGH_CARD),
                        PlanetName.PLUTO,
                    )
                    state.inventory.consumables.append(Planet(name=p_name))

    # 2. Unused discards, rental, perishable, and round-end Joker effects
    state.unused_discards += state.discards_left
    state.free_rerolls = 0
    for j in jokers:
        if j.rental:
            state.money -= 3
        if j.perishable:
            j.perish_tally = max(0, j.perish_tally - 1)
            if j.perish_tally == 0:
                j.debuffed = True

    if getattr(state, "rng", None):
        sdeck = SimDeck(state.full_deck)
        # balatro_source/functions/state_events.lua:248-276 runs ease_ante(1) on Boss Blinds
        # before resetting Idol, Mail-In Rebate, Ancient Joker, and Castle targets.
        target_ante = (
            state.ante + 1
            if state.current_blind not in (BlindName.SMALL, BlindName.BIG)
            else state.ante
        )
        state.ancient_suit = get_ancient_joker_suit(
            state.rng, ante=target_ante, current_suit=state.ancient_suit
        )
        idol_target = get_idol_card(state.rng, sdeck, ante=target_ante)
        state.idol_card = (idol_target.rank, idol_target.suit)
        state.castle_suit = get_castle_suit(state.rng, sdeck, ante=target_ante)
        state.mail_card_rank = get_mail_in_rebate_target(
            state.rng, sdeck, ante=target_ante
        )

    ctx = TriggerContext(
        event=TriggerEvent.END_OF_ROUND,
        state=state,
        action=action,
        extra={"phase": "end_of_round", "round_targets_rolled": True},
    )
    dispatch_jokers(ctx)

    # Revert temporary Juggle Tag hand size bonus at end of round
    if getattr(state, "temp_hand_size_bonus", 0) > 0:
        state.hand.max_size = max(0, state.hand.max_size - state.temp_hand_size_bonus)
        state.temp_hand_size_bonus = 0

    # Anaglyph Deck grants a Double Tag after defeating a Boss Blind
    if getattr(
        state, "deck", None
    ) == Deck.ANAGLYPH_DECK and state.current_blind not in (
        BlindName.SMALL,
        BlindName.BIG,
    ):
        _grant_skip_tag(state, SkipTag.DOUBLE_TAG, action=action)

    # Restore Boss Blind round modifiers & clear per-ante state on Boss defeat
    if hasattr(state, "clear_boss_round_modifiers"):
        state.clear_boss_round_modifiers()
    if state.current_blind not in (BlindName.SMALL, BlindName.BIG):
        state.played_cards_this_ante.clear()
        if hasattr(state, "update_ox_target_hand"):
            state.update_ox_target_hand()

    state.discard_pile.clear()


def _evaluate_played_cards(state, action: Action, played_cards: list) -> None:
    """Runs the scoring pipeline for a played hand and triggers end_round() if the blind is beaten.

    Note: Chip and mult score calculation (including The Flint's 0.5x base Chips/Mult) is
    intentionally omitted here because round scores are obtained via OCR.
    """
    state._last_play_lucky_triggers = 0
    state._last_play_lucky_money_hits = 0
    if state.current_blind == BlindName.THE_TOOTH and not state.boss_disabled:
        state.money -= len(played_cards)

    jokers = state.inventory.jokers if state.inventory else None
    scoring_cards = get_scoring_cards(played_cards, jokers)
    poker_hand = evaluate_poker_hand(played_cards, jokers)

    is_repeat_eye = (
        state.current_blind == BlindName.THE_EYE
        and poker_hand in state.hands_played_this_round_by_type
    )
    is_wrong_mouth = (
        state.current_blind == BlindName.THE_MOUTH
        and state.locked_hand_type is not None
        and poker_hand != state.locked_hand_type
    )
    is_small_psychic = (
        state.current_blind == BlindName.THE_PSYCHIC and len(played_cards) < 5
    )
    hand_debuffed = (not state.boss_disabled) and (
        is_repeat_eye or is_wrong_mouth or is_small_psychic
    )

    state.last_hand_played = poker_hand
    state.hand_play_counts[poker_hand] = state.hand_play_counts.get(poker_hand, 0) + 1
    state.hands_played_this_round_by_type.add(poker_hand)
    if (
        state.current_blind == BlindName.THE_MOUTH
        and not state.boss_disabled
        and state.locked_hand_type is None
    ):
        state.locked_hand_type = poker_hand

    if hand_debuffed:
        ctx_debuffed = TriggerContext(
            event=TriggerEvent.DEBUFFED_HAND,
            state=state,
            action=action,
            scoring_cards=scoring_cards,
            full_hand=played_cards,
            poker_hand=poker_hand,
            boss_triggered=True,
        )
        dispatch_jokers(ctx_debuffed)
    else:
        boss_triggered = False
        if not state.boss_disabled:
            if state.current_blind == BlindName.THE_ARM:
                cur_lvl = state.hand_levels.get(poker_hand, 1)
                if cur_lvl > 1:
                    boss_triggered = True
                    state.hand_levels[poker_hand] = cur_lvl - 1
            elif state.current_blind == BlindName.THE_OX:
                ox_target = (
                    getattr(state, "ox_target_hand", None) or PokerHand.HIGH_CARD
                )
                if poker_hand == ox_target:
                    boss_triggered = True
                    state.money = 0
            elif state.current_blind == BlindName.THE_FLINT:
                boss_triggered = True

            if any(
                getattr(card, "debuffed", False)
                or (
                    hasattr(state, "card_is_debuffed_by_boss")
                    and state.card_is_debuffed_by_boss(card)
                )
                for card in scoring_cards
            ):
                boss_triggered = True

        # 1. Before play
        ctx = TriggerContext(
            event=TriggerEvent.BEFORE_PLAY,
            state=state,
            action=action,
            scoring_cards=scoring_cards,
            full_hand=played_cards,
            poker_hand=poker_hand,
        )
        dispatch_jokers(ctx)

        # 2. Individual card scored (for each active, non-debuffed scoring card + retriggers)
        for idx, card in enumerate(scoring_cards):
            if getattr(card, "debuffed", False):
                continue
            reps = get_card_repetitions(card, idx, state)
            for _ in range(reps):
                if getattr(card, "destroyed", False):
                    break
                if card.seal == Seal.GOLD:
                    state.money += 3
                extra = {}
                if card.enhancement == Enhancement.LUCKY and state.rng:
                    state._last_play_lucky_triggers += 1
                    hit = evaluate_lucky_hit(
                        state.rng, probability_modifier=get_oops_multiplier(state)
                    )
                    if hit.mult:
                        card.bonus_mult = getattr(card, "bonus_mult", 0) + 20
                    if hit.money:
                        state.money += 20
                        state._last_play_lucky_money_hits += 1
                    extra["lucky_triggered"] = hit.any_hit

                ctx_ind = TriggerContext(
                    event=TriggerEvent.INDIVIDUAL_CARD_SCORED,
                    state=state,
                    action=action,
                    target_card=card,
                    scoring_cards=scoring_cards,
                    full_hand=played_cards,
                    poker_hand=poker_hand,
                    extra=extra,
                )
                dispatch_jokers(ctx_ind)

        # 3. Held-in-hand individual card effects (Reserved Parking, Baron, Shoot the Moon retriggers)
        active_jokers = state.inventory.jokers if state.inventory else []
        mime_count = 0
        for j in active_jokers:
            if j.debuffed:
                continue
            if j.name == JokerName.MIME:
                mime_count += 1
            elif j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
                target = resolve_copy_target(j, active_jokers)
                if target and not target.debuffed and target.name == JokerName.MIME:
                    mime_count += 1

        held_cards = [hc.card for hc in state.hand.cards if not hc.selected]
        for card in held_cards:
            if getattr(card, "debuffed", False):
                continue
            reps = 1 + (1 if card.seal == Seal.RED else 0) + mime_count
            for _ in range(reps):
                ctx_held = TriggerContext(
                    event=TriggerEvent.HELD_CARD_SCORED,
                    state=state,
                    action=action,
                    target_card=card,
                    scoring_cards=scoring_cards,
                    full_hand=played_cards,
                    poker_hand=poker_hand,
                )
                dispatch_jokers(ctx_held)

        # 4. Glass cards shatter check (evaluated once per scoring card after repetitions)
        for card in scoring_cards:
            if getattr(card, "debuffed", False) or getattr(card, "destroyed", False):
                continue
            if card.enhancement == Enhancement.GLASS and state.rng:
                if state.rng.poll_odds("glass", 1, 4, get_oops_multiplier(state)):
                    card.destroyed = True
                    state.remove_card_from_deck(card, shattered=True)

        # 4b. Main Joker scoring triggers (e.g. Matador on Boss Blind triggered)
        ctx_main = TriggerContext(
            event=TriggerEvent.JOKER_MAIN,
            state=state,
            action=action,
            scoring_cards=scoring_cards,
            full_hand=played_cards,
            poker_hand=poker_hand,
            boss_triggered=boss_triggered,
        )
        dispatch_jokers(ctx_main)

        # 5. After play
        ctx_after = TriggerContext(
            event=TriggerEvent.AFTER_PLAY,
            state=state,
            action=action,
            scoring_cards=scoring_cards,
            full_hand=played_cards,
            poker_hand=poker_hand,
        )
        dispatch_jokers(ctx_after)

    state.hand.cards = [c for c in state.hand.cards if not c.selected]
    for card in played_cards:
        if not getattr(card, "destroyed", False):
            state.played_cards_this_round.append(card)
            state.played_cards_this_ante.append(card)
            state.discard_pile.append(card)
    state.update_face_down_candidates()
    state.reconcile_face_down_adjacencies()

    if (action.metadata and action.metadata.get("won_round")) or (
        state.target_chips > 0 and state.current_chips >= state.target_chips
    ):
        trigger_end_of_round_effects(state, action)


def _filter_non_surviving_hand_cards(
    hand_cards: list[PlayingCard], action: Action | None, *, min_needed: int
) -> list[PlayingCard]:
    """Prefers hand cards not witnessed surviving on the next frame when resolving random destroys."""
    surviving = (
        action.metadata.get("_surviving_hand_ids")
        if action and action.metadata
        else None
    )
    if not surviving:
        return hand_cards
    surviving_set = set(surviving)
    pool = [c for c in hand_cards if c.id not in surviving_set]
    return pool if len(pool) >= min_needed else hand_cards


def _has_joker(state, joker_name: JokerName) -> bool:
    """Checks whether state holds an active (undebuffed) copy of joker_name."""
    if hasattr(state, "has_joker"):
        return state.has_joker(joker_name)
    if not state.inventory:
        return False
    return any(j.name == joker_name and not j.debuffed for j in state.inventory.jokers)


def _spawn_joker_to_inventory(
    state,
    *,
    rarity_tier: int | None = None,
    legendary: bool = False,
    key_append: str = "",
) -> Joker | None:
    """Rolls a Joker via Balatro PRNG into inventory if space allows."""
    if not state.inventory or not state.inventory.can_add_joker:
        return None
    if not getattr(state, "rng", None):
        state.pending_cv_checks.add("jokers")
        return None
    j = poll_joker_pool(
        state.rng,
        ante=_effective_ante(state),
        rarity_tier=rarity_tier,
        legendary=legendary,
        key_append=key_append,
        owned_jokers=get_blocked_card_names(state, CardType.JOKER),
        gros_michel_extinct=getattr(state, "gros_michel_extinct", False),
        has_showman=state_has_showman(state),
        edition_mod=getattr(state, "edition_rate", 1.0),
    )
    state.inventory.jokers.append(j)
    on_joker_added(state, j)
    return j


def _consume_tag(state, tag: SkipTag) -> int:
    """Removes and returns count of `tag` from state.active_tags."""
    count = sum(1 for t in state.active_tags if t == tag)
    if count:
        state.active_tags = [t for t in state.active_tags if t != tag]
    return count


def _open_pack_on_state(
    state,
    action: Action | None,
    pack: BoosterPack,
    pre_pack_location: Location,
) -> None:
    """Opens a booster pack immediately or queues it behind the active pack."""
    if state.active_pack is not None or state.location == Location.PACK:
        state.pending_packs.append(pack)
        return
    state.pre_pack_location = pre_pack_location
    state.active_pack = pack
    state.pack_choices_remaining = pack.choose
    state.location = Location.PACK
    ctx = TriggerContext(event=TriggerEvent.OPEN_BOOSTER, state=state, action=action)
    dispatch_jokers(ctx)


def _close_pack_or_advance_queue(state, action: Action | None) -> None:
    """Closes current pack, opening any queued pack or restoring pre-pack location."""
    if getattr(state, "pending_packs", None):
        next_pack = state.pending_packs.pop(0)
        state.active_pack = next_pack
        state.pack_cards.clear()
        state.pack_choices_remaining = next_pack.choose
        state.location = Location.PACK
        ctx = TriggerContext(
            event=TriggerEvent.OPEN_BOOSTER, state=state, action=action
        )
        dispatch_jokers(ctx)
        return
    state.active_pack = None
    state.pack_cards.clear()
    state.pack_choices_remaining = 0
    # Restore pre-pack location (Location.SELECT for Skip Tag packs, Location.SHOP for shop packs)
    state.location = getattr(state, "pre_pack_location", None) or Location.SHOP
    state.pre_pack_location = None


def _apply_discard_cards(
    state,
    action: Action,
    discarded_cards: list,
    is_hook: bool = False,
) -> None:
    """Shared discard pipeline for normal Discard and Boss Blind The Hook forced discards."""
    for card in discarded_cards:
        if (
            card.seal == Seal.PURPLE
            and not getattr(card, "debuffed", False)
            and state.inventory
            and state.inventory.can_add_consumable
        ):
            if state.rng:
                pred = predict_purple_seal_tarot(
                    state.rng,
                    ante=state.ante,
                    active_tarots=get_blocked_card_names(state, CardType.TAROT),
                    has_showman=state_has_showman(state),
                )
                state.inventory.consumables.append(pred.to_tarot_card())
            else:
                state.pending_cv_checks.add("consumables")

    ctx = TriggerContext(
        event=TriggerEvent.DISCARD,
        state=state,
        action=action,
        full_hand=discarded_cards,
        hook=is_hook,
    )
    dispatch_jokers(ctx)

    discarded_ids = {id(c) for c in discarded_cards}
    state.hand.cards = [
        hc for hc in state.hand.cards if id(hc.card) not in discarded_ids
    ]
    for card in discarded_cards:
        if not getattr(card, "destroyed", False):
            state.discard_pile.append(card)

    state.update_face_down_candidates()
    state.reconcile_face_down_adjacencies()


def _advance_round_if_needed(state) -> None:
    """Advances state.round once per round (primary END_ROUND, fallback CASH_OUT)."""
    if not getattr(state, "round_advanced", False):
        state.round += 1
        state.round_advanced = True


def _effective_ante(state) -> int:
    """Returns the live Balatro ante for PRNG pool keys, accounting for post-Boss shop advance."""
    if getattr(state, "pending_blind_advance", False) and state.current_blind not in (
        BlindName.SMALL,
        BlindName.BIG,
    ):
        return state.ante + 1
    return state.ante


def _advance_blind_and_ante_if_pending(state, force: bool = False) -> None:
    """Advances Small->Big->Boss->Next Ante(+1) when leaving Shop or on fallback."""
    if not force and not getattr(state, "pending_blind_advance", False):
        return
    state.pending_blind_advance = False

    if state.current_blind == BlindName.SMALL:
        state.current_blind = BlindName.BIG
    elif state.current_blind == BlindName.BIG:
        state.current_blind = state.boss_blind or BlindName.THE_HOOK
    else:
        state.ante += 1
        state.current_blind = BlindName.SMALL
        if state.seed or getattr(state, "rng", None):
            if hasattr(state, "roll_boss_blind"):
                state.roll_boss_blind(is_reroll=False)
            if hasattr(state, "roll_blind_tags"):
                state.roll_blind_tags()

    if hasattr(state, "compute_target_chips"):
        state.target_chips = state.compute_target_chips()


def _trigger_single_skip_tag(
    state,
    tag: SkipTag,
    action: Action | None = None,
    orbital_hand: PokerHand | None = None,
    pre_pack_location: Location = Location.SELECT,
) -> None:
    """Applies one occurrence of a non-Double SkipTag."""
    if tag == SkipTag.TOP_UP_TAG:
        for _ in range(2):
            _spawn_joker_to_inventory(state, rarity_tier=1, key_append="top")
    elif tag == SkipTag.SPEED_TAG:
        state.money += 5 * state.blinds_skipped
    elif tag == SkipTag.GARBAGE_TAG:
        state.money += getattr(state, "unused_discards", 0)
    elif tag == SkipTag.HANDY_TAG:
        state.money += sum(getattr(state, "hand_play_counts", {}).values())
    elif tag == SkipTag.ECONOMY_TAG:
        state.money += min(40, max(0, state.money))
    elif tag == SkipTag.ORBITAL_TAG:
        chosen_hand = orbital_hand
        if chosen_hand is None and action and isinstance(action.metadata, dict):
            raw = action.metadata.get("orbital_hand")
            if isinstance(raw, PokerHand):
                chosen_hand = raw
            elif isinstance(raw, str):
                try:
                    chosen_hand = PokerHand(raw)
                except ValueError:
                    chosen_hand = None
        if chosen_hand is None and getattr(state, "rng", None):
            visible = [
                h
                for h in PokerHand
                if h
                not in (
                    PokerHand.FIVE_OF_A_KIND,
                    PokerHand.FLUSH_HOUSE,
                    PokerHand.FLUSH_FIVE,
                )
                or state.hand_play_counts.get(h, 0) > 0
            ]
            chosen_hand, _ = state.rng.choice("orbital", visible)
        if chosen_hand is not None:
            state.hand_levels[chosen_hand] = state.hand_levels.get(chosen_hand, 1) + 3
        else:
            state.pending_cv_checks.add("hand_levels")
    elif tag in _TAG_PACK_SPECS:
        b_type, b_size, choose, extra = _TAG_PACK_SPECS[tag]
        pack = BoosterPack(
            booster_type=b_type, size=b_size, cost=0, choose=choose, extra=extra
        )
        _open_pack_on_state(state, action, pack, pre_pack_location=pre_pack_location)
    elif tag == SkipTag.BOSS_TAG:
        if hasattr(state, "roll_boss_blind"):
            state.roll_boss_blind(is_reroll=True)
            if (
                state.current_blind not in (BlindName.SMALL, BlindName.BIG)
                and state.boss_blind
            ):
                state.current_blind = state.boss_blind
    else:
        state.active_tags.append(tag)


def _grant_skip_tag(
    state,
    tag: SkipTag | None,
    action: Action | None = None,
    orbital_hand: PokerHand | None = None,
    pre_pack_location: Location = Location.SELECT,
) -> None:
    """Grants a SkipTag, consuming any active Double Tags to duplicate non-Double tags."""
    if tag is None:
        return
    if tag == SkipTag.DOUBLE_TAG:
        state.active_tags.append(SkipTag.DOUBLE_TAG)
        return
    doubles = _consume_tag(state, SkipTag.DOUBLE_TAG)
    for _ in range(1 + doubles):
        _trigger_single_skip_tag(
            state,
            tag,
            action=action,
            orbital_hand=orbital_hand,
            pre_pack_location=pre_pack_location,
        )


def _apply_shop_entry_tags(state) -> None:
    """Consumes shop-triggered SkipTags upon entering the shop."""
    if _consume_tag(state, SkipTag.COUPON_TAG):
        state.coupon_shop_active = True
        for card in state.shop_cards:
            card.couponed = True
        for pack in state.shop_packs:
            pack.couponed = True

    voucher_count = _consume_tag(state, SkipTag.VOUCHER_TAG)
    for _ in range(voucher_count):
        if getattr(state, "rng", None):
            pred = get_next_voucher(
                state.rng,
                ante=state.ante,
                from_tag=True,
                used_vouchers=set(state.vouchers_redeemed),
                shop_vouchers=[v.name for v in state.shop_vouchers],
            )
            state.shop_vouchers.append(pred.to_model())
        else:
            state.pending_cv_checks.add("shop_vouchers")

    for tag, tier, append in (
        (SkipTag.UNCOMMON_TAG, 2, "uta"),
        (SkipTag.RARE_TAG, 3, "rta"),
    ):
        count = _consume_tag(state, tag)
        for _ in range(count):
            if getattr(state, "rng", None):
                j = poll_joker_pool(
                    state.rng,
                    ante=state.ante,
                    rarity_tier=tier,
                    key_append=append,
                    owned_jokers=get_blocked_card_names(state, CardType.JOKER),
                    gros_michel_extinct=getattr(state, "gros_michel_extinct", False),
                    has_showman=state_has_showman(state),
                    edition_mod=getattr(state, "edition_rate", 1.0),
                )
                j.couponed = True
                state.shop_cards.append(j)
            else:
                state.pending_cv_checks.add("shop_cards")

    remaining_tags: list[SkipTag] = []
    for tag in state.active_tags:
        ed = _EDITION_TAG_MAP.get(tag)
        if ed is None:
            remaining_tags.append(tag)
            continue
        applied = False
        for card in state.shop_cards:
            if card.type == CardType.JOKER and card.edition == Edition.NONE:
                card.edition = ed
                card.couponed = True
                applied = True
                break
        if not applied:
            remaining_tags.append(tag)
            state.pending_cv_checks.add("shop_cards")
    state.active_tags = remaining_tags


def _apply_consumable_state(state, card, action: Action | None = None) -> None:
    """Updates consumable-related GameState memory (tarots_used, last_tarot_planet, unique_planets_used, ectoplasm_minus, deck mutations)."""
    if card is None:
        return
    c_type = getattr(card, "type", None)
    c_name = getattr(card, "name", None)
    if c_type == CardType.TAROT:
        state.tarots_used += 1
        if c_name == TarotName.FOOL:
            last = state.last_tarot_planet
            if (
                last is not None
                and last != TarotName.FOOL
                and state.inventory
                and state.inventory.can_add_consumable
            ):
                if isinstance(last, PlanetName):
                    state.inventory.consumables.append(Planet(name=last))
                elif isinstance(last, TarotName):
                    state.inventory.consumables.append(Tarot(name=last))
        elif c_name == TarotName.EMPEROR:
            if state.inventory:
                has_showman = state_has_showman(state)
                for _ in range(2):
                    if not state.inventory.can_add_consumable:
                        break
                    if state.rng:
                        pred = poll_tarot_pool(
                            state.rng,
                            ante=state.ante,
                            key_append="emp",
                            active_tarots=get_blocked_card_names(state, CardType.TAROT),
                            has_showman=has_showman,
                        )
                        state.inventory.consumables.append(pred.to_tarot_card())
                    else:
                        state.pending_cv_checks.add("consumables")
        elif c_name == TarotName.HIGH_PRIESTESS:
            if state.inventory:
                has_showman = state_has_showman(state)
                for _ in range(2):
                    if not state.inventory.can_add_consumable:
                        break
                    if state.rng:
                        planet = poll_planet_pool(
                            state.rng,
                            ante=_effective_ante(state),
                            key_append="pri",
                            hand_play_counts=state.hand_play_counts,
                            active_planets=get_blocked_card_names(state, CardType.PLANET),
                            has_showman=has_showman,
                        )
                        state.inventory.consumables.append(planet)
                    else:
                        state.pending_cv_checks.add("consumables")
        elif c_name == TarotName.JUDGEMENT:
            _spawn_joker_to_inventory(state, key_append="jud")
        elif c_name == TarotName.HANGED_MAN:
            selected = [
                (hc._real_card if hc.facing_back and hc._real_card else hc.card)
                for hc in state.hand.cards
                if hc.selected
            ]
            is_inferred = bool(
                action and action.metadata and action.metadata.get("inferred")
            )
            to_destroy = (
                selected[:2]
                if selected
                else ([] if is_inferred else [hc.card for hc in state.hand.cards[:2]])
            )
            for dc in to_destroy:
                state.remove_card_from_deck(dc, shattered=True)
            state.hand.cards = [
                hc for hc in state.hand.cards if hc.card not in to_destroy
            ]
            state.update_face_down_candidates()
        elif c_name == TarotName.DEATH:
            selected = [hc for hc in state.hand.cards if hc.selected]
            if len(selected) >= 2:
                left, right = selected[0], selected[1]
                source_card = right._real_card if right._real_card else right.card
                if (
                    source_card.rank != Rank.NONE
                    or source_card.enhancement == Enhancement.STONE
                ):
                    left.resolve_face_down(source_card, preserve_id=True)
                    state.sync_card_to_deck(
                        left._real_card
                        if left.facing_back and left._real_card
                        else left.card,
                        rebind_hand_card=left,
                    )
                if right.facing_back:
                    right.reveal_temporarily()
                state.update_face_down_candidates()
        elif c_name == TarotName.STRENGTH:
            selected = [hc for hc in state.hand.cards if hc.selected][:2]
            rank_order = [
                Rank.TWO,
                Rank.THREE,
                Rank.FOUR,
                Rank.FIVE,
                Rank.SIX,
                Rank.SEVEN,
                Rank.EIGHT,
                Rank.NINE,
                Rank.TEN,
                Rank.JACK,
                Rank.QUEEN,
                Rank.KING,
                Rank.ACE,
            ]
            next_rank = {
                rank_order[i]: rank_order[(i + 1) % len(rank_order)]
                for i in range(len(rank_order))
            }
            for hc in selected:
                if hc.facing_back:
                    if hc._real_card and hc._real_card.rank in next_rank:
                        hc._real_card.rank = next_rank[hc._real_card.rank]
                    elif hc.card.rank in next_rank:
                        hc.card.rank = next_rank[hc.card.rank]
                    hc.reveal_temporarily()
                elif hc.card.rank in next_rank:
                    hc.card.rank = next_rank[hc.card.rank]
                state.sync_card_to_deck(
                    hc._real_card if hc.facing_back and hc._real_card else hc.card,
                    rebind_hand_card=hc,
                )
            state.update_face_down_candidates()
        elif c_name in (
            TarotName.MAGICIAN,
            TarotName.EMPRESS,
            TarotName.HIEROPHANT,
            TarotName.CHARIOT,
            TarotName.JUSTICE,
            TarotName.TOWER,
            TarotName.LOVERS,
            TarotName.DEVIL,
        ):
            enh_map = {
                TarotName.MAGICIAN: Enhancement.LUCKY,
                TarotName.EMPRESS: Enhancement.MULT,
                TarotName.HIEROPHANT: Enhancement.BONUS,
                TarotName.CHARIOT: Enhancement.STEEL,
                TarotName.JUSTICE: Enhancement.GLASS,
                TarotName.TOWER: Enhancement.STONE,
                TarotName.LOVERS: Enhancement.WILD,
                TarotName.DEVIL: Enhancement.GOLD,
            }
            target_enh = enh_map[c_name]
            max_cards = (
                1
                if c_name
                in (
                    TarotName.CHARIOT,
                    TarotName.JUSTICE,
                    TarotName.TOWER,
                    TarotName.LOVERS,
                    TarotName.DEVIL,
                )
                else 2
            )
            selected = [hc for hc in state.hand.cards if hc.selected][:max_cards]
            for hc in selected:
                hc.card.enhancement = target_enh
                if hc._real_card:
                    hc._real_card.enhancement = target_enh
                if hc.facing_back:
                    hc.reveal_temporarily()
                elif hc.bounds:
                    hc.bounds.possible_enhancements = {target_enh}
                state.sync_card_to_deck(
                    hc._real_card if hc.facing_back and hc._real_card else hc.card,
                    rebind_hand_card=hc,
                )
            state.update_face_down_candidates()
        elif c_name in (TarotName.SUN, TarotName.MOON, TarotName.STAR, TarotName.WORLD):
            suit_map = {
                TarotName.SUN: Suit.HEARTS,
                TarotName.MOON: Suit.CLUBS,
                TarotName.STAR: Suit.DIAMONDS,
                TarotName.WORLD: Suit.SPADES,
            }
            target_suit = suit_map[c_name]
            selected = [hc for hc in state.hand.cards if hc.selected][:3]
            for hc in selected:
                hc.card.suit = target_suit
                if hc._real_card:
                    hc._real_card.suit = target_suit
                if hc.facing_back:
                    hc.reveal_temporarily()
                elif hc.bounds:
                    hc.bounds.possible_suits = {target_suit}
                state.sync_card_to_deck(
                    hc._real_card if hc.facing_back and hc._real_card else hc.card,
                    rebind_hand_card=hc,
                )
            state.update_face_down_candidates()
        elif c_name == TarotName.WHEEL_OF_FORTUNE:
            if state.rng:
                oops = get_oops_multiplier(state)
                if state.rng.poll_odds("wheel_of_fortune", 1, 4, oops):
                    eligible = [
                        j for j in state.inventory.jokers if j.edition == Edition.NONE
                    ]
                    if eligible:
                        eligible_sorted = sorted(
                            eligible, key=lambda x: getattr(x, "sort_id", 0) or 0
                        )
                        chosen, _ = state.rng.choice(
                            "wheel_of_fortune", eligible_sorted
                        )
                        ed = poll_edition(
                            state.rng, "wheel_of_fortune", no_neg=True, guaranteed=True
                        )
                        chosen.edition = ed
            else:
                state.pending_cv_checks.add("jokers")
        elif c_name == TarotName.HERMIT:
            state.money += max(0, min(state.money, 20))
        elif c_name == TarotName.TEMPERANCE:
            if state.inventory:
                has_astro = _has_joker(state, JokerName.ASTRONOMER)
                total_sell = sum(
                    j.get_sell_value(
                        discount_percent=state.discount_percent,
                        inflation=state.inflation,
                        astronomer=has_astro,
                    )
                    for j in state.inventory.jokers
                )
                state.money += min(50, total_sell)
        if isinstance(c_name, TarotName):
            state.last_tarot_planet = c_name
    elif c_type == CardType.PLANET:
        if isinstance(c_name, PlanetName):
            state.last_tarot_planet = c_name
            target_hand = PLANET_TO_POKER_HAND.get(c_name)
            if target_hand is not None:
                state.hand_levels[target_hand] = (
                    state.hand_levels.get(target_hand, 1) + 1
                )
        state.unique_planets_used.add(str(c_name))
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.BLACK_HOLE:
        for h in PokerHand:
            state.hand_levels[h] = state.hand_levels.get(h, 1) + 1
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.THE_SOUL:
        _spawn_joker_to_inventory(state, legendary=True, key_append="sou")
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.WRAITH:
        state.money = 0
        _spawn_joker_to_inventory(state, rarity_tier=3, key_append="wra")
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.ANKH:
        if state.inventory and state.inventory.jokers:
            if state.rng:
                chosen = predict_ankh_choice(state.rng, state.inventory.jokers)
                for j in list(state.inventory.jokers):
                    if j is not chosen and not getattr(j, "eternal", False):
                        state.inventory.jokers.remove(j)
                        on_joker_removed(state, j)
                if chosen is not None:
                    clone = copy.deepcopy(chosen)
                    clone.id = 0
                    if clone.edition == Edition.NEGATIVE:
                        clone.edition = Edition.NONE
                    state.inventory.jokers.append(clone)
                    on_joker_added(state, clone)
            else:
                state.pending_cv_checks.add("jokers")
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.OUIJA:
        rank = predict_ouija_rank(state.rng) if state.rng else Rank.ACE
        for hc in state.hand.cards:
            hc.card.rank = rank
            if hc._real_card:
                hc._real_card.rank = rank
            if hc.facing_back:
                hc.reveal_temporarily()
            elif hc.bounds:
                hc.bounds.possible_ranks = {rank}
            state.sync_card_to_deck(
                hc._real_card if hc.facing_back and hc._real_card else hc.card,
                rebind_hand_card=hc,
            )
        state.hand.max_size = max(0, state.hand.max_size - 1)
        state.update_face_down_candidates()
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.ECTOPLASM:
        state.hand.max_size = max(0, state.hand.max_size - state.ectoplasm_minus)
        state.ectoplasm_minus += 1
        if state.inventory:
            eligible = [j for j in state.inventory.jokers if j.edition == Edition.NONE]
            if eligible:
                if state.rng:
                    eligible_sorted = sorted(
                        eligible, key=lambda x: getattr(x, "sort_id", 0) or 0
                    )
                    chosen, _ = state.rng.choice("ectoplasm", eligible_sorted)
                    chosen.edition = Edition.NEGATIVE
                else:
                    state.pending_cv_checks.add("jokers")
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.HEX:
        if state.inventory:
            eligible = [j for j in state.inventory.jokers if j.edition == Edition.NONE]
            if eligible:
                if state.rng:
                    eligible_sorted = sorted(
                        eligible, key=lambda x: getattr(x, "sort_id", 0) or 0
                    )
                    chosen, _ = state.rng.choice("hex", eligible_sorted)
                    chosen.edition = Edition.POLYCHROME
                    for j in list(state.inventory.jokers):
                        if j != chosen and not getattr(j, "eternal", False):
                            state.inventory.jokers.remove(j)
                else:
                    state.pending_cv_checks.add("jokers")
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.CRYPTID:
        selected = [c for c in state.hand.cards if c.selected]
        is_inferred = bool(
            action and action.metadata and action.metadata.get("inferred")
        )
        target_hc = (
            selected[0]
            if selected
            else (
                None
                if is_inferred
                else (state.hand.cards[0] if state.hand.cards else None)
            )
        )
        if target_hc:
            if target_hc.facing_back:
                target_hc.reveal_temporarily()
            source = target_hc._real_card or target_hc.card
            if source.rank != Rank.NONE or source.enhancement == Enhancement.STONE:
                state.add_card_to_deck(copy.deepcopy(source), add_to_hand=True)
                state.add_card_to_deck(copy.deepcopy(source), add_to_hand=True)
            else:
                state.deck_cards_count += 2
        elif not is_inferred:
            state.deck_cards_count += 2
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.AURA:
        selected = [hc for hc in state.hand.cards if hc.selected][:1]
        if selected:
            target_hc = selected[0]
            ed = Edition.POLYCHROME
            if state.rng:
                ed = poll_edition(state.rng, "aura", no_neg=True, guaranteed=True)
            target_hc.card.edition = ed
            if target_hc._real_card:
                target_hc._real_card.edition = ed
            if target_hc.facing_back:
                target_hc.reveal_temporarily()
            state.sync_card_to_deck(
                target_hc._real_card
                if target_hc.facing_back and target_hc._real_card
                else target_hc.card,
                rebind_hand_card=target_hc,
            )
            state.update_face_down_candidates()
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.SIGIL:
        suit = Suit.SPADES
        if state.rng:
            suit = state.rng.choice(
                "sigil", [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
            )[0]
        for hc in state.hand.cards:
            hc.card.suit = suit
            if hc._real_card:
                hc._real_card.suit = suit
            if hc.facing_back:
                hc.reveal_temporarily()
            elif hc.bounds:
                hc.bounds.possible_suits = {suit}
            state.sync_card_to_deck(
                hc._real_card if hc.facing_back and hc._real_card else hc.card,
                rebind_hand_card=hc,
            )
        state.update_face_down_candidates()
    elif c_type == CardType.SPECTRAL and c_name in (
        SpectralName.GRIM,
        SpectralName.FAMILIAR,
        SpectralName.INCANTATION,
    ):
        if state.rng:
            hand_cards = _filter_non_surviving_hand_cards(
                [hc.card for hc in state.hand.cards], action, min_needed=1
            )
            victim = predict_random_destroy(state.rng, hand_cards)
            if victim is not None:
                state.remove_card_from_deck(victim)
            if c_name == SpectralName.FAMILIAR:
                new_cards = predict_familiar_spawns(state.rng)
            elif c_name == SpectralName.GRIM:
                new_cards = predict_grim_spawns(state.rng)
            else:
                new_cards = predict_incantation_spawns(state.rng)
            for nc in new_cards:
                state.add_card_to_deck(nc, add_to_hand=True)
        else:
            victim = state.hand.cards[0].card if state.hand.cards else None
            if victim is not None:
                state.remove_card_from_deck(victim)
                state.hand.cards.pop(0)
            else:
                state.remove_card_from_deck(
                    PlayingCard(rank=Rank.ACE, suit=Suit.SPADES)
                )
            spawn_count = (
                2
                if c_name == SpectralName.GRIM
                else (3 if c_name == SpectralName.FAMILIAR else 4)
            )
            spawn_rank = (
                Rank.ACE
                if c_name == SpectralName.GRIM
                else (Rank.KING if c_name == SpectralName.FAMILIAR else Rank.TWO)
            )
            for _ in range(spawn_count):
                state.add_card_to_deck(
                    PlayingCard(
                        rank=spawn_rank, suit=Suit.SPADES, enhancement=Enhancement.BONUS
                    ),
                    add_to_hand=True,
                )
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.IMMOLATE:
        if state.rng:
            hand_cards = _filter_non_surviving_hand_cards(
                [hc.card for hc in state.hand.cards], action, min_needed=5
            )
            destroyed = predict_immolate_destructions(state.rng, hand_cards, count=5)
            for dc in destroyed:
                state.remove_card_from_deck(dc)
            state.money += 20
        else:
            to_destroy = [hc.card for hc in state.hand.cards[:5]]
            for dc in to_destroy:
                state.remove_card_from_deck(dc)
            for _ in range(max(0, 5 - len(to_destroy))):
                state.remove_card_from_deck(
                    PlayingCard(rank=Rank.ACE, suit=Suit.SPADES)
                )
            state.hand.cards = state.hand.cards[5:]
            state.money += 20
    elif c_type == CardType.SPECTRAL and c_name in (
        SpectralName.TALISMAN,
        SpectralName.DEJA_VU,
        SpectralName.TRANCE,
        SpectralName.MEDIUM,
    ):
        seal_map = {
            SpectralName.TALISMAN: Seal.GOLD,
            SpectralName.DEJA_VU: Seal.RED,
            SpectralName.TRANCE: Seal.BLUE,
            SpectralName.MEDIUM: Seal.PURPLE,
        }
        target_seal = seal_map[c_name]
        selected = [hc for hc in state.hand.cards if hc.selected][:1]
        for hc in selected:
            hc.card.seal = target_seal
            if hc._real_card:
                hc._real_card.seal = target_seal
            if hc.facing_back:
                hc.reveal_temporarily()
            elif hc.bounds:
                hc.bounds.possible_seals = {target_seal}
            state.sync_card_to_deck(
                hc._real_card if hc.facing_back and hc._real_card else hc.card,
                rebind_hand_card=hc,
            )
        state.update_face_down_candidates()


def _leave_shop(state, action: Action, *, force_advance: bool = False) -> None:
    """Dispatches `LEAVE_SHOP` hooks, clears shop shelves, advances blind/ante, and enters `Location.SELECT`."""
    state.coupon_shop_active = False
    ctx = TriggerContext(event=TriggerEvent.LEAVE_SHOP, state=state, action=action)
    dispatch_jokers(ctx)
    state.shop_cards.clear()
    state.shop_vouchers.clear()
    state.shop_packs.clear()
    _advance_blind_and_ante_if_pending(
        state,
        force=(
            force_advance
            or state.location == Location.SHOP
            or state.current_blind not in (BlindName.SMALL, BlindName.BIG)
        ),
    )
    state.location = Location.SELECT


def on_action(state, action: Action):
    """Executes deterministic game engine state transitions and triggers."""
    # Implicit joker reordering: if the action specifies a joker order, apply it before evaluating the action!
    j_order = action.joker_order or (
        action.metadata and action.metadata.get("joker_order")
    )
    if j_order and state.inventory and state.inventory.jokers:
        if hasattr(state, "ensure_joker_ids"):
            state.ensure_joker_ids()
        current_jokers = state.inventory.jokers
        by_id = {j.id: j for j in current_jokers if j.id != 0}
        if (
            len(j_order) == len(current_jokers)
            and len(set(j_order)) == len(j_order)
            and all(jid in by_id for jid in j_order)
        ):
            state.inventory.jokers = [by_id[jid] for jid in j_order]
        elif (
            len(j_order) == len(current_jokers)
            and len(set(j_order)) == len(j_order)
            and all(
                isinstance(x, int) and 0 <= x < len(current_jokers) for x in j_order
            )
        ):
            state.inventory.jokers = [current_jokers[i] for i in j_order]
        sync_dynamic_joker_states(state)

    # Implicit hand reordering: if the action specifies a hand order, apply it before evaluating the action!
    h_order = action.hand_order or (
        action.metadata and action.metadata.get("hand_order")
    )
    if h_order and state.hand and state.hand.cards:
        current_hand = state.hand.cards
        if len(h_order) == len(current_hand) and all(
            isinstance(x, int) and 0 <= x < len(current_hand) for x in h_order
        ):
            state.hand.cards = [current_hand[i] for i in h_order]
        else:
            by_id = {c.card.id: c for c in current_hand if c.card and c.card.id != 0}
            if all(cid in by_id for cid in h_order):
                state.hand.cards = [by_id[cid] for cid in h_order]
        state.reconcile_face_down_adjacencies()

    if state.location == Location.PACK and action.type in (
        ActionType.BUY_CARD,
        ActionType.BUY_AND_USE,
        ActionType.REROLL_SHOP,
        ActionType.REDEEM_VOUCHER,
        ActionType.OPEN_PACK,
        ActionType.NEXT_ROUND,
        ActionType.SELECT_BLIND,
        ActionType.SKIP_BLIND,
        ActionType.REROLL_BOSS,
    ):
        state.active_pack = None
        state.pack_cards.clear()
        state.pack_choices_remaining = 0
        if hasattr(state, "pending_packs") and state.pending_packs is not None:
            state.pending_packs.clear()
        if action.type in (
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.REROLL_BOSS,
        ):
            state.location = getattr(state, "pre_pack_location", None) or Location.SHOP
        else:
            state.location = Location.SHOP
        state.pre_pack_location = None

    match action.type:
        case ActionType.SORT_RANK:
            state.apply_sort_observation(
                sort_type="rank",
                selected_slot_uid=action.metadata.get("selected_slot_uid")
                if action.metadata
                else None,
            )

        case ActionType.SORT_SUIT:
            state.apply_sort_observation(
                sort_type="suit",
                selected_slot_uid=action.metadata.get("selected_slot_uid")
                if action.metadata
                else None,
            )

        case ActionType.PLAY_HAND:
            _advance_blind_and_ante_if_pending(state)
            state.hands_left = max(0, state.hands_left - 1)
            state.hands_played_this_round += 1

            for c in state.hand.cards:
                c.forced_selection = False
                if c.selected and c.facing_back and c._real_card:
                    c.resolve_face_down(c._real_card, lift_facing_back=True)
                    state.sync_card_to_deck(c.card, rebind_hand_card=c)

            if state.current_blind == BlindName.THE_FISH and not state.boss_disabled:
                state.fish_prepped = True
            if (
                state.current_blind == BlindName.CRIMSON_HEART
                and not state.boss_disabled
                and state.inventory
                and state.inventory.jokers
            ):
                state.crimson_heart_prepped = True

            played_cards = [c.card for c in state.hand.cards if c.selected]
            state.hand.cards = [c for c in state.hand.cards if not c.selected]
            if action.metadata and action.metadata.get("defer_play_for_hook"):
                state._hook_pending_cards = played_cards
            else:
                # The Hook forces up to 2 remaining hand cards to be discarded via PRNG before scoring
                if (
                    state.current_blind == BlindName.THE_HOOK
                    and not state.boss_disabled
                    and state.hand.cards
                ):
                    hook_cards: list[PlayingCard] = []
                    override_ids = (
                        list(action.metadata.get("_hook_discard_ids") or [])
                        if action.metadata
                        else []
                    )
                    if override_ids:
                        used_ptrs: set[int] = set()
                        for target_id in override_ids:
                            if len(hook_cards) >= 2:
                                break
                            for hc in state.hand.cards:
                                if id(hc) not in used_ptrs and hc.card.id == target_id:
                                    hook_cards.append(hc.card)
                                    used_ptrs.add(id(hc))
                                    break
                        remaining_for_hook = [
                            hc for hc in state.hand.cards if id(hc) not in used_ptrs
                        ]
                        need = min(2, len(state.hand.cards)) - len(hook_cards)
                        if need > 0 and remaining_for_hook:
                            extra = predict_hook_discards(
                                state.rng, remaining_for_hook, count=need
                            )
                            hook_cards.extend(hc.card for hc in extra)
                    else:
                        hook_hand_cards = predict_hook_discards(
                            state.rng, state.hand.cards, count=2
                        )
                        hook_cards = [hc.card for hc in hook_hand_cards]
                    _apply_discard_cards(state, action, hook_cards, is_hook=True)
                _evaluate_played_cards(state, action, played_cards)

        case ActionType.DISCARD:
            is_hook = bool(action.metadata and action.metadata.get("hook"))
            if not is_hook:
                _advance_blind_and_ante_if_pending(state)
                state.discards_left = max(0, state.discards_left - 1)
                state.discards_used_this_round += 1
                for c in state.hand.cards:
                    c.forced_selection = False

            for c in state.hand.cards:
                if c.selected and c.facing_back and c._real_card:
                    c.resolve_face_down(c._real_card, lift_facing_back=True)
                    state.sync_card_to_deck(c.card, rebind_hand_card=c)

            discarded_cards = [c.card for c in state.hand.cards if c.selected]
            _apply_discard_cards(state, action, discarded_cards, is_hook=is_hook)

            pending_hook_cards = getattr(state, "_hook_pending_cards", None)
            if is_hook and pending_hook_cards is not None:
                state._hook_pending_cards = None
                _evaluate_played_cards(state, action, pending_hook_cards)

        case ActionType.REROLL_SHOP:
            state.coupon_shop_active = False
            if state.free_rerolls > 0:
                state.free_rerolls -= 1
            else:
                state.money -= state.reroll_cost
                state.reroll_cost += 1

            ctx = TriggerContext(
                event=TriggerEvent.REROLL_SHOP, state=state, action=action
            )
            dispatch_jokers(ctx)

        case ActionType.BUY_CARD:
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_cards
            ):
                card = state.shop_cards.pop(action.target_idx)
                if getattr(state, "coupon_shop_active", False):
                    card.couponed = True
                cost = card.get_cost(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                    astronomer=_has_joker(state, JokerName.ASTRONOMER),
                )
                state.money -= cost
                extra = {}
                if card.type == CardType.JOKER:
                    state.inventory.jokers.append(card)
                    on_joker_added(state, card)
                elif card.type in (CardType.TAROT, CardType.PLANET, CardType.SPECTRAL):
                    state.inventory.consumables.append(card)
                elif card.type == CardType.PLAYING_CARD:
                    state.add_card_to_deck(card, add_to_hand=False)
                    extra["already_added_to_deck"] = True
                ctx = TriggerContext(
                    event=TriggerEvent.CARD_BOUGHT,
                    state=state,
                    action=action,
                    target_card=card,
                    extra=extra,
                )
                dispatch_jokers(ctx)

        case ActionType.BUY_AND_USE:
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_cards
            ):
                card = state.shop_cards.pop(action.target_idx)
                if getattr(state, "coupon_shop_active", False):
                    card.couponed = True
                cost = card.get_cost(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                    astronomer=_has_joker(state, JokerName.ASTRONOMER),
                )
                state.money -= cost
                _apply_consumable_state(state, card, action=action)
                ctx = TriggerContext(
                    event=TriggerEvent.USE_CONSUMABLE,
                    state=state,
                    action=action,
                    target_card=card,
                )
                dispatch_jokers(ctx)

        case ActionType.USE_CONSUMABLE:
            used_card = None
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.inventory.consumables
            ):
                used_card = state.inventory.consumables.pop(action.target_idx)
            if used_card is not None:
                _apply_consumable_state(state, used_card, action=action)
                ctx = TriggerContext(
                    event=TriggerEvent.USE_CONSUMABLE,
                    state=state,
                    action=action,
                    target_card=used_card,
                )
                dispatch_jokers(ctx)

        case ActionType.OPEN_PACK:
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_packs
            ):
                pack = state.shop_packs.pop(action.target_idx)
                if getattr(state, "coupon_shop_active", False):
                    pack.couponed = True
                cost = pack.get_cost(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                    astronomer=_has_joker(state, JokerName.ASTRONOMER),
                )
                state.money -= cost
                _open_pack_on_state(
                    state, action, pack, pre_pack_location=Location.SHOP
                )

        case ActionType.USE_IN_PACK:
            chosen_card = None
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.pack_cards
            ):
                chosen_card = state.pack_cards.pop(action.target_idx)
            elif action.metadata and isinstance(action.metadata, dict):
                meta_set = action.metadata.get("set")
                meta_card = action.metadata.get("card")
                if meta_set in ("Default", "Enhanced"):
                    state.deck_cards_count += 1
                elif meta_card == "Cryptid":
                    state.deck_cards_count += 2

            if chosen_card is not None:
                if chosen_card.type == CardType.JOKER:
                    state.inventory.jokers.append(chosen_card)
                    on_joker_added(state, chosen_card)
                elif chosen_card.type == CardType.PLAYING_CARD:
                    state.add_card_to_deck(chosen_card, add_to_hand=False)
                    ctx = TriggerContext(
                        event=TriggerEvent.CARD_BOUGHT,
                        state=state,
                        action=action,
                        target_card=chosen_card,
                        extra={"already_added_to_deck": True},
                    )
                    dispatch_jokers(ctx)
                else:
                    _apply_consumable_state(state, chosen_card, action=action)
                    ctx = TriggerContext(
                        event=TriggerEvent.USE_CONSUMABLE,
                        state=state,
                        action=action,
                        target_card=chosen_card,
                    )
                    dispatch_jokers(ctx)

            if chosen_card is not None or state.active_pack is not None:
                state.pack_choices_remaining = max(0, state.pack_choices_remaining - 1)
                if state.pack_choices_remaining == 0 or not state.pack_cards:
                    _close_pack_or_advance_queue(state, action)

        case ActionType.SKIP_PACK:
            ctx = TriggerContext(
                event=TriggerEvent.SKIP_BOOSTER, state=state, action=action
            )
            dispatch_jokers(ctx)
            _close_pack_or_advance_queue(state, action)

        case ActionType.REDEEM_VOUCHER:
            voucher = None
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_vouchers
            ):
                voucher = state.shop_vouchers.pop(action.target_idx)
            elif action.metadata and isinstance(action.metadata, dict):
                v_str = action.metadata.get("card")
                for vn in VoucherName:
                    if vn.value == v_str:
                        voucher = Voucher(name=vn)
                        break
            if voucher is not None:
                cost = voucher.get_cost(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                )
                state.money -= cost
                state.apply_voucher(voucher.name, adjust_round_counters=True)

        case ActionType.SELL_CARD:
            sold_card = None
            if action.target_idx is not None:
                if action.target_idx < len(state.inventory.jokers):
                    sold_card = state.inventory.jokers.pop(action.target_idx)
                else:
                    c_idx = action.target_idx - len(state.inventory.jokers)
                    if 0 <= c_idx < len(state.inventory.consumables):
                        sold_card = state.inventory.consumables.pop(c_idx)
            if sold_card:
                state.money += sold_card.get_sell_value(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                    astronomer=_has_joker(state, JokerName.ASTRONOMER),
                )
                if sold_card.type == CardType.JOKER:
                    on_joker_removed(state, sold_card)
                    if state.current_blind == BlindName.VERDANT_LEAF:
                        if hasattr(state, "disable_active_boss"):
                            state.disable_active_boss()
                        else:
                            state.boss_disabled = True
                            for j in state.inventory.jokers:
                                if not (j.perishable and j.perish_tally <= 0):
                                    j.debuffed = False
                ctx = TriggerContext(
                    event=TriggerEvent.CARD_SOLD,
                    state=state,
                    action=action,
                    target_card=sold_card,
                )
                # Trigger the sold Joker's own selling_self effect (Invisible Joker, Luchador, Diet Cola)
                # before dispatching to remaining inventory Jokers (e.g. Campfire).
                if (
                    sold_card.type == CardType.JOKER
                    and not sold_card.debuffed
                    and sold_card.name in JOKER_EFFECTS
                ):
                    JOKER_EFFECTS[sold_card.name](sold_card, ctx)
                dispatch_jokers(ctx)

        case ActionType.CASH_OUT:
            # Note: Location.CASH_OUT is internal to post_processing.py; GameState
            # models the cash-out phase as Location.BLIND with round_won=True.
            if (
                getattr(state, "pending_blind_advance", False)
                and state.location == Location.SHOP
            ):
                _advance_blind_and_ante_if_pending(state)
                state.round_advanced = False

            if not getattr(state, "round_won", False):
                trigger_end_of_round_effects(state, action)
            _advance_round_if_needed(state)

            investment_count = (
                _consume_tag(state, SkipTag.INVESTMENT_TAG)
                if state.current_blind not in (BlindName.SMALL, BlindName.BIG)
                else 0
            )
            if action.metadata and action.metadata.get("round_dollars") is not None:
                state.money += action.metadata["round_dollars"]
            else:
                include_payout = (
                    action.metadata.get("include_round_payout", True)
                    if action.metadata is not None
                    else True
                )
                if include_payout:
                    state.money += calculate_base_round_payout(state)
                if investment_count:
                    state.money += 25 * investment_count
                ctx = TriggerContext(
                    event=TriggerEvent.END_OF_ROUND,
                    state=state,
                    action=action,
                    extra={"phase": "cash_out_dollars"},
                )
                dispatch_jokers(ctx)

            state.round_won = False
            state.coupon_shop_active = False
            state.location = Location.SHOP
            base_reroll = getattr(state, "base_reroll_cost", 5)
            if (
                VoucherName.REROLL_SURPLUS in state.vouchers_redeemed
                and base_reroll == 5
            ):
                base_reroll = max(0, base_reroll - 2)
            if VoucherName.REROLL_GLUT in state.vouchers_redeemed and base_reroll >= 3:
                base_reroll = max(0, base_reroll - 2)
            if _consume_tag(state, SkipTag.D6_TAG):
                base_reroll = 0
            state.reroll_cost = base_reroll
            _apply_shop_entry_tags(state)
            state.pending_blind_advance = True

        case ActionType.NEXT_ROUND:
            _leave_shop(state, action)

        case ActionType.SELECT_BLIND:
            if state.location == Location.SHOP:
                _leave_shop(state, action, force_advance=True)
            else:
                _advance_blind_and_ante_if_pending(state)
            state.round_won = False
            state.round_advanced = False
            state.coupon_shop_active = False
            state.location = Location.BLIND
            if state.boss_blind is None and (state.seed or getattr(state, "rng", None)):
                if hasattr(state, "roll_boss_blind"):
                    state.roll_boss_blind(is_reroll=False)
            state.current_chips = 0
            if hasattr(state, "compute_target_chips"):
                state.target_chips = state.compute_target_chips()
            state.hands_played_this_round = 0
            state.discards_used_this_round = 0

            base_hands = getattr(state, "base_hands", 4)
            if VoucherName.GRABBER in state.vouchers_redeemed:
                base_hands += 1
            if VoucherName.NACHO_TONG in state.vouchers_redeemed:
                base_hands += 1
            if VoucherName.HIEROGLYPH in state.vouchers_redeemed:
                base_hands -= 1
            for j in state.inventory.jokers:
                if j.name == JokerName.TROUBADOUR and not j.debuffed:
                    base_hands -= 1
            state.hands_left = max(1, base_hands)

            base_discards = getattr(state, "base_discards", 3)
            if VoucherName.WASTEFUL in state.vouchers_redeemed:
                base_discards += 1
            if VoucherName.RECYCLOMANCY in state.vouchers_redeemed:
                base_discards += 1
            if VoucherName.PETROGLYPH in state.vouchers_redeemed:
                base_discards -= 1
            for j in state.inventory.jokers:
                if not j.debuffed:
                    if j.name == JokerName.DRUNKARD:
                        base_discards += 1
                    elif j.name == JokerName.MERRY_ANDY:
                        base_discards += 3
            state.discards_left = max(0, base_discards)

            juggle_count = _consume_tag(state, SkipTag.JUGGLE_TAG)
            if juggle_count:
                bonus = 3 * juggle_count
                state.hand.max_size += bonus
                state.temp_hand_size_bonus += bonus

            state.discard_pile.clear()
            state.played_cards_this_round.clear()
            state.hands_played_this_round_by_type.clear()
            state.locked_hand_type = None
            state.boss_disabled = False
            state.fish_prepped = False
            state.crimson_heart_prepped = False
            state._manacle_active = False
            state._water_discards_sub = 0
            state._needle_hands_sub = 0

            if state.current_blind == BlindName.THE_WATER:
                state._water_discards_sub = state.discards_left
                state.discards_left = 0
            elif state.current_blind == BlindName.THE_NEEDLE:
                state._needle_hands_sub = max(0, state.hands_left - 1)
                state.hands_left = 1

            ctx = TriggerContext(
                event=TriggerEvent.SETTING_BLIND,
                state=state,
                action=action,
                extra={"initial_jokers": list(state.inventory.jokers)},
            )
            dispatch_jokers(ctx)

            if not state.boss_disabled and state.current_blind not in (
                BlindName.SMALL,
                BlindName.BIG,
            ):
                if state.current_blind == BlindName.THE_MANACLE:
                    state.hand.max_size = max(0, state.hand.max_size - 1)
                    state._manacle_active = True
                elif (
                    state.current_blind == BlindName.AMBER_ACORN
                    and state.inventory
                    and state.inventory.jokers
                ):
                    for j in state.inventory.jokers:
                        j.facing_back = True
                    if len(state.inventory.jokers) > 1 and getattr(state, "rng", None):
                        for _ in range(3):
                            state.rng.shuffle("aajk", state.inventory.jokers)
                elif state.current_blind == BlindName.CRIMSON_HEART and hasattr(
                    state, "apply_crimson_heart"
                ):
                    state.apply_crimson_heart()

            if hasattr(state, "refresh_debuffs"):
                state.refresh_debuffs()
            if (
                state.current_blind == BlindName.CERULEAN_BELL
                and not state.boss_disabled
                and state.hand.cards
                and hasattr(state, "apply_cerulean_bell")
            ):
                state.apply_cerulean_bell()

        case ActionType.SKIP_BLIND:
            if state.location == Location.SHOP:
                _leave_shop(state, action, force_advance=True)
            else:
                _advance_blind_and_ante_if_pending(state)
            state.blinds_skipped += 1
            tag: SkipTag | None = None
            orbital_hand: PokerHand | None = None
            if state.current_blind == BlindName.SMALL:
                tag = state.small_blind_tag
                orbital_hand = getattr(state, "small_blind_orbital_hand", None)
                state.current_blind = BlindName.BIG
            elif state.current_blind == BlindName.BIG:
                tag = state.big_blind_tag
                orbital_hand = getattr(state, "big_blind_orbital_hand", None)
                if state.boss_blind is not None:
                    state.current_blind = state.boss_blind

            _grant_skip_tag(
                state,
                tag,
                action=action,
                orbital_hand=orbital_hand,
                pre_pack_location=Location.SELECT,
            )
            if hasattr(state, "compute_target_chips"):
                state.target_chips = state.compute_target_chips()

        case ActionType.REROLL_BOSS:
            if state.location == Location.SHOP:
                _leave_shop(state, action, force_advance=True)
            else:
                _advance_blind_and_ante_if_pending(state)
            state.money -= 10
            if hasattr(state, "roll_boss_blind"):
                state.roll_boss_blind(is_reroll=True)
            elif state.seed or getattr(state, "rng", None):
                if getattr(state, "rng", None) is None and state.seed:
                    state.set_seed(state.seed)
                try:
                    from game.engine.rng_sim.events import get_next_boss_blind
                except ImportError:
                    from engine.rng_sim.events import get_next_boss_blind
                if not hasattr(state, "bosses_used") or state.bosses_used is None:
                    state.bosses_used = {}
                pred = get_next_boss_blind(
                    state.rng,
                    ante=state.ante,
                    bosses_used=state.bosses_used,
                    is_reroll=True,
                )
                state.boss_blind = pred.blind_name

        case ActionType.END_ROUND:
            # Note: Location.CASH_OUT is post_processing-internal only; after END_ROUND,
            # GameState remains in Location.BLIND with round_won=True until CASH_OUT.
            if (
                getattr(state, "pending_blind_advance", False)
                and state.location == Location.SHOP
            ):
                _advance_blind_and_ante_if_pending(state)
                state.round_advanced = False
            trigger_end_of_round_effects(state, action)
            _advance_round_if_needed(state)

    sync_dynamic_joker_states(state)
