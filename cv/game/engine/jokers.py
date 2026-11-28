"""
Complete 150-Joker Engine implementation.
Connects all 150 Balatro Jokers to active state changes, accumulators,
game triggers, passive deck modifiers, and CV observation hooks.
"""

from collections import Counter
import copy
import random
from typing import Callable

try:
    from game.balatro_models import (
        BlindName,
        Card,
        CardType,
        Consumable,
        Edition,
        Enhancement,
        Joker,
        JokerName,
        JokerRarity,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        Suit,
    )
except ImportError:
    from balatro_models import (
        BlindName,
        Card,
        CardType,
        Consumable,
        Edition,
        Enhancement,
        Joker,
        JokerName,
        JokerRarity,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        Suit,
    )
from .context import TriggerContext, TriggerEvent
from .poker_hands import evaluate_poker_hand

try:
    from game.engine.rng_sim.events import (
        EventResolver,
        evaluate_lucky_hit,
        get_ancient_joker_suit,
        get_blocked_card_names,
        get_castle_suit,
        get_idol_card,
        get_mail_in_rebate_target,
        get_to_do_list_hand,
        poll_tarot_pool,
        predict_cartomancer_tarot,
        predict_certificate,
        predict_eight_ball_tarot,
        predict_marble_joker_card,
        predict_riff_raff_jokers,
        predict_seance_spectral,
        predict_sixth_sense_spectral,
        predict_superposition_tarot,
        predict_vagabond_tarot,
        state_has_showman,
    )
    from game.engine.rng_sim.state import DEFAULT_VISIBLE_POKER_HANDS, SimDeck
    from game.balatro_models import Spectral, SpectralName, Tarot
except ImportError:
    from engine.rng_sim.events import (
        EventResolver,
        evaluate_lucky_hit,
        get_ancient_joker_suit,
        get_blocked_card_names,
        get_castle_suit,
        get_idol_card,
        get_mail_in_rebate_target,
        get_to_do_list_hand,
        poll_tarot_pool,
        predict_cartomancer_tarot,
        predict_certificate,
        predict_eight_ball_tarot,
        predict_marble_joker_card,
        predict_riff_raff_jokers,
        predict_seance_spectral,
        predict_sixth_sense_spectral,
        predict_superposition_tarot,
        predict_vagabond_tarot,
        state_has_showman,
    )
    from engine.rng_sim.state import DEFAULT_VISIBLE_POKER_HANDS, SimDeck
    from balatro_models import Spectral, SpectralName, Tarot


def get_oops_multiplier(state) -> float:
    """Calculates active probability multiplier from all non-debuffed Oops! All 6s Jokers."""
    if not state or not state.inventory:
        return 1.0
    oops_count = sum(
        1
        for j in state.inventory.jokers
        if j.name == JokerName.OOPS_ALL_6S and not j.debuffed
    )
    return float(2 ** oops_count)


JOKER_EFFECTS: dict[JokerName, Callable[[Joker, TriggerContext], None]] = {}


def joker_hook(name: JokerName):
    """Registers a handler for a specific Joker."""
    def decorator(fn: Callable[[Joker, TriggerContext], None]):
        JOKER_EFFECTS[name] = fn
        return fn
    return decorator


# =========================================================================
# HELPER FUNCTIONS (Face / Suit / Retriggers / Passive Deck Modifiers)
# =========================================================================

def is_card_face(card: PlayingCard, jokers: list[Joker] | None = None) -> bool:
    """Checks if a PlayingCard counts as a face card, respecting debuffs, Stone, and Pareidolia."""
    if getattr(card, "debuffed", False):
        return False
    if jokers and any(j.name == JokerName.PAREIDOLIA and not j.debuffed for j in jokers):
        return True
    if card.enhancement == Enhancement.STONE or card.rank == Rank.NONE:
        return False
    return card.is_face


def card_matches_suit(
    card: PlayingCard,
    target_suit: Suit,
    jokers: list[Joker] | None = None,
    bypass_debuff: bool = False,
) -> bool:
    """Checks if a PlayingCard matches target_suit, respecting Stone, Wild, debuffs, and Smeared Joker."""
    if card.enhancement == Enhancement.STONE or card.rank == Rank.NONE:
        return False
    if getattr(card, "debuffed", False) and not bypass_debuff:
        return False
    if card.enhancement == Enhancement.WILD:
        return target_suit != Suit.NONE
    if jokers and any(j.name == JokerName.SMEARED_JOKER and not j.debuffed for j in jokers):
        if target_suit in (Suit.HEARTS, Suit.DIAMONDS) and card.suit in (Suit.HEARTS, Suit.DIAMONDS):
            return True
        if target_suit in (Suit.SPADES, Suit.CLUBS) and card.suit in (Suit.SPADES, Suit.CLUBS):
            return True
        return False
    return card.suit == target_suit


def get_card_repetitions(card: PlayingCard, scoring_idx: int, state) -> int:
    """Calculates total scoring triggers (1 + retriggers) for a scoring card at scoring_idx."""
    if getattr(card, "debuffed", False):
        return 0
    triggers = 1
    if card.seal == Seal.RED:
        triggers += 1

    jokers = state.inventory.jokers if state and state.inventory else []
    for j in jokers:
        if j.debuffed:
            continue
        eff = resolve_copy_target(j, jokers) if j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM) else j
        if eff is None or eff.debuffed:
            continue
        if eff.name == JokerName.HANGING_CHAD and scoring_idx == 0:
            triggers += 2
        elif eff.name == JokerName.HACK:
            if card.enhancement != Enhancement.STONE and card.rank in (
                Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE
            ):
                triggers += 1
        elif eff.name == JokerName.SOCK_AND_BUSKIN:
            if is_card_face(card, jokers):
                triggers += 1
        elif eff.name == JokerName.SELTZER:
            triggers += 1
        elif eff.name == JokerName.DUSK:
            if state.hands_left == 0:
                triggers += 1
    return triggers


def _get_visible_poker_hands(state) -> list[PokerHand]:
    """Returns currently visible poker hands (default 9 plus any unlocked secret hands)."""
    visible = list(DEFAULT_VISIBLE_POKER_HANDS)
    counts = getattr(state, "hand_play_counts", None) or {}
    for secret in (PokerHand.FIVE_OF_A_KIND, PokerHand.FLUSH_HOUSE, PokerHand.FLUSH_FIVE):
        if counts.get(secret, 0) > 0 and secret not in visible:
            visible.append(secret)
    return visible


def init_joker_default_state(state, joker: Joker) -> None:
    """Populates default keys in joker.state if not already initialized."""
    match joker.name:
        case JokerName.ANCIENT_JOKER:
            joker.state.setdefault("target_suit", state.ancient_suit)
        case JokerName.TO_DO_LIST:
            if "target_hand" not in joker.state:
                if getattr(state, "rng", None):
                    hand = get_to_do_list_hand(
                        state.rng,
                        current_hand=None,
                        visible_hands=_get_visible_poker_hands(state),
                        is_initial_spawn=True,
                    )
                else:
                    hand = state.to_do_poker_hand
                joker.state["target_hand"] = hand
            else:
                hand = joker.state["target_hand"]
                if not isinstance(hand, PokerHand):
                    hand = PokerHand(str(hand))
                    joker.state["target_hand"] = hand
            state.to_do_poker_hand = hand
        case JokerName.THE_IDOL:
            joker.state.setdefault("target_rank", state.idol_card[0])
            joker.state.setdefault("target_suit", state.idol_card[1])
        case JokerName.CASTLE:
            joker.state.setdefault("chips", 0)
            joker.state.setdefault("target_suit", state.castle_suit)
        case JokerName.MAIL_IN_REBATE:
            joker.state.setdefault("target_rank", state.mail_card_rank)
        case (
            JokerName.CEREMONIAL_DAGGER
            | JokerName.FLASH_CARD
            | JokerName.RED_CARD
            | JokerName.GREEN_JOKER
            | JokerName.RIDE_THE_BUS
            | JokerName.SPARE_TROUSERS
        ):
            joker.state.setdefault("mult", 0)
        case JokerName.SQUARE_JOKER | JokerName.RUNNER | JokerName.WEE_JOKER:
            joker.state.setdefault("chips", 0)
        case (
            JokerName.CAMPFIRE
            | JokerName.CONSTELLATION
            | JokerName.LUCKY_CAT
            | JokerName.HOLOGRAM
            | JokerName.MADNESS
            | JokerName.VAMPIRE
            | JokerName.OBELISK
            | JokerName.GLASS_JOKER
            | JokerName.CANIO
            | JokerName.HIT_THE_ROAD
        ):
            joker.state.setdefault("x_mult", 1.0)
        case JokerName.YORICK:
            joker.state.setdefault("x_mult", 1.0)
            joker.state.setdefault("discards", 23)
        case JokerName.RAMEN:
            joker.state.setdefault("x_mult", 2.0)
        case JokerName.LOYALTY_CARD:
            joker.state.setdefault("remaining", 5)
        case JokerName.POPCORN:
            joker.state.setdefault("mult", 20)
        case JokerName.ICE_CREAM:
            joker.state.setdefault("chips", 100)
        case JokerName.TURTLE_BEAN:
            joker.state.setdefault("h_size", 5)
        case JokerName.SELTZER:
            joker.state.setdefault("hands_left", 10)
        case JokerName.INVISIBLE_JOKER:
            joker.state.setdefault("rounds", 0)
        case JokerName.ROCKET:
            joker.state.setdefault("payout", 1)


def on_joker_added(state, joker: Joker) -> None:
    """Applies Card:add_to_deck passive effects and ensures a persistent Joker ID."""
    if state is not None and hasattr(state, "assign_joker_id"):
        state.assign_joker_id(joker)
    init_joker_default_state(state, joker)
    if joker.debuffed:
        return
    match joker.name:
        case JokerName.JUGGLER:
            state.hand.max_size += 1
        case JokerName.DRUNKARD:
            state.discards_left += 1
        case JokerName.MERRY_ANDY:
            state.discards_left += 3
            state.hand.max_size = max(0, state.hand.max_size - 1)
        case JokerName.TROUBADOUR:
            state.hand.max_size += 2
            state.hands_left = max(1, state.hands_left - 1)
        case JokerName.STUNTMAN:
            state.hand.max_size = max(0, state.hand.max_size - 2)
        case JokerName.TURTLE_BEAN:
            state.hand.max_size += joker.state.get("h_size", 5)
        case JokerName.CREDIT_CARD:
            state.bankrupt_at -= 20
        case JokerName.CHAOS_THE_CLOWN:
            state.free_rerolls += 1
        case JokerName.CHICOT:
            if state.current_blind not in (BlindName.SMALL, BlindName.BIG):
                if hasattr(state, "disable_active_boss"):
                    state.disable_active_boss()
                else:
                    state.boss_disabled = True


def on_joker_removed(state, joker: Joker) -> None:
    """Reverses Card:remove_from_deck passive effects when a Joker is sold or destroyed."""
    if joker.debuffed:
        return
    match joker.name:
        case JokerName.JUGGLER:
            state.hand.max_size = max(0, state.hand.max_size - 1)
        case JokerName.DRUNKARD:
            state.discards_left = max(0, state.discards_left - 1)
        case JokerName.MERRY_ANDY:
            state.discards_left = max(0, state.discards_left - 3)
            state.hand.max_size += 1
        case JokerName.TROUBADOUR:
            state.hand.max_size = max(0, state.hand.max_size - 2)
            state.hands_left += 1
        case JokerName.STUNTMAN:
            state.hand.max_size += 2
        case JokerName.TURTLE_BEAN:
            state.hand.max_size = max(0, state.hand.max_size - joker.state.get("h_size", 5))
        case JokerName.CREDIT_CARD:
            state.bankrupt_at += 20
        case JokerName.CHAOS_THE_CLOWN:
            state.free_rerolls = max(0, state.free_rerolls - 1)


def sync_dynamic_joker_states(state) -> None:
    """Synchronizes dynamic state values on all Jokers at each action boundary."""
    if not state or not state.inventory:
        return
    jokers = state.inventory.jokers
    for j in jokers:
        init_joker_default_state(state, j)
        match j.name:
            case JokerName.SWASHBUCKLER:
                j.state["mult"] = sum(other.sell_value for other in jokers if other is not j)
            case JokerName.ABSTRACT_JOKER:
                j.state["mult"] = 3 * len(jokers)
            case JokerName.JOKER_STENCIL:
                non_neg = sum(1 for other in jokers if other.edition != Edition.NEGATIVE)
                empty_slots = max(0, state.inventory.joker_slots - non_neg)
                stencils = sum(
                    1 for other in jokers if other.name == JokerName.JOKER_STENCIL and not other.debuffed
                )
                j.state["x_mult"] = float(max(1, empty_slots + stencils))
            case JokerName.FORTUNE_TELLER:
                j.state["mult"] = state.tarots_used
            case JokerName.THROWBACK:
                j.state["x_mult"] = round(1.0 + 0.25 * state.blinds_skipped, 2)
            case JokerName.STONE_JOKER:
                j.state["chips"] = 25 * sum(
                    1 for c in state.full_deck if c.enhancement == Enhancement.STONE
                )
            case JokerName.STEEL_JOKER:
                j.state["x_mult"] = round(
                    1.0 + 0.2 * sum(1 for c in state.full_deck if c.enhancement == Enhancement.STEEL),
                    2,
                )
            case JokerName.DRIVER:
                tally = sum(1 for c in state.full_deck if c.enhancement != Enhancement.NONE)
                j.state["driver_tally"] = tally
                j.state["x_mult"] = 3.0 if tally >= 16 else 1.0
            case JokerName.BLUE_JOKER:
                j.state["chips"] = 2 * len(state.remaining_deck)
            case JokerName.EROSION:
                j.state["mult"] = max(0, 4 * (state.starting_deck_size - len(state.full_deck)))
            case JokerName.CLOUD_9:
                j.state["nine_tally"] = sum(
                    1
                    for c in state.full_deck
                    if c.rank == Rank.NINE and c.enhancement != Enhancement.STONE
                )
            case JokerName.BULL:
                j.state["chips"] = max(0, 2 * state.money)
            case JokerName.BOOTSTRAPS:
                j.state["mult"] = 2 * max(0, state.money // 5)
            case JokerName.BANNER:
                j.state["chips"] = 30 * state.discards_left
            case JokerName.MYSTIC_SUMMIT:
                j.state["mult"] = 15 if state.discards_left == 0 else 0
            case JokerName.BASEBALL_CARD:
                j.state["uncommon_count"] = sum(
                    1 for other in jokers if other.info.rarity == JokerRarity.UNCOMMON
                )


# =========================================================================
# 1. TARGET-SHIFTING & ROTATING JOKERS
# =========================================================================

def _is_end_of_round_phase(ctx: TriggerContext) -> bool:
    """True when END_OF_ROUND is dispatched at round evaluation (or standalone without phase)."""
    if ctx.event != TriggerEvent.END_OF_ROUND:
        return False
    return ctx.extra is None or ctx.extra.get("phase") == "end_of_round"


def _is_cash_out_dollars_phase(ctx: TriggerContext) -> bool:
    """True when END_OF_ROUND is dispatched for Cash Out dollar payouts (or standalone without phase)."""
    if ctx.event != TriggerEvent.END_OF_ROUND:
        return False
    return ctx.extra is None or ctx.extra.get("phase") == "cash_out_dollars"


@joker_hook(JokerName.ANCIENT_JOKER)
def handle_ancient_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        if ctx.extra and ctx.extra.get("round_targets_rolled"):
            suit = ctx.state.ancient_suit
        elif ctx.state.rng:
            suit = get_ancient_joker_suit(
                ctx.state.rng,
                ante=ctx.state.ante,
                current_suit=ctx.state.ancient_suit,
            )
        else:
            suit = ctx.state.ancient_suit
        ctx.state.ancient_suit = suit
        joker.state["target_suit"] = suit


@joker_hook(JokerName.TO_DO_LIST)
def handle_to_do_list(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        target = joker.state.get("target_hand", ctx.state.to_do_poker_hand)
        played_hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        if played_hand == target:
            ctx.state.money += 4
    elif _is_end_of_round_phase(ctx) and not ctx.is_blueprint:
        if ctx.state.rng:
            next_hand = get_to_do_list_hand(
                ctx.state.rng,
                current_hand=joker.state.get("target_hand", ctx.state.to_do_poker_hand),
                visible_hands=_get_visible_poker_hands(ctx.state),
            )
        else:
            next_hand = joker.state.get("target_hand", ctx.state.to_do_poker_hand)
        ctx.state.to_do_poker_hand = next_hand
        joker.state["target_hand"] = next_hand


@joker_hook(JokerName.THE_IDOL)
def handle_the_idol(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        if ctx.extra and ctx.extra.get("round_targets_rolled"):
            rank, suit = ctx.state.idol_card
        elif ctx.state.rng:
            sdeck = SimDeck(ctx.state.full_deck)
            target = get_idol_card(ctx.state.rng, sdeck, ante=ctx.state.ante)
            rank, suit = target.rank, target.suit
        else:
            rank, suit = ctx.state.idol_card
        ctx.state.idol_card = (rank, suit)
        joker.state["target_rank"] = rank
        joker.state["target_suit"] = suit


@joker_hook(JokerName.CASTLE)
def handle_castle(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        target_suit = joker.state.get("target_suit", ctx.state.castle_suit)
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        matching = sum(1 for c in discarded if card_matches_suit(c, target_suit, jokers))
        if matching > 0:
            joker.state["chips"] = joker.state.get("chips", 0) + (matching * 3)
    elif _is_end_of_round_phase(ctx):
        if ctx.extra and ctx.extra.get("round_targets_rolled"):
            suit = ctx.state.castle_suit
        elif ctx.state.rng:
            sdeck = SimDeck(ctx.state.full_deck)
            suit = get_castle_suit(ctx.state.rng, sdeck, ante=ctx.state.ante)
        else:
            suit = ctx.state.castle_suit
        ctx.state.castle_suit = suit
        joker.state["target_suit"] = suit


@joker_hook(JokerName.MAIL_IN_REBATE)
def handle_mail_in_rebate(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        target_rank = joker.state.get("target_rank", ctx.state.mail_card_rank)
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        matches = sum(
            1
            for c in discarded
            if not getattr(c, "debuffed", False)
            and c.enhancement != Enhancement.STONE
            and c.rank == target_rank
        )
        if matches > 0:
            ctx.state.money += matches * 5
    elif _is_end_of_round_phase(ctx) and not ctx.is_blueprint:
        if ctx.extra and ctx.extra.get("round_targets_rolled"):
            rank = ctx.state.mail_card_rank
        elif ctx.state.rng:
            sdeck = SimDeck(ctx.state.full_deck)
            rank = get_mail_in_rebate_target(ctx.state.rng, sdeck, ante=ctx.state.ante)
        else:
            rank = ctx.state.mail_card_rank
        ctx.state.mail_card_rank = rank
        joker.state["target_rank"] = rank


# =========================================================================
# 2. ACCUMULATOR & STATE-SCALING JOKERS
# =========================================================================

@joker_hook(JokerName.CEREMONIAL_DAGGER)
def handle_ceremonial_dagger(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        initial_jokers = ctx.extra.get("initial_jokers") if ctx.extra else None
        jokers_to_check = initial_jokers if initial_jokers is not None else ctx.state.inventory.jokers
        if joker in jokers_to_check:
            idx = jokers_to_check.index(joker)
            if idx + 1 < len(jokers_to_check):
                sliced = jokers_to_check[idx + 1]
                if not sliced.eternal:
                    if sliced in ctx.state.inventory.jokers:
                        ctx.state.inventory.jokers.remove(sliced)
                    on_joker_removed(ctx.state, sliced)
                    sell_val = sliced.get_sell_value(
                        discount_percent=ctx.state.discount_percent,
                        inflation=ctx.state.inflation,
                    )
                    added_mult = sell_val * 2
                    joker.state["mult"] = joker.state.get("mult", 0) + added_mult


@joker_hook(JokerName.GIFT_CARD)
def handle_gift_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        for c in ctx.state.inventory.jokers:
            c.extra_value += 1
        for c in ctx.state.inventory.consumables:
            c.extra_value += 1


@joker_hook(JokerName.EGG)
def handle_egg(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        joker.extra_value += 3


@joker_hook(JokerName.FLASH_CARD)
def handle_flash_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.REROLL_SHOP:
        joker.state["mult"] = joker.state.get("mult", 0) + 2


@joker_hook(JokerName.RED_CARD)
def handle_red_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SKIP_BOOSTER:
        joker.state["mult"] = joker.state.get("mult", 0) + 3


@joker_hook(JokerName.GREEN_JOKER)
def handle_green_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        joker.state["mult"] = joker.state.get("mult", 0) + 1
    elif ctx.event == TriggerEvent.DISCARD:
        joker.state["mult"] = max(0, joker.state.get("mult", 0) - 1)


@joker_hook(JokerName.RIDE_THE_BUS)
def handle_ride_the_bus(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(jokers) if ctx.state and ctx.state.hand else []
        )
        if any(is_card_face(c, jokers) for c in scoring):
            joker.state["mult"] = 0
        else:
            joker.state["mult"] = joker.state.get("mult", 0) + 1


@joker_hook(JokerName.SQUARE_JOKER)
def handle_square_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if len(played) == 4:
            joker.state["chips"] = joker.state.get("chips", 0) + 4


@joker_hook(JokerName.RUNNER)
def handle_runner(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else []),
            ctx.state.inventory.jokers if ctx.state else None,
        )
        if hand in (PokerHand.STRAIGHT, PokerHand.STRAIGHT_FLUSH):
            joker.state["chips"] = joker.state.get("chips", 0) + 15


@joker_hook(JokerName.WEE_JOKER)
def handle_wee_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if (
            ctx.target_card
            and not getattr(ctx.target_card, "debuffed", False)
            and ctx.target_card.enhancement != Enhancement.STONE
            and ctx.target_card.rank == Rank.TWO
        ):
            joker.state["chips"] = joker.state.get("chips", 0) + 8
    elif ctx.event == TriggerEvent.BEFORE_PLAY:
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
            if ctx.state and ctx.state.hand
            else []
        )
        twos = sum(
            1
            for c in scoring
            if not getattr(c, "debuffed", False)
            and c.enhancement != Enhancement.STONE
            and c.rank == Rank.TWO
        )
        if twos > 0 and ctx.extra and ctx.extra.get("batch_wee"):
            joker.state["chips"] = joker.state.get("chips", 0) + (twos * 8)


@joker_hook(JokerName.SPARE_TROUSERS)
def handle_spare_trousers(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (
            ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else []
        )
        valid_cards = [
            c
            for c in played
            if getattr(c, "enhancement", None) != Enhancement.STONE
            and getattr(c, "rank", Rank.NONE) != Rank.NONE
        ]
        counts = sorted(Counter(c.rank for c in valid_cards).values(), reverse=True)
        contains_two_pair = len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2
        hand = ctx.poker_hand or evaluate_poker_hand(
            played,
            ctx.state.inventory.jokers if ctx.state else None,
        )
        if contains_two_pair or hand in (
            PokerHand.TWO_PAIR,
            PokerHand.FULL_HOUSE,
            PokerHand.FLUSH_HOUSE,
        ):
            joker.state["mult"] = joker.state.get("mult", 0) + 2


@joker_hook(JokerName.CAMPFIRE)
def handle_campfire(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.CARD_SOLD:
        joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)
    elif _is_end_of_round_phase(ctx):
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            joker.state["x_mult"] = 1.0


@joker_hook(JokerName.CONSTELLATION)
def handle_constellation(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.USE_CONSUMABLE:
        if ctx.target_card and ctx.target_card.type == CardType.PLANET:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.1, 2)
            ctx.state.unique_planets_used.add(str(getattr(ctx.target_card, "name", "")))


@joker_hook(JokerName.LUCKY_CAT)
def handle_lucky_cat(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and not getattr(ctx.target_card, "debuffed", False):
            if ctx.extra and "lucky_triggered" in ctx.extra:
                if ctx.extra.get("lucky_triggered"):
                    joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)
            elif ctx.target_card.enhancement == Enhancement.LUCKY and ctx.state.rng:
                oops = get_oops_multiplier(ctx.state)
                hit = evaluate_lucky_hit(ctx.state.rng, probability_modifier=oops)
                if hit.mult:
                    ctx.target_card.bonus_mult = getattr(ctx.target_card, "bonus_mult", 0) + 20
                if hit.money:
                    ctx.state.money += 20
                if hit.any_hit:
                    joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)


@joker_hook(JokerName.HIKER)
def handle_hiker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and not getattr(ctx.target_card, "debuffed", False):
            ctx.target_card.bonus_chips += 5
            if ctx.state:
                ctx.state.sync_card_to_deck(ctx.target_card)
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_hiker"):
            scoring = ctx.scoring_cards or (
                ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
                if ctx.state and ctx.state.hand
                else []
            )
            for card in scoring:
                if not getattr(card, "debuffed", False):
                    card.bonus_chips += 5
                    if ctx.state:
                        ctx.state.sync_card_to_deck(card)


@joker_hook(JokerName.HOLOGRAM)
def handle_hologram(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.CARD_BOUGHT:
        if ctx.extra and ctx.extra.get("already_added_to_deck"):
            return
        if ctx.target_card and ctx.target_card.type == CardType.PLAYING_CARD:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)


@joker_hook(JokerName.MADNESS)
def handle_madness(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.current_blind in (BlindName.SMALL, BlindName.BIG):
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.5, 2)
            destructible = [
                j for j in ctx.state.inventory.jokers if j is not joker and not j.eternal
            ]
            if destructible:
                if ctx.state.rng:
                    victim, _ = ctx.state.rng.choice("madness", destructible)
                    if victim in ctx.state.inventory.jokers:
                        on_joker_removed(ctx.state, victim)
                        ctx.state.inventory.jokers.remove(victim)
                else:
                    ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.VAMPIRE)
def handle_vampire(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
            if ctx.state and ctx.state.hand
            else []
        )
        for card in scoring:
            if card.enhancement != Enhancement.NONE and not getattr(card, "debuffed", False):
                card.enhancement = Enhancement.NONE
                if ctx.state:
                    ctx.state.sync_card_to_deck(card)
                joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.1, 2)


@joker_hook(JokerName.OBELISK)
def handle_obelisk(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        counts = ctx.state.hand_play_counts
        play_more_than = counts.get(hand, 0) if counts else 0
        # In Balatro (card.lua:3545), Obelisk resets iff play_more_than > 0 and NO other hand has played >= play_more_than
        if play_more_than > 0 and not any(k != hand and v >= play_more_than for k, v in counts.items()):
            joker.state["x_mult"] = 1.0
        else:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.2, 2)


@joker_hook(JokerName.GLASS_JOKER)
def handle_glass_joker(joker: Joker, ctx: TriggerContext):
    # State update is handled deterministically in GameState.remove_card_from_deck when a Glass Card shatters.
    pass


@joker_hook(JokerName.CANIO)
def handle_canio(joker: Joker, ctx: TriggerContext):
    # State update is handled deterministically in GameState.remove_card_from_deck when a face card is destroyed.
    pass


@joker_hook(JokerName.YORICK)
def handle_yorick(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if not discarded:
            return
        discards = joker.state.get("discards", 23) - len(discarded)
        while discards <= 0:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 1.0, 2)
            discards += 23
        joker.state["discards"] = discards


@joker_hook(JokerName.RAMEN)
def handle_ramen(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        count = len(discarded)
        if count > 0:
            current_xmult = joker.state.get("x_mult", 2.0)
            new_xmult = round(current_xmult - (count * 0.01), 2)
            if new_xmult <= 1.0:
                if joker in ctx.state.inventory.jokers:
                    on_joker_removed(ctx.state, joker)
                    ctx.state.inventory.jokers.remove(joker)
            else:
                joker.state["x_mult"] = new_xmult


@joker_hook(JokerName.LOYALTY_CARD)
def handle_loyalty_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        rem = joker.state.get("remaining", 5)
        if isinstance(rem, str):
            import re
            m = re.search(r"\d+", rem)
            rem = int(m.group(0)) if m else 5
        rem = int(rem) - 1
        if rem < 0:
            rem = 5
        joker.state["remaining"] = rem


# =========================================================================
# 3. COUNTDOWN & EXPIRY JOKERS
# =========================================================================

@joker_hook(JokerName.POPCORN)
def handle_popcorn(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        current_mult = joker.state.get("mult", 20)
        new_mult = current_mult - 4
        if new_mult <= 0:
            if joker in ctx.state.inventory.jokers:
                on_joker_removed(ctx.state, joker)
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["mult"] = new_mult


@joker_hook(JokerName.ICE_CREAM)
def handle_ice_cream(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.AFTER_PLAY:
        current_chips = joker.state.get("chips", 100)
        new_chips = max(0, current_chips - 5)
        if new_chips <= 0:
            if joker in ctx.state.inventory.jokers:
                on_joker_removed(ctx.state, joker)
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["chips"] = new_chips


@joker_hook(JokerName.TURTLE_BEAN)
def handle_turtle_bean(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        current_size = joker.state.get("h_size", 5)
        new_size = current_size - 1
        ctx.state.hand.max_size = max(0, ctx.state.hand.max_size - 1)
        if new_size <= 0:
            joker.state["h_size"] = 0
            if joker in ctx.state.inventory.jokers:
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["h_size"] = new_size


@joker_hook(JokerName.SELTZER)
def handle_seltzer(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.AFTER_PLAY:
        remaining = joker.state.get("hands_left", 10) - 1
        if remaining <= 0:
            if joker in ctx.state.inventory.jokers:
                on_joker_removed(ctx.state, joker)
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["hands_left"] = remaining


@joker_hook(JokerName.INVISIBLE_JOKER)
def handle_invisible_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        joker.state["rounds"] = joker.state.get("rounds", 0) + 1
    elif ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card is joker and joker.state.get("rounds", 0) >= 2:
            others = [j for j in ctx.state.inventory.jokers if j is not joker]
            if not others or not ctx.state.inventory.can_add_joker:
                return
            if ctx.state.rng:
                chosen, _ = ctx.state.rng.choice("invisible", others)
                clone = copy.deepcopy(chosen)
                clone.id = 0
                # Invisible Joker strips Negative edition when duplicating (card.lua:2384)
                if clone.edition == Edition.NEGATIVE:
                    clone.edition = Edition.NONE
                if clone.name == JokerName.INVISIBLE_JOKER:
                    clone.state["rounds"] = 0
                if hasattr(ctx.state, "assign_joker_id"):
                    ctx.state.assign_joker_id(clone, force_new=True)
                ctx.state.inventory.jokers.append(clone)
                on_joker_added(ctx.state, clone)
            else:
                ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.GROS_MICHEL)
def handle_gros_michel(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        if ctx.state.rng:
            oops = get_oops_multiplier(ctx.state)
            if ctx.state.rng.poll_odds("gros_michel", 1, 6, oops):
                ctx.state.gros_michel_extinct = True
                if joker in ctx.state.inventory.jokers:
                    on_joker_removed(ctx.state, joker)
                    ctx.state.inventory.jokers.remove(joker)
        else:
            ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.CAVENDISH)
def handle_cavendish(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        if ctx.state.rng:
            oops = get_oops_multiplier(ctx.state)
            if ctx.state.rng.poll_odds("cavendish", 1, 1000, oops):
                if joker in ctx.state.inventory.jokers:
                    on_joker_removed(ctx.state, joker)
                    ctx.state.inventory.jokers.remove(joker)
        else:
            ctx.state.pending_cv_checks.add("jokers")


# =========================================================================
# 4. ROUND & BLIND START JOKERS (Hands, Discards, Spawns)
# =========================================================================

@joker_hook(JokerName.BURGLAR)
def handle_burglar(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        ctx.state.hands_left += 3
        ctx.state.discards_left = 0


@joker_hook(JokerName.DRUNKARD)
def handle_drunkard(joker: Joker, ctx: TriggerContext):
    # Passive modifier applied to base discards on SETTING_BLIND / add_to_deck
    pass


@joker_hook(JokerName.MERRY_ANDY)
def handle_merry_andy(joker: Joker, ctx: TriggerContext):
    # Passive modifier applied to base discards on SETTING_BLIND / add_to_deck
    pass


@joker_hook(JokerName.TROUBADOUR)
def handle_troubadour(joker: Joker, ctx: TriggerContext):
    # Passive modifier applied to base hands on SETTING_BLIND / add_to_deck
    pass


@joker_hook(JokerName.MARBLE_JOKER)
def handle_marble_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.rng:
            res = predict_marble_joker_card(ctx.state.rng)
            stone_card = PlayingCard(
                rank=res.rank,
                suit=res.suit,
                enhancement=Enhancement.STONE,
            )
        else:
            stone_card = PlayingCard(
                rank=Rank.NONE,
                suit=Suit.NONE,
                enhancement=Enhancement.STONE,
            )
        ctx.state.add_card_to_deck(stone_card, add_to_hand=False)


@joker_hook(JokerName.CARTOMANCER)
def handle_cartomancer(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.inventory and ctx.state.inventory.can_add_consumable:
            if ctx.state.rng:
                pred = predict_cartomancer_tarot(
                    ctx.state.rng,
                    ante=ctx.state.ante,
                    active_tarots=get_blocked_card_names(ctx.state, CardType.TAROT),
                    has_showman=state_has_showman(ctx.state),
                )
                ctx.state.inventory.consumables.append(pred.to_tarot_card())
            else:
                ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.RIFF_RAFF)
def handle_riff_raff(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.inventory and ctx.state.inventory.can_add_joker:
            if ctx.action and ctx.action.metadata and ctx.action.metadata.get("riff_raff_spawned"):
                queue = ctx.action.metadata["riff_raff_spawned"]
                space = ctx.state.inventory.max_jokers - len(ctx.state.inventory.jokers)
                spawn_count = min(2, space, len(queue))
                for _ in range(spawn_count):
                    sj = queue.pop(0)
                    ctx.state.inventory.jokers.append(sj)
                    on_joker_added(ctx.state, sj)
            elif ctx.state.rng:
                space = ctx.state.inventory.max_jokers - len(ctx.state.inventory.jokers)
                spawn_count = min(2, space)
                if spawn_count > 0:
                    spawned = predict_riff_raff_jokers(
                        ctx.state.rng,
                        ante=ctx.state.ante,
                        count=spawn_count,
                        owned_jokers=get_blocked_card_names(ctx.state, CardType.JOKER),
                        has_showman=state_has_showman(ctx.state),
                    )
                    for sj in spawned:
                        ctx.state.inventory.jokers.append(sj)
                        on_joker_added(ctx.state, sj)
            else:
                ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.CERTIFICATE)
def handle_certificate(joker: Joker, ctx: TriggerContext):
    if ctx.event in (TriggerEvent.FIRST_HAND_DRAWN, TriggerEvent.SETTING_BLIND):
        if ctx.state.rng:
            res = predict_certificate(ctx.state.rng, ante=ctx.state.ante)
            cert_card = PlayingCard(
                rank=res.rank,
                suit=res.suit,
                seal=res.seal,
            )
        else:
            cert_card = PlayingCard(rank=Rank.ACE, suit=Suit.SPADES, seal=Seal.RED)
        ctx.state.add_card_to_deck(cert_card, add_to_hand=True)


@joker_hook(JokerName.CHICOT)
def handle_chicot(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            if hasattr(ctx.state, "disable_active_boss"):
                ctx.state.disable_active_boss()
            else:
                ctx.state.boss_disabled = True


@joker_hook(JokerName.LUCHADOR)
def handle_luchador(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card is joker and ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            if hasattr(ctx.state, "disable_active_boss"):
                ctx.state.disable_active_boss()
            else:
                ctx.state.boss_disabled = True
                for j in ctx.state.inventory.jokers:
                    if not (j.perishable and j.perish_tally <= 0):
                        j.debuffed = False
                for hc in ctx.state.hand.cards:
                    hc.card.debuffed = False


@joker_hook(JokerName.DIET_COLA)
def handle_diet_cola(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card is joker or ctx.is_blueprint:
            ctx.state.active_tags.append(SkipTag.DOUBLE_TAG)


# =========================================================================
# 5. DISCARD-TRIGGERED JOKERS
# =========================================================================

@joker_hook(JokerName.TRADING_CARD)
def handle_trading_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint or ctx.hook:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.discards_used_this_round == 1 and len(discarded) == 1:
            target = discarded[0]
            ctx.state.money += 3
            ctx.state.remove_card_from_deck(target)


@joker_hook(JokerName.FACELESS_JOKER)
def handle_faceless_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        faces = sum(1 for c in discarded if is_card_face(c, jokers))
        if faces >= 3:
            ctx.state.money += 5


@joker_hook(JokerName.BURNT_JOKER)
def handle_burnt_joker(joker: Joker, ctx: TriggerContext):
    if ctx.hook:
        return
    if ctx.event == TriggerEvent.DISCARD:
        if ctx.state.discards_used_this_round == 1:
            discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
            hand_type = evaluate_poker_hand(discarded, ctx.state.inventory.jokers)
            ctx.state.hand_levels[hand_type] = ctx.state.hand_levels.get(hand_type, 1) + 1


# =========================================================================
# 6. ECONOMY & PAYOUT JOKERS
# =========================================================================

@joker_hook(JokerName.GOLDEN_JOKER)
def handle_golden_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_cash_out_dollars_phase(ctx):
        ctx.state.money += 4


@joker_hook(JokerName.DELAYED_GRATIFICATION)
def handle_delayed_gratification(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_cash_out_dollars_phase(ctx):
        if ctx.state.discards_used_this_round == 0:
            ctx.state.money += ctx.state.discards_left * 2


@joker_hook(JokerName.ROCKET)
def handle_rocket(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        payout = joker.state.get("payout", 1)
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            payout += 2
            joker.state["payout"] = payout
    if _is_cash_out_dollars_phase(ctx):
        ctx.state.money += joker.state.get("payout", 1)


@joker_hook(JokerName.CLOUD_9)
def handle_cloud_9(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_cash_out_dollars_phase(ctx):
        deck_to_check = ctx.state.full_deck if ctx.state.full_deck else [c.card for c in ctx.state.hand.cards]
        nines = sum(
            1 for c in deck_to_check if c.rank == Rank.NINE and c.enhancement != Enhancement.STONE
        )
        ctx.state.money += nines


@joker_hook(JokerName.SATELLITE)
def handle_satellite(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_cash_out_dollars_phase(ctx):
        ctx.state.money += len(ctx.state.unique_planets_used)


@joker_hook(JokerName.TO_THE_MOON)
def handle_to_the_moon(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.CHAOS_THE_CLOWN)
def handle_chaos_the_clown(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if _is_end_of_round_phase(ctx):
        ctx.state.free_rerolls += 1


# =========================================================================
# 7. SPECIAL PLAY / SPAWN / CARD MODIFIER JOKERS
# =========================================================================

@joker_hook(JokerName.PERKEO)
def handle_perkeo(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.LEAVE_SHOP:
        if ctx.state.inventory and ctx.state.inventory.consumables:
            if ctx.state.rng:
                chosen, _ = ctx.state.rng.choice(
                    "perkeo", ctx.state.inventory.consumables
                )
                clone = copy.deepcopy(chosen)
                clone.edition = Edition.NEGATIVE
                ctx.state.inventory.consumables.append(clone)
            else:
                ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.DNA)
def handle_dna(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.hands_played_this_round <= 1 and len(played) == 1:
            copied_card = copy.deepcopy(played[0])
            copied_card.id = 0
            ctx.state.add_card_to_deck(copied_card, add_to_hand=True)


@joker_hook(JokerName.SIXTH_SENSE)
def handle_sixth_sense(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.AFTER_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.hands_played_this_round <= 1 and len(played) == 1:
            c = played[0]
            if c.rank == Rank.SIX and c.enhancement != Enhancement.STONE:
                ctx.state.remove_card_from_deck(c)
                if ctx.state.inventory and ctx.state.inventory.can_add_consumable:
                    if ctx.state.rng:
                        sp = predict_sixth_sense_spectral(
                            ctx.state.rng,
                            ante=ctx.state.ante,
                            active_spectrals=get_blocked_card_names(ctx.state, CardType.SPECTRAL),
                            has_showman=state_has_showman(ctx.state),
                        )
                        ctx.state.inventory.consumables.append(sp)
                    else:
                        ctx.state.inventory.consumables.append(Spectral(name=SpectralName.ANKH))


@joker_hook(JokerName.SEANCE)
def handle_seance(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        if hand == PokerHand.STRAIGHT_FLUSH and ctx.state.inventory.can_add_consumable:
            if ctx.state.rng:
                sp = predict_seance_spectral(
                    ctx.state.rng,
                    ante=ctx.state.ante,
                    active_spectrals=get_blocked_card_names(ctx.state, CardType.SPECTRAL),
                    has_showman=state_has_showman(ctx.state),
                )
                ctx.state.inventory.consumables.append(sp)
            else:
                ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.SUPERPOSITION)
def handle_superposition(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        scoring = ctx.scoring_cards or ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
        has_ace = any(c.rank == Rank.ACE and c.enhancement != Enhancement.STONE for c in scoring)
        if hand in (PokerHand.STRAIGHT, PokerHand.STRAIGHT_FLUSH) and has_ace:
            if ctx.state.inventory.can_add_consumable:
                if ctx.state.rng:
                    pred = predict_superposition_tarot(
                        ctx.state.rng,
                        ante=ctx.state.ante,
                        active_tarots=get_blocked_card_names(ctx.state, CardType.TAROT),
                        has_showman=state_has_showman(ctx.state),
                    )
                    ctx.state.inventory.consumables.append(pred.to_tarot_card())
                else:
                    ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.VAGABOND)
def handle_vagabond(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        if ctx.state.money <= 4 and ctx.state.inventory.can_add_consumable:
            if ctx.state.rng:
                pred = predict_vagabond_tarot(
                    ctx.state.rng,
                    ante=ctx.state.ante,
                    active_tarots=get_blocked_card_names(ctx.state, CardType.TAROT),
                    has_showman=state_has_showman(ctx.state),
                )
                ctx.state.inventory.consumables.append(pred.to_tarot_card())
            else:
                ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.HALLUCINATION)
def handle_hallucination(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.OPEN_BOOSTER:
        if ctx.state.inventory and ctx.state.inventory.can_add_consumable:
            if ctx.state.rng and ctx.state.resolver:
                oops = get_oops_multiplier(ctx.state)
                if ctx.state.resolver.hallucination(oops_multiplier=oops, ante=ctx.state.ante):
                    pred = poll_tarot_pool(
                        ctx.state.rng,
                        ante=ctx.state.ante,
                        key_append=f"halu{ctx.state.ante}",
                        active_tarots=get_blocked_card_names(ctx.state, CardType.TAROT),
                        has_showman=state_has_showman(ctx.state),
                    )
                    ctx.state.inventory.consumables.append(pred.to_tarot_card())
            else:
                ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.EIGHT_BALL)
def handle_eight_ball(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if (
            ctx.target_card
            and not getattr(ctx.target_card, "debuffed", False)
            and ctx.target_card.enhancement != Enhancement.STONE
            and ctx.target_card.rank == Rank.EIGHT
            and ctx.state.inventory.can_add_consumable
        ):
            if ctx.state.rng:
                oops = get_oops_multiplier(ctx.state)
                if ctx.state.rng.poll_odds("8ball", 1, 4, oops):
                    pred = predict_eight_ball_tarot(
                        ctx.state.rng,
                        ante=ctx.state.ante,
                        active_tarots=get_blocked_card_names(ctx.state, CardType.TAROT),
                        has_showman=state_has_showman(ctx.state),
                    )
                    ctx.state.inventory.consumables.append(pred.to_tarot_card())
            else:
                ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.SPACE_JOKER)
def handle_space_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        if ctx.state.rng:
            oops = get_oops_multiplier(ctx.state)
            if ctx.state.rng.poll_odds("space", 1, 4, oops):
                hand = ctx.poker_hand or evaluate_poker_hand(
                    ctx.full_hand or ctx.state.hand.selected_cards,
                    ctx.state.inventory.jokers,
                )
                ctx.state.hand_levels[hand] = ctx.state.hand_levels.get(hand, 1) + 1
        else:
            ctx.state.pending_cv_checks.add("hand_levels")


@joker_hook(JokerName.BUSINESS_CARD)
def handle_business_card(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        if ctx.target_card and is_card_face(ctx.target_card, jokers):
            if ctx.state.rng:
                oops = get_oops_multiplier(ctx.state)
                if ctx.state.rng.poll_odds("business", 1, 2, oops):
                    ctx.state.money += 2
            else:
                ctx.state.pending_cv_checks.add("money")


@joker_hook(JokerName.RESERVED_PARKING)
def handle_reserved_parking(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.HELD_CARD_SCORED:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        c = ctx.target_card
        if c and is_card_face(c, jokers) and not getattr(c, "debuffed", False):
            if ctx.state.rng:
                oops = get_oops_multiplier(ctx.state)
                if ctx.state.rng.poll_odds("parking", 1, 2, oops):
                    ctx.state.money += 1
            else:
                ctx.state.pending_cv_checks.add("money")


@joker_hook(JokerName.MATADOR)
def handle_matador(joker: Joker, ctx: TriggerContext):
    if ctx.event in (TriggerEvent.DEBUFFED_HAND, TriggerEvent.JOKER_MAIN) and ctx.boss_triggered:
        ctx.state.money += 8


@joker_hook(JokerName.MIDAS_MASK)
def handle_midas_mask(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(jokers) if ctx.state and ctx.state.hand else []
        )
        for c in scoring:
            if is_card_face(c, jokers):
                c.enhancement = Enhancement.GOLD
                if ctx.state:
                    ctx.state.sync_card_to_deck(c)


@joker_hook(JokerName.ROUGH_GEM)
def handle_rough_gem(joker: Joker, ctx: TriggerContext):
    jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and card_matches_suit(ctx.target_card, Suit.DIAMONDS, jokers):
            ctx.state.money += 1
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_payout"):
            scoring = ctx.scoring_cards or (
                ctx.state.hand.get_scoring_cards(jokers) if ctx.state and ctx.state.hand else []
            )
            diamonds = sum(1 for c in scoring if card_matches_suit(c, Suit.DIAMONDS, jokers))
            ctx.state.money += diamonds


@joker_hook(JokerName.GOLDEN_TICKET)
def handle_golden_ticket(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if (
            ctx.target_card
            and not getattr(ctx.target_card, "debuffed", False)
            and ctx.target_card.enhancement == Enhancement.GOLD
        ):
            ctx.state.money += 4
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_payout"):
            scoring = ctx.scoring_cards or (
                ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
                if ctx.state and ctx.state.hand
                else []
            )
            golds = sum(
                1
                for c in scoring
                if not getattr(c, "debuffed", False) and c.enhancement == Enhancement.GOLD
            )
            ctx.state.money += golds * 4


@joker_hook(JokerName.HIT_THE_ROAD)
def handle_hit_the_road(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        jacks = sum(
            1
            for c in discarded
            if not getattr(c, "debuffed", False)
            and c.enhancement != Enhancement.STONE
            and c.rank == Rank.JACK
        )
        if jacks > 0:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + (jacks * 0.5), 2)
    elif _is_end_of_round_phase(ctx):
        joker.state["x_mult"] = 1.0


# =========================================================================
# 8. PURE PASSIVE SCORING & STATIC JOKERS
# =========================================================================
# Chip and mult score calculation for static jokers is intentionally omitted
# because round scores are captured directly via OCR.
STATIC_SCORING_JOKERS = [
    JokerName.JOKER, JokerName.GREEDY_JOKER, JokerName.LUSTY_JOKER, JokerName.WRATHFUL_JOKER,
    JokerName.GLUTTONOUS_JOKER, JokerName.JOLLY_JOKER, JokerName.ZANY_JOKER, JokerName.MAD_JOKER,
    JokerName.CRAZY_JOKER, JokerName.DROLL_JOKER, JokerName.SLY_JOKER, JokerName.WILY_JOKER,
    JokerName.CLEVER_JOKER, JokerName.DEVIOUS_JOKER, JokerName.CRAFTY_JOKER, JokerName.HALF_JOKER,
    JokerName.JOKER_STENCIL, JokerName.FOUR_FINGERS, JokerName.MIME, JokerName.CREDIT_CARD,
    JokerName.BANNER, JokerName.MYSTIC_SUMMIT, JokerName.MISPRINT, JokerName.DUSK,
    JokerName.RAISED_FIST, JokerName.CHAOS_THE_CLOWN, JokerName.FIBONACCI, JokerName.STEEL_JOKER,
    JokerName.SCARY_FACE, JokerName.ABSTRACT_JOKER, JokerName.HACK, JokerName.PAREIDOLIA,
    JokerName.EVEN_STEVEN, JokerName.ODD_TODD, JokerName.SCHOLAR, JokerName.BUSINESS_CARD,
    JokerName.SUPERNOVA, JokerName.SPACE_JOKER, JokerName.BLACKBOARD, JokerName.SPLASH,
    JokerName.BLUE_JOKER, JokerName.CARD_SHARP, JokerName.SHORTCUT, JokerName.BARON,
    JokerName.PHOTOGRAPH, JokerName.EROSION, JokerName.RESERVED_PARKING, JokerName.FORTUNE_TELLER,
    JokerName.JUGGLER, JokerName.DRUNKARD, JokerName.STONE_JOKER, JokerName.BASEBALL_CARD,
    JokerName.BULL, JokerName.WALKIE_TALKIE, JokerName.SMILEY_FACE,
    JokerName.MR_BONES, JokerName.ACROBAT, JokerName.SOCK_AND_BUSKIN, JokerName.SWASHBUCKLER,
    JokerName.TROUBADOUR, JokerName.SMEARED_JOKER, JokerName.THROWBACK, JokerName.HANGING_CHAD,
    JokerName.BLOODSTONE, JokerName.ARROWHEAD, JokerName.ONYX_AGATE, JokerName.SHOWMAN,
    JokerName.FLOWER_POT, JokerName.MERRY_ANDY, JokerName.OOPS_ALL_6S, JokerName.SEEING_DOUBLE,
    JokerName.MATADOR, JokerName.THE_DUO, JokerName.THE_TRIO, JokerName.THE_FAMILY,
    JokerName.THE_ORDER, JokerName.THE_TRIBE, JokerName.STUNTMAN, JokerName.SHOOT_THE_MOON,
    JokerName.DRIVER, JokerName.ASTRONOMER, JokerName.BOOTSTRAPS, JokerName.TRIBOULET,
    JokerName.BLUEPRINT, JokerName.BRAINSTORM,
]


def _noop_hook(joker: Joker, ctx: TriggerContext):
    pass


for static_joker in STATIC_SCORING_JOKERS:
    if static_joker not in JOKER_EFFECTS:
        JOKER_EFFECTS[static_joker] = _noop_hook


# =========================================================================
# CENTRAL DISPATCHER & BLUEPRINT / BRAINSTORM RESOLUTION
# =========================================================================

def resolve_copy_target(joker: Joker, jokers: list[Joker], visited: set[int] | None = None) -> Joker | None:
    """Resolves the ultimate concrete Joker that Blueprint or Brainstorm copies.

    Follows copy chains (e.g. Blueprint copying Brainstorm) with cycle protection.
    Returns None if:
    - Target slot does not exist (Blueprint on right edge, or Brainstorm on slot 0 targeting itself).
    - Chain forms a cycle.
    - Target joker has blueprint_compat=False.
    """
    if visited is None:
        visited = set()

    j_id = id(joker)
    if j_id in visited:
        return None
    visited.add(j_id)

    if joker.name == JokerName.BLUEPRINT:
        if joker not in jokers:
            return None
        idx = jokers.index(joker)
        if idx + 1 >= len(jokers):
            return None
        candidate = jokers[idx + 1]
    elif joker.name == JokerName.BRAINSTORM:
        if not jokers or jokers[0] == joker:
            return None
        candidate = jokers[0]
    else:
        candidate = joker

    # If the candidate is also a copy Joker, recursively resolve down the chain
    if candidate.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
        return resolve_copy_target(candidate, jokers, visited)

    # Check whether the target Joker is compatible with Blueprint / Brainstorm
    if not candidate.blueprint_compat:
        return None

    return candidate


def dispatch_jokers(ctx: TriggerContext):
    """
    Dispatches the event to all Jokers currently in inventory,
    fully respecting Blueprint and Brainstorm copy targets, compatibility rules, and debuff status.
    """
    jokers = list(ctx.state.inventory.jokers)
    for j in jokers:
        init_joker_default_state(ctx.state, j)
        if j.debuffed:
            continue
        if j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
            target = resolve_copy_target(j, jokers)
            if target and not target.debuffed and target.blueprint_compat and target.name in JOKER_EFFECTS:
                copy_ctx = TriggerContext(
                    event=ctx.event,
                    state=ctx.state,
                    action=ctx.action,
                    target_card=ctx.target_card,
                    scoring_cards=ctx.scoring_cards,
                    full_hand=ctx.full_hand,
                    poker_hand=ctx.poker_hand,
                    extra=ctx.extra,
                    is_blueprint=True,
                    hook=ctx.hook,
                    boss_triggered=ctx.boss_triggered,
                )
                JOKER_EFFECTS[target.name](target, copy_ctx)
        elif j.name in JOKER_EFFECTS:
            JOKER_EFFECTS[j.name](j, ctx)
    sync_dynamic_joker_states(ctx.state)
