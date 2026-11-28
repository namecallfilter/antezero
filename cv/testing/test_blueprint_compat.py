"""
Tests for Blueprint & Brainstorm compatibility rules and copying mechanics.

Validates:
1. Exact 1:1 parity with Balatro game source (121 compatible, 29 incompatible jokers).
2. Copy target resolution for Blueprint (right neighbor) and Brainstorm (leftmost).
3. Recursive chaining (Blueprint -> Brainstorm -> Target) and cycle protection.
4. Functional behavior during dispatch_jokers (e.g. Burnt Joker double-level, Burglar double-hands).
5. Protection against illegal scaling duplication on accumulator jokers.

Usage:
    python testing/test_blueprint_compat.py
"""

from pathlib import Path
import re
import sys

# Ensure 'game' and repo root are in pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from game.balatro_models import (
        Action,
        ActionType,
        BlindName,
        CardType,
        Hand,
        HandCard,
        Inventory,
        Joker,
        JokerName,
        PlayingCard,
        PokerHand,
        Rank,
        Suit,
        JOKER_REGISTRY,
    )
    from game.engine.context import TriggerContext, TriggerEvent
    from game.engine.jokers import dispatch_jokers, resolve_copy_target
    from game.gamestate import GameState
except ImportError:
    from balatro_models import (
        Action,
        ActionType,
        BlindName,
        CardType,
        Hand,
        HandCard,
        Inventory,
        Joker,
        JokerName,
        PlayingCard,
        PokerHand,
        Rank,
        Suit,
        JOKER_REGISTRY,
    )
    from engine.context import TriggerContext, TriggerEvent
    from engine.jokers import dispatch_jokers, resolve_copy_target
    from gamestate import GameState


# The 29 Jokers verified to have blueprint_compat = false in balatro_source/game.lua
EXPECTED_INCOMPATIBLE = {
    JokerName.ASTRONOMER,
    JokerName.CHAOS_THE_CLOWN,
    JokerName.CHICOT,
    JokerName.CLOUD_9,
    JokerName.CREDIT_CARD,
    JokerName.DELAYED_GRATIFICATION,
    JokerName.DRUNKARD,
    JokerName.EGG,
    JokerName.FOUR_FINGERS,
    JokerName.GIFT_CARD,
    JokerName.GOLDEN_JOKER,
    JokerName.INVISIBLE_JOKER,
    JokerName.JUGGLER,
    JokerName.MERRY_ANDY,
    JokerName.MIDAS_MASK,
    JokerName.MR_BONES,
    JokerName.OOPS_ALL_6S,
    JokerName.PAREIDOLIA,
    JokerName.ROCKET,
    JokerName.SATELLITE,
    JokerName.SHORTCUT,
    JokerName.SHOWMAN,
    JokerName.SIXTH_SENSE,
    JokerName.SMEARED_JOKER,
    JokerName.SPLASH,
    JokerName.TO_THE_MOON,
    JokerName.TRADING_CARD,
    JokerName.TROUBADOUR,
    JokerName.TURTLE_BEAN,
}


def test_source_code_registry_parity():
    """Validates that JOKER_REGISTRY blueprint_compat matches balatro_source/game.lua exactly."""
    print("Validating blueprint_compat registry parity against Balatro source code...")

    source_path = REPO_ROOT / "balatro_source" / "game.lua"
    if source_path.exists():
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()

        lua_incompat_names = set()
        for match in re.finditer(r"^\s*(j_\w+)\s*=\s*\{(.+?)\},?\s*$", content, re.MULTILINE):
            body = match.group(2)
            if "set = 'Joker'" in body or 'set = "Joker"' in body:
                name_m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', body)
                bp_m = re.search(r'blueprint_compat\s*=\s*(true|false)', body)
                if name_m and bp_m and bp_m.group(1) == "false":
                    name = name_m.group(1)
                    # Normalize naming variations
                    if name == "Caino":
                        name = "Canio"
                    elif name == "Seance":
                        name = "Séance"
                    elif name == "Riff-raff":
                        name = "Riff-Raff"
                    lua_incompat_names.add(name)

        assert lua_incompat_names == set(j.value for j in EXPECTED_INCOMPATIBLE), (
            f"Discrepancy with source game.lua: {lua_incompat_names ^ set(j.value for j in EXPECTED_INCOMPATIBLE)}"
        )

    # Verify every Joker in JOKER_REGISTRY
    incompat_in_registry = set()
    compat_in_registry = set()
    for name, info in JOKER_REGISTRY.items():
        if info.blueprint_compat:
            compat_in_registry.add(name)
        else:
            incompat_in_registry.add(name)

    assert incompat_in_registry == EXPECTED_INCOMPATIBLE, (
        f"Registry mismatch: expected {len(EXPECTED_INCOMPATIBLE)} incompatible, got {len(incompat_in_registry)}"
    )
    assert len(compat_in_registry) == 121, f"Expected 121 compatible, got {len(compat_in_registry)}"
    assert len(JOKER_REGISTRY) == 150, f"Expected 150 total jokers, got {len(JOKER_REGISTRY)}"

    print("Source code registry parity: PERFECT (121 compatible, 29 incompatible).")


def test_copy_target_resolution():
    """Tests resolution rules: right neighbor, leftmost, chaining, boundary, and incompatibility."""
    print("Testing copy target resolution logic...")

    bp = Joker(name=JokerName.BLUEPRINT)
    bs = Joker(name=JokerName.BRAINSTORM)
    burnt = Joker(name=JokerName.BURNT_JOKER)
    four_fingers = Joker(name=JokerName.FOUR_FINGERS)

    # 1. Blueprint on far right has no target
    jokers = [burnt, bp]
    assert resolve_copy_target(bp, jokers) is None, "Blueprint on right edge should resolve to None"

    # 2. Blueprint next to incompatible joker resolves to None
    jokers = [bp, four_fingers]
    assert resolve_copy_target(bp, jokers) is None, "Blueprint copying incompatible joker should resolve to None"

    # 3. Blueprint next to compatible joker resolves to target
    jokers = [bp, burnt]
    assert resolve_copy_target(bp, jokers) == burnt, "Blueprint copying Burnt Joker should resolve to Burnt Joker"

    # 4. Brainstorm in slot 0 (copying itself) resolves to None
    jokers = [bs, burnt]
    assert resolve_copy_target(bs, jokers) is None, "Brainstorm at slot 0 cannot copy itself"

    # 5. Brainstorm copying compatible leftmost joker
    jokers = [burnt, bs]
    assert resolve_copy_target(bs, jokers) == burnt, "Brainstorm should resolve to leftmost joker (Burnt Joker)"

    # 6. Brainstorm copying incompatible leftmost joker
    jokers = [four_fingers, bs]
    assert resolve_copy_target(bs, jokers) is None, "Brainstorm copying incompatible leftmost joker should resolve to None"

    # 7. Chained copy: Blueprint -> Brainstorm -> Burnt Joker
    jokers = [burnt, bp, bs]
    # bp at slot 1 targets bs at slot 2; bs at slot 2 targets jokers[0] (burnt)
    assert resolve_copy_target(bp, jokers) == burnt, "Blueprint chaining through Brainstorm should resolve to Burnt Joker"

    # 8. Cycle protection: Blueprint at 0 targets Brainstorm at 1; Brainstorm at 1 targets jokers[0] (Blueprint)
    cycle_bp = Joker(name=JokerName.BLUEPRINT)
    cycle_bs = Joker(name=JokerName.BRAINSTORM)
    cycle_jokers = [cycle_bp, cycle_bs]
    assert resolve_copy_target(cycle_bp, cycle_jokers) is None, "Circular copy chain should safely resolve to None"

    print("Copy target resolution: ALL PASS.")


def test_functional_dispatch():
    """Tests functional behavior when Blueprint/Brainstorm copy action vs scaling jokers."""
    print("Testing functional dispatch with Blueprint & Brainstorm...")

    # Scenario 1: Burnt Joker + Blueprint (Compatible trigger duplication)
    # First discard of the round with a Pair should level up Pair TWICE (+2)
    burnt = Joker(name=JokerName.BURNT_JOKER)
    bp = Joker(name=JokerName.BLUEPRINT)
    pair_cards = [
        HandCard(card=PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS), selected=True),
        HandCard(card=PlayingCard(rank=Rank.ACE, suit=Suit.SPADES), selected=True),
    ]
    state = GameState(
        inventory=Inventory(jokers=[bp, burnt]),
        hand=Hand(cards=pair_cards),
        discards_used_this_round=1,
        hand_levels={PokerHand.PAIR: 1},
    )
    ctx = TriggerContext(event=TriggerEvent.DISCARD, state=state)
    dispatch_jokers(ctx)
    assert state.hand_levels[PokerHand.PAIR] == 3, f"Expected level 3 (1 base + 1 burnt + 1 blueprint), got {state.hand_levels[PokerHand.PAIR]}"

    # Scenario 2: Burnt Joker + Blueprint next to an INCOMPATIBLE joker
    # Blueprint is next to Four Fingers, so Blueprint does NOTHING; Burnt Joker levels up once (+1)
    ff = Joker(name=JokerName.FOUR_FINGERS)
    state_incompat = GameState(
        inventory=Inventory(jokers=[burnt, bp, ff]),
        hand=Hand(cards=pair_cards),
        discards_used_this_round=1,
        hand_levels={PokerHand.PAIR: 1},
    )
    ctx_incompat = TriggerContext(event=TriggerEvent.DISCARD, state=state_incompat)
    dispatch_jokers(ctx_incompat)
    assert state_incompat.hand_levels[PokerHand.PAIR] == 2, "Blueprint next to Four Fingers must not trigger"

    # Scenario 3: Burglar + Brainstorm (Compatible trigger duplication)
    # Brainstorm at index 1 copying Burglar at index 0 should give 3 + 3 = 6 hands
    burglar = Joker(name=JokerName.BURGLAR)
    bs = Joker(name=JokerName.BRAINSTORM)
    state_burglar = GameState(
        inventory=Inventory(jokers=[burglar, bs]),
        hands_left=4,
        discards_left=3,
    )
    ctx_burglar = TriggerContext(event=TriggerEvent.SETTING_BLIND, state=state_burglar)
    dispatch_jokers(ctx_burglar)
    assert state_burglar.hands_left == 10, f"Expected 10 hands (4 base + 3 burglar + 3 brainstorm), got {state_burglar.hands_left}"

    # Scenario 4: Flash Card + Blueprint (Scaling Jokers do NOT scale on Blueprint copy)
    # Rerolling shop should increase Flash Card's mult once (+2), NOT twice (+4)
    flash = Joker(name=JokerName.FLASH_CARD)
    flash.state["mult"] = 0
    state_flash = GameState(
        inventory=Inventory(jokers=[bp, flash]),
        money=10,
        reroll_cost=5,
    )
    ctx_reroll = TriggerContext(event=TriggerEvent.REROLL_SHOP, state=state_flash)
    dispatch_jokers(ctx_reroll)
    assert flash.state["mult"] == 2, f"Flash Card must only scale once (+2), got {flash.state['mult']}"

    print("Functional dispatch: ALL PASS.")


def test_blueprint_compat():
    """Runs all Blueprint & Brainstorm compatibility tests."""
    test_source_code_registry_parity()
    test_copy_target_resolution()
    test_functional_dispatch()
    print("\nALL BLUEPRINT & BRAINSTORM COMPATIBILITY TESTS PASSED PERFECTLY!")


if __name__ == "__main__":
    test_blueprint_compat()
