"""
Coarse screen-state tracker for template gating in `main.py`.

Maintains a conservative state machine over Balatro's visible UI screens so frame
scanning only evaluates templates that can legitimately appear and be clicked in
the current screen state. Starting mid-run is out of scope: tracking starts at
run initialization / Ante 1 (`PRE_GAME`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from game.balatro_models import ActionType, BlindName, Location
from game.gamestate import GameState

T = TypeVar("T")


class CoarseScreenState(StrEnum):
    """Coarse visible screen states (including conservative hybrid states)."""

    PRE_GAME = "pre_game"
    SELECT = "select"
    SELECT_OR_PACK = "select_or_pack"
    BLIND = "blind"
    CASH_OUT = "cash_out"
    SHOP = "shop"
    PACK = "pack"
    PACK_OR_SHOP = "pack_or_shop"
    PACK_OR_SELECT = "pack_or_select"
    SHOP_OR_SELECT_OR_PACK = "shop_or_select_or_pack"


# Exact mapping from templates.json keys to game ActionType (None for non-gameplay UI).
TEMPLATE_TO_ACTION: dict[str, ActionType | None] = {
    "Play Game": None,
    "Open Options Menu": None,
    "Open Run Info Menu": None,
    "Back (Run Info)": None,
    "Full Deck": None,
    "Select Blind": ActionType.SELECT_BLIND,
    "Skip Blind": ActionType.SKIP_BLIND,
    "Reroll Boss": ActionType.REROLL_BOSS,
    "Play Hand": ActionType.PLAY_HAND,
    "Discard": ActionType.DISCARD,
    "Sort by Rank": ActionType.SORT_RANK,
    "Sort by Suit": ActionType.SORT_SUIT,
    "Cash Out": ActionType.CASH_OUT,
    "Buy Card": ActionType.BUY_CARD,
    "Buy and Use": ActionType.BUY_AND_USE,
    "Open Pack": ActionType.OPEN_PACK,
    "Redeem Voucher": ActionType.REDEEM_VOUCHER,
    "Reroll Shop": ActionType.REROLL_SHOP,
    "Next Round": ActionType.NEXT_ROUND,
    "Use In Pack": ActionType.USE_IN_PACK,
    "Select Card": ActionType.USE_IN_PACK,
    "Skip Pack": ActionType.SKIP_PACK,
    "Sell Card": ActionType.SELL_CARD,
    "Use Consumable": ActionType.USE_CONSUMABLE,
}

# Verified against balatro_source/card.lua (can_sell_card, can_use_consumeable),
# functions/button_callbacks.lua, and game.lua (create_UIBox_buttons only in SELECTING_HAND).
_UI_ALWAYS = frozenset({"Open Options Menu", "Open Run Info Menu", "Full Deck"})
_HELD_ITEM_ACTIONS = frozenset({"Sell Card", "Use Consumable"})

_PRE_GAME_BUTTONS = frozenset(
    {
        "Play Game",
        "Select Blind",
        "Skip Blind",
        "Reroll Boss",
        "Open Options Menu",
        "Open Run Info Menu",
        "Full Deck",
    }
)
_SELECT_BUTTONS = (
    frozenset({"Select Blind", "Skip Blind", "Reroll Boss"})
    | _HELD_ITEM_ACTIONS
    | _UI_ALWAYS
)
_BLIND_BUTTONS = (
    # "Cash Out" is included because beating a blind transitions directly to ROUND_EVAL
    # with no intervening button click until temporal END_ROUND detection notifies us.
    frozenset({"Play Hand", "Discard", "Sort by Rank", "Sort by Suit", "Cash Out"})
    | _HELD_ITEM_ACTIONS
    | _UI_ALWAYS
)
_CASH_OUT_BUTTONS = frozenset({"Cash Out"}) | _HELD_ITEM_ACTIONS | _UI_ALWAYS
_SHOP_BUTTONS = (
    frozenset(
        {
            "Buy Card",
            "Buy and Use",
            "Open Pack",
            "Redeem Voucher",
            "Reroll Shop",
            "Next Round",
        }
    )
    | _HELD_ITEM_ACTIONS
    | _UI_ALWAYS
)
_PACK_BUTTONS = (
    frozenset({"Use In Pack", "Select Card", "Skip Pack"})
    | _HELD_ITEM_ACTIONS
    | _UI_ALWAYS
)

ALLOWED_TEMPLATES_BY_STATE: dict[CoarseScreenState, frozenset[str]] = {
    CoarseScreenState.PRE_GAME: _PRE_GAME_BUTTONS,
    CoarseScreenState.SELECT: _SELECT_BUTTONS,
    CoarseScreenState.SELECT_OR_PACK: _SELECT_BUTTONS | _PACK_BUTTONS,
    CoarseScreenState.BLIND: _BLIND_BUTTONS,
    CoarseScreenState.CASH_OUT: _CASH_OUT_BUTTONS,
    CoarseScreenState.SHOP: _SHOP_BUTTONS,
    CoarseScreenState.PACK: _PACK_BUTTONS,
    CoarseScreenState.PACK_OR_SHOP: _PACK_BUTTONS | _SHOP_BUTTONS,
    CoarseScreenState.PACK_OR_SELECT: _PACK_BUTTONS | _SELECT_BUTTONS,
    CoarseScreenState.SHOP_OR_SELECT_OR_PACK: (
        _SHOP_BUTTONS | _PACK_BUTTONS | frozenset({"Select Blind", "Skip Blind"})
    ),
}


def is_action_clickable_in_gamestate(state: GameState, template_name: str) -> bool:
    """Checks `GameState` semantics to confirm when a template action is legal.

    Used to validate coarse state permissions against `game.gamestate` state fields
    such as `location`, `round_won`, `hands_left`, `discards_left`, and `pre_pack_location`.
    """
    action_type = TEMPLATE_TO_ACTION.get(template_name)
    if action_type is None:
        return True

    if action_type in (ActionType.SELL_CARD, ActionType.USE_CONSUMABLE):
        return state.location in (
            Location.SELECT,
            Location.BLIND,
            Location.CASH_OUT,
            Location.SHOP,
            Location.PACK,
        )

    match state.location:
        case Location.SELECT:
            return action_type in (
                ActionType.SELECT_BLIND,
                ActionType.SKIP_BLIND,
                ActionType.REROLL_BOSS,
            )
        case Location.BLIND:
            if state.round_won:
                return action_type == ActionType.CASH_OUT
            if action_type == ActionType.PLAY_HAND:
                return (state.hands_left or 0) > 0
            if action_type == ActionType.DISCARD:
                return (state.discards_left or 0) > 0
            if action_type in (ActionType.SORT_RANK, ActionType.SORT_SUIT):
                return True
            if action_type == ActionType.CASH_OUT:
                # Implicit win before END_ROUND detection can transition to Cash Out.
                return True
            return False
        case Location.SHOP:
            return action_type in (
                ActionType.BUY_CARD,
                ActionType.BUY_AND_USE,
                ActionType.OPEN_PACK,
                ActionType.REDEEM_VOUCHER,
                ActionType.REROLL_SHOP,
                ActionType.NEXT_ROUND,
            )
        case Location.PACK:
            return action_type in (
                ActionType.USE_IN_PACK,
                ActionType.SKIP_PACK,
            )
        case Location.CASH_OUT:
            return action_type == ActionType.CASH_OUT
    return False


@dataclass
class ScreenStateTracker:
    """Tracks conservative coarse Balatro UI state across detected template clicks.

    Note: Starting mid-run is out of scope. Tracking begins in `PRE_GAME`, where only
    pre-game / Ante 1 Blind Select templates are enabled until the run starts.
    """

    state: CoarseScreenState = CoarseScreenState.PRE_GAME
    pre_pack_location: Location | None = None
    allow_play_game: bool = True
    last_cash_out_frame: int | None = None
    shop_next_round_cooldown_frames: int = 60
    last_held_item_action_frame: int | None = None
    cash_out_after_held_item_cooldown_frames: int = 30

    def allowed_template_names(self, frame_idx: int | None = None) -> frozenset[str]:
        """Returns the exact set of template names valid to scan in the current state."""
        allowed = ALLOWED_TEMPLATES_BY_STATE[self.state]
        if self.allow_play_game:
            allowed = allowed | {"Play Game"}
        if (
            self.state
            in (
                CoarseScreenState.SHOP,
                CoarseScreenState.PACK_OR_SHOP,
                CoarseScreenState.SHOP_OR_SELECT_OR_PACK,
            )
            and frame_idx is not None
            and self.last_cash_out_frame is not None
            and 0 <= (frame_idx - self.last_cash_out_frame) < self.shop_next_round_cooldown_frames
        ):
            allowed = allowed - {"Next Round"}
        if (
            self.state in (CoarseScreenState.BLIND, CoarseScreenState.CASH_OUT)
            and frame_idx is not None
            and self.last_held_item_action_frame is not None
            and 0 <= (frame_idx - self.last_held_item_action_frame) < self.cash_out_after_held_item_cooldown_frames
        ):
            allowed = allowed - {"Cash Out"}
        return allowed

    def filter_templates(
        self, templates: Sequence[T], frame_idx: int | None = None
    ) -> list[T]:
        """Filters loaded `Template` objects to those valid in the current screen state."""
        allowed = self.allowed_template_names(frame_idx=frame_idx)
        return [t for t in templates if getattr(t, "name", None) in allowed]

    def _resolve_pack_return_location(self) -> Location:
        if self.pre_pack_location in (Location.SELECT, Location.SHOP):
            return self.pre_pack_location
        if self.state in (
            CoarseScreenState.SELECT,
            CoarseScreenState.SELECT_OR_PACK,
            CoarseScreenState.PACK_OR_SELECT,
        ):
            return Location.SELECT
        return Location.SHOP

    def on_end_round(self) -> CoarseScreenState:
        """Called when a round-end boundary is externally known (e.g., temporal model)."""
        if self.state == CoarseScreenState.BLIND:
            self.state = CoarseScreenState.CASH_OUT
            self.pre_pack_location = None
        return self.state

    def on_pack_possibly_opened(self) -> CoarseScreenState:
        """Widens SHOP -> PACK_OR_SHOP when a missed Open Pack may have entered a pack."""
        if self.state == CoarseScreenState.SHOP:
            self.pre_pack_location = Location.SHOP
            self.state = CoarseScreenState.PACK_OR_SHOP
        return self.state

    def on_shop_idle_fallback(self) -> CoarseScreenState:
        """Widens SHOP -> SHOP_OR_SELECT_OR_PACK on prolonged inactivity so Next Round/Skip Pack can recover."""
        if self.state == CoarseScreenState.SHOP:
            self.pre_pack_location = Location.SHOP
            self.state = CoarseScreenState.SHOP_OR_SELECT_OR_PACK
        return self.state

    def on_blind_view_deck_observed(self) -> CoarseScreenState:
        """Moves non-blind coarse states to BLIND when two-tab Blind View Deck UI is active."""
        if self.state not in (CoarseScreenState.BLIND, CoarseScreenState.CASH_OUT):
            self.state = CoarseScreenState.BLIND
            self.pre_pack_location = None
            self.last_cash_out_frame = None
            self.last_held_item_action_frame = None
        return self.state

    def on_template_clicked(
        self, template_name: str, frame_idx: int | None = None
    ) -> CoarseScreenState:
        """Advances coarse screen state based on a clicked template name."""
        if template_name == "Open Options Menu":
            # Options can open the New Run -> PLAY setup modal from any screen.
            self.allow_play_game = True
            return self.state

        if template_name in ("Open Run Info Menu", "Back (Run Info)", "Full Deck"):
            return self.state

        if template_name == "Play Game":
            # Main Menu PLAY opens Run Setup (which also has PLAY -> Blind Select);
            # keep Play Game allowed until a gameplay button confirms Blind Select started.
            self.state = CoarseScreenState.SELECT
            self.pre_pack_location = None
            self.allow_play_game = True
            self.last_cash_out_frame = None
            self.last_held_item_action_frame = None
            return self.state

        # Any concrete in-run gameplay click clears temporary Options/Main-Menu Play Game flag.
        self.allow_play_game = False

        if template_name == "Select Blind":
            self.state = CoarseScreenState.BLIND
            self.pre_pack_location = None
            self.last_cash_out_frame = None
            self.last_held_item_action_frame = None
        elif template_name == "Skip Blind":
            # Skip Tag may either jump to next blind on SELECT or immediately open a pack.
            self.state = CoarseScreenState.SELECT_OR_PACK
            self.pre_pack_location = Location.SELECT
        elif template_name == "Reroll Boss":
            self.state = CoarseScreenState.SELECT
            self.pre_pack_location = None
        elif template_name in ("Play Hand", "Discard", "Sort by Rank", "Sort by Suit"):
            self.state = CoarseScreenState.BLIND
            self.pre_pack_location = None
            if template_name in ("Play Hand", "Discard"):
                self.last_held_item_action_frame = None
        elif template_name == "Cash Out":
            self.state = CoarseScreenState.SHOP
            self.pre_pack_location = None
            self.last_cash_out_frame = frame_idx
            self.last_held_item_action_frame = None
        elif template_name in ("Buy Card", "Buy and Use", "Redeem Voucher", "Reroll Shop"):
            self.state = CoarseScreenState.SHOP
            self.pre_pack_location = None
        elif template_name == "Open Pack":
            self.pre_pack_location = Location.SHOP
            self.state = CoarseScreenState.PACK_OR_SHOP
        elif template_name == "Next Round":
            self.state = CoarseScreenState.SELECT
            self.pre_pack_location = None
            self.last_cash_out_frame = None
            self.last_held_item_action_frame = None
        elif template_name in ("Use In Pack", "Select Card"):
            return_loc = self._resolve_pack_return_location()
            self.pre_pack_location = return_loc
            # Mega packs (choose=2) or Double-Tagged Skip Packs keep the player in PACK,
            # while single-choice packs return immediately to return_loc.
            if return_loc == Location.SHOP and frame_idx is not None:
                self.last_cash_out_frame = frame_idx
            self.state = (
                CoarseScreenState.PACK_OR_SELECT
                if return_loc == Location.SELECT
                else CoarseScreenState.PACK_OR_SHOP
            )
        elif template_name == "Skip Pack":
            return_loc = self._resolve_pack_return_location()
            if return_loc == Location.SELECT:
                # Double Tag can queue a second pack behind a Skip Tag pack.
                self.state = CoarseScreenState.SELECT_OR_PACK
                self.pre_pack_location = Location.SELECT
            else:
                self.state = CoarseScreenState.SHOP
                self.pre_pack_location = None
                if frame_idx is not None:
                    self.last_cash_out_frame = frame_idx
        elif template_name in _HELD_ITEM_ACTIONS:
            # Selling or using a held consumable never leaves the current screen; block
            # immediate "Cash Out" false positives triggered by dissolving card/UI particles.
            if frame_idx is not None:
                self.last_held_item_action_frame = frame_idx
        return self.state


def validate_templates_covered(template_names: Iterable[str]) -> list[str]:
    """Returns any template names missing from `TEMPLATE_TO_ACTION`."""
    return [name for name in template_names if name not in TEMPLATE_TO_ACTION]
