from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from game.balatro_models import Action, Card, Joker, PlayingCard, PokerHand
        from game.gamestate import GameState
    except ImportError:
        from balatro_models import Action, Card, Joker, PlayingCard, PokerHand
        from gamestate import GameState


class TriggerEvent(Enum):
    # Blind & Round Events
    SETTING_BLIND = auto()
    END_OF_ROUND = auto()
    FIRST_HAND_DRAWN = auto()

    # Shop Events
    REROLL_SHOP = auto()
    CARD_BOUGHT = auto()
    CARD_SOLD = auto()
    OPEN_BOOSTER = auto()
    SKIP_BOOSTER = auto()
    USE_CONSUMABLE = auto()
    LEAVE_SHOP = auto()

    # Hand Play & Discard Events
    BEFORE_PLAY = auto()
    JOKER_MAIN = auto()
    DEBUFFED_HAND = auto()
    AFTER_PLAY = auto()
    DISCARD = auto()
    INDIVIDUAL_CARD_SCORED = auto()
    HELD_CARD_SCORED = auto()


@dataclass(kw_only=True)
class TriggerContext:
    event: TriggerEvent
    state: "GameState"
    action: "Action | None" = None
    target_card: "Any | None" = None
    extra: dict[str, Any] | None = None
    is_blueprint: bool = False
    hook: bool = False
    boss_triggered: bool = False
    scoring_cards: list[Any] = field(default_factory=list)
    full_hand: list[Any] = field(default_factory=list)
    poker_hand: Any = None

    def __post_init__(self):
        if not self.scoring_cards and self.state and hasattr(self.state, "hand") and self.state.hand:
            try:
                from game.engine.poker_hands import get_scoring_cards
            except ImportError:
                from engine.poker_hands import get_scoring_cards
            jokers = self.state.inventory.jokers if self.state.inventory else None
            self.scoring_cards = get_scoring_cards(self.state.hand.selected_cards, jokers)

        if not self.full_hand and self.state and hasattr(self.state, "hand") and self.state.hand:
            self.full_hand = list(self.state.hand.selected_cards)
