from __future__ import annotations

from game.balatro_models import (
    Deck as DeckType,
)
from game.balatro_models import (
    Enhancement,
    JokerName,
    PlayingCard,
    PokerHand,
    Rank,
    Suit,
    TarotName,
    VoucherName,
)
from game.engine.rng_sim import BalatroRun, Card, Deck
from game.gamestate import GameState as EngineGameState


run = BalatroRun(seed="HELLO")

for ante in range(1, 9):
    castle_ante = ""
    for round in range(3):
        castle_ante += f"{run.get_castle_suit(ante=ante)} -> "
    print(f"Ante {ante}: {castle_ante}")
