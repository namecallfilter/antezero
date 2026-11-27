# `game.engine.rng_sim` Sample & Reference Guide

Run all examples from the repository root:

```bash
python -m game.engine.rng_sim.sample.examples_all_features
```

---

## 1. How to Handle Mid-Ante Deck Changes (Without Throwing Away `n-1` Rounds)

For deck-dependent round selectors (`Castle`, `The Idol`, `Mail-in Rebate`), Balatro calls:

```lua
pseudorandom_element(valid_cards, pseudoseed('cas' .. G.GAME.round_resets.ante))
```

Notice what happens inside `pseudoseed('cas1')`:
1. The stream state `G.GAME.pseudorandom['cas1']` advances via the 64-bit float LCG (`(1.72431234 * st + 2.134453429141) % 1`).
2. **Crucially**, that LCG state transition depends **only** on how many times `'cas1'` has been called in Ante 1 (`0 -> 1 -> 2 -> 3`)—it does **not** depend on the size or contents of `valid_cards`! The deck (`valid_cards`) is only used inside the ephemeral `Tw223(pseed)` seeded for that specific round's draw.

Therefore, you have **two clean ways** to query mid-ante deck changes without ever re-evaluating or throwing away previous rounds:

### Pattern A: Stateful Round-by-Round Progression (Recommended for Live Runs)
Keep a single `BalatroRun` instance and update `run.state.deck` (or call `run.update_deck(gs.full_deck)` / `run.sync_from_gamestate(gs)`) as the run progresses:

```python
from game.balatro_models import Suit
from game.engine.rng_sim import BalatroRun

run = BalatroRun(seed="HELLO")

# Round 1 (Small Blind): advances 'cas1' call 0 -> 1 using Round 1's deck
r1_suit = run.get_castle_suit(ante=1)
r1_idol = run.get_idol_card(ante=1)

# Mid-ante deck mutation (e.g., in Round 1 or the shop after Round 1):
for card in run.state.deck.cards:
    if card.suit == Suit.SPADES:
        card.suit = Suit.HEARTS

# Round 2 (Big Blind): 'cas1' is ALREADY at call 1, so this advances call 1 -> 2
# directly against the mutated deck! Round 1 is never re-run.
r2_suit = run.get_castle_suit(ante=1)
r2_idol = run.get_idol_card(ante=1)
```

### Pattern B: Direct Jump via `round_num=1 | 2 | 3` (Stateless / Snapshot Queries)
If you don't want to track earlier rounds at all and just have a snapshot of the deck at Round `n` (where `round_num=1` is Small Blind, `2` is Big Blind, `3` is Boss Blind), pass `round_num=n`:

```python
# Jumps the 'cas1' / 'idol1' / 'mail1' LCG directly to Round 2 (call #2)
# and evaluates against the current deck only once!
r2_suit = run.get_castle_suit(ante=1, round_num=2)
r2_idol = run.get_idol_card(ante=1, round_num=2)
r2_mail = run.get_mail_in_rebate_target(ante=1, round_num=2)
```

---

## 2. Integration with `game.balatro_models` & `game.gamestate.GameState`

`game.engine.rng_sim` uses the existing enums and dataclasses in `game/balatro_models.py` directly:
- **Enums**: `Suit`, `Rank`, `Enhancement`, `Edition`, `Seal`, `Stake`, `PokerHand`, `TarotName`, `PlanetName`, `SpectralName`, `VoucherName`, `Deck as DeckType`, `SkipTag`, `BlindName`, `JokerName`
- **Models**:
  - `SimCard` subclasses `PlayingCard` (`isinstance(card, PlayingCard)` is `True`), adding `.id` (`sort_id` for Balatro's `table.sort(valid_cards, function(a, b) return a.ID < b.ID end)`), `.card_key`, `.is_suit()`, and `.is_rank()`.
  - `run.state.owned_jokers` stores `game.balatro_models.Joker` instances.
  - `TarotPrediction.to_model()` returns a `game.balatro_models.Tarot` and `.tarot_name` is a `TarotName`.
  - `VoucherPrediction.to_model()` returns a `game.balatro_models.Voucher` and `.voucher_name` is a `VoucherName`.
  - `TagPrediction.skip_tag` is a `SkipTag` and `BossBlindPrediction.blind_name` is a `BlindName`.
  - `BalatroRun.from_gamestate(seed, gs)` and `run.sync_from_gamestate(gs)` synchronize directly with `game.gamestate.GameState`.

