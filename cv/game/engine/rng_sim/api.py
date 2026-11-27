"""Public Python API for Balatro Deterministic Simulation (`game.engine.rng_sim`).

Provides a clean, expressive top-level interface (`BalatroRun`) verified against
`balatro_source` and integrated with `game.balatro_models` (`PlayingCard`, `Joker`,
`Suit`, `Rank`, `Enhancement`, `Edition`, `Seal`, `Stake`, `PokerHand`, `TarotName`,
`VoucherName`, `SkipTag`, `BlindName`, `Deck`) and `game.gamestate.GameState`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from game.balatro_models import (
    BlindName,
    Deck as DeckType,
    Edition,
    Enhancement,
    Joker,
    JokerName,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    SkipTag,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
    VoucherName,
)
from game.engine.rng_sim.events import (
    BOSS_BY_KEY,
    DECK_STARTING_VOUCHERS,
    P_BOSS_BLINDS,
    P_CARDS_KEYS,
    P_SEALS_KEYS,
    P_TAGS,
    P_TAROTS,
    P_VOUCHERS,
    TAG_BY_KEY,
    TAROT_BY_KEY,
    VOUCHER_BY_KEY,
    BossBlindPrediction,
    CertificateResult,
    EventResolver,
    IdolTarget,
    LuckyHitResult,
    RebateEvaluation,
    StoneCardResult,
    TagPrediction,
    TarotPrediction,
    VoucherPrediction,
    evaluate_lucky_hit,
    evaluate_mail_in_rebate_discard,
    get_ancient_joker_suit,
    get_castle_suit,
    get_idol_card,
    get_mail_in_rebate_target,
    get_next_boss_blind,
    get_next_tag,
    get_next_voucher,
    get_orbital_tag_hands_for_ante,
    get_to_do_list_hand,
    normalize_boss_key,
    normalize_tag_key,
    normalize_tarot_key,
    normalize_voucher_key,
    poll_tarot_pool,
    predict_cartomancer_tarot,
    predict_certificate,
    predict_marble_joker_card,
    predict_stone_joker_card,
    predict_vagabond_tarot,
)
from game.engine.rng_sim.rng import BalatroRNG, pseudohash, pseudoseed
from game.engine.rng_sim.state import (
    ALL_POKER_HANDS_ORDER,
    DEFAULT_VISIBLE_POKER_HANDS,
    Card,
    Deck,
    GameState,
    JokerInstance,
    SimCard,
    SimDeck,
    SimGameState,
)

# Allowed Balatro seed characters: 1-9 and A-Z (excluding 0)
BALATRO_SEED_PATTERN = re.compile(r"^[1-9A-Z]{1,8}$")


def validate_seed(seed: str) -> tuple[bool, str]:
    """Validate and sanitize a Balatro seed.

    Balatro run seeds are strings up to 8 characters composed of '1'-'9' and 'A'-'Z'.
    Lowercase characters are automatically capitalized.

    Returns `(is_valid, sanitized_seed_or_error_message)`.
    """
    clean = seed.strip().upper()
    if not clean:
        return False, "Seed cannot be empty."
    if len(clean) > 8:
        return False, f"Seed length cannot exceed 8 characters (got {len(clean)})."
    invalid_chars = [c for c in clean if not (c.isdigit() or c.isalpha()) or c == "0"]
    if invalid_chars:
        return (
            False,
            f"Seed contains invalid character(s): {set(invalid_chars)}. Allowed: 1-9, A-Z.",
        )
    return True, clean


class BalatroRun:
    """Deterministic simulation interface for a Balatro run."""

    def __init__(
        self,
        seed: str = "HELLO",
        stake: Stake | str = Stake.WHITE,
        deck: SimDeck | DeckType | Sequence[PlayingCard] | str | None = None,
        validate: bool = True,
    ) -> None:
        if validate:
            is_valid, sanitized = validate_seed(seed)
            if not is_valid:
                raise ValueError(f"Invalid Balatro seed '{seed}': {sanitized}")
            self.seed = sanitized
        else:
            self.seed = str(seed).strip().upper()

        self.rng = BalatroRNG(self.seed)

        deck_type_enum: DeckType | str = DeckType.RED_DECK
        if isinstance(deck, (DeckType, str)):
            deck_type_enum = DeckType(deck) if deck in DeckType._value2member_map_ else deck
            lower_deck = str(deck.value if isinstance(deck, DeckType) else deck).strip().lower()
            if "abandoned" in lower_deck:
                resolved_deck = SimDeck.create_abandoned_40()
            elif "checkered" in lower_deck:
                resolved_deck = SimDeck.create_checkered_52()
            elif "erratic" in lower_deck:
                resolved_deck = SimDeck.create_erratic_52(self.rng)
            else:
                resolved_deck = SimDeck.create_standard_52()
        elif isinstance(deck, SimDeck):
            resolved_deck = deck
        elif isinstance(deck, Sequence):
            resolved_deck = SimDeck.from_playing_cards(deck)
        else:
            resolved_deck = SimDeck.create_standard_52()

        self.state = SimGameState(
            seed=self.seed,
            stake=Stake(stake) if isinstance(stake, str) else stake,
            deck=resolved_deck,
            deck_type=deck_type_enum,
        )

        # Apply starting deck vouchers if applicable (Magic, Nebula, Zodiac)
        dt_key = str(
            deck_type_enum.value if isinstance(deck_type_enum, DeckType) else deck_type_enum
        ).strip().lower()
        starting_vouchers = DECK_STARTING_VOUCHERS.get(dt_key, ())
        for v_key in starting_vouchers:
            self.state.used_vouchers.add(v_key)

        self.events = EventResolver(self.rng)

    @classmethod
    def from_gamestate(
        cls,
        seed: str,
        gs: Any,
        validate: bool = True,
    ) -> BalatroRun:
        """Create a `BalatroRun` initialized from a live `game.gamestate.GameState`."""
        run = cls(
            seed=seed,
            stake=getattr(gs, "stake", Stake.WHITE),
            deck=getattr(gs, "deck_type", DeckType.RED_DECK),
            validate=validate,
        )
        run.sync_from_gamestate(gs)
        return run

    # -----------------------------------------------------------------------
    # State & Mid-Ante Deck Synchronization
    # -----------------------------------------------------------------------

    def set_state(
        self,
        ante: int | None = None,
        round_num: int | None = None,
        total_rounds: int | None = None,
        money: int | None = None,
        stake: Stake | str | None = None,
    ) -> None:
        """Fast-forward / configure path state parameters."""
        if ante is not None:
            self.state.ante = int(ante)
        if round_num is not None:
            self.state.round_within_ante = int(round_num)
        if total_rounds is not None:
            self.state.total_rounds = int(total_rounds)
        if money is not None:
            self.state.money = int(money)
        if stake is not None:
            self.state.stake = Stake(stake) if isinstance(stake, str) else stake

    def update_deck(
        self,
        cards: Sequence[PlayingCard] | SimDeck,
        preserve_existing_ids: bool = True,
    ) -> None:
        """Update the active deck mid-ante or mid-round from a list of `PlayingCard`s or a `SimDeck`.

        When the deck changes in the middle of an ante (e.g., destroying cards with Hanged Man,
        converting suits with World/Star/Sun/Moon, or adding cards in the shop), call `update_deck()`
        (or mutate `run.state.deck` directly) before querying the next round's Castle / Idol / Mail-in Rebate.
        """
        if isinstance(cards, SimDeck):
            self.state.deck = cards
        else:
            self.state.deck.sync_from_playing_cards(
                cards, preserve_existing_ids=preserve_existing_ids
            )

    def sync_from_gamestate(self, gs: Any) -> None:
        """Synchronize `ante`, `round`, `money`, `full_deck`, `jokers`, and `vouchers` from `game.gamestate.GameState`."""
        self.state.sync_from_gamestate(gs)

    def start_ante(self, ante_num: int) -> None:
        """Transition into a new ante."""
        self.state.start_ante(ante_num)

    def start_round(self, round_num: int | None = None) -> None:
        """Begin a new round within the current ante."""
        self.state.start_round(round_num)

    def end_round(self) -> None:
        """Conclude the active round."""
        self.state.end_round()

    # -----------------------------------------------------------------------
    # Joker & Hand Management
    # -----------------------------------------------------------------------

    def add_joker(
        self,
        name: JokerName | str | Joker,
        edition: Edition | str = Edition.NONE,
        eternal: bool = False,
        perishable: bool = False,
        rental: bool = False,
        state: dict[str, Any] | None = None,
    ) -> Joker:
        """Add an active joker (`game.balatro_models.Joker`) to `state.owned_jokers`."""
        ed = Edition(edition) if isinstance(edition, str) else edition
        return self.state.add_joker(
            name,
            edition=ed,
            eternal=eternal,
            perishable=perishable,
            rental=rental,
            state=state,
        )

    def remove_joker(self, target: int | JokerName | str) -> Joker | None:
        """Remove a joker by index or name."""
        return self.state.remove_joker(target)

    def has_joker(self, name: JokerName | str) -> bool:
        """Check whether an active joker is present."""
        return self.state.has_joker(name)

    def unlock_secret_hand(self, hand_name: PokerHand | str) -> None:
        """Unlock a secret poker hand (`Flush Five`, `Flush House`, `Five of a Kind`) into `visible_hands`."""
        self.state.unlock_secret_hand(hand_name)

    # -----------------------------------------------------------------------
    # 1. Castle Suit Selection (`reset_castle_card`)
    # -----------------------------------------------------------------------

    def get_castle_suit(
        self,
        ante: int | None = None,
        round_num: int | None = None,
    ) -> Suit:
        """Determine the suit Castle selects for the round (`'cas' + str(ante)`).

        How to use when the deck changes mid-ante:
        - **Stateful round-by-round (recommended)**: Call `get_castle_suit(ante=1)` in Round 1
          (advances `'cas1'` call `0 -> 1`). When the deck changes after Round 1, mutate
          `run.state.deck` (or call `run.update_deck(new_cards)`) and call `get_castle_suit(ante=1)`
          in Round 2. Because `run.rng` remembers `'cas1'` is already at call 1, it directly
          evaluates Round 2 (call `1 -> 2`) against the updated deck without re-running Round 1!
        - **Direct round jump (`round_num=1|2|3`)**: Pass `round_num=2` (Big Blind) or `round_num=3`
          (Boss Blind) to fast-forward `'cas{ante}'` directly to that round's LCG state and evaluate
          against the current deck without computing earlier rounds.
        """
        target_ante = ante if ante is not None else self.state.ante
        return get_castle_suit(
            rng=self.rng,
            deck=self.state.deck,
            ante=target_ante,
            state=self.state,
            round_num=round_num,
        )

    def predict_castle_suits_for_ante(self, ante: int, rounds: int = 3) -> list[Suit]:
        """Predict the sequence of Castle suits across `rounds` (Small, Big, Boss) in `ante` without mutating main RNG."""
        forked = self.fork()
        return [forked.get_castle_suit(ante=ante) for _ in range(rounds)]

    def predict_castle_suits_across_antes(
        self,
        antes: Sequence[int] = range(1, 9),
        rounds_per_ante: int = 3,
    ) -> dict[int, list[Suit]]:
        """Predict Castle suits across multiple antes (default Antes 1..8, 3 rounds each) without mutating main RNG."""
        forked = self.fork()
        result: dict[int, list[Suit]] = {}
        for a in antes:
            result[a] = [forked.get_castle_suit(ante=a) for _ in range(rounds_per_ante)]
        return result

    # -----------------------------------------------------------------------
    # 2. Ancient Joker Suit Selection (`reset_ancient_card`)
    # -----------------------------------------------------------------------

    def get_ancient_joker_suit(
        self,
        ante: int | None = None,
        current_suit: Suit | str | None = None,
    ) -> Suit:
        """Determine the suit Ancient Joker selects for the round (`'anc' + str(ante)`).

        Note: On the very first roll of a run (`state.ancient_suit is None`), all 4 suits
        `['Spades', 'Hearts', 'Clubs', 'Diamonds']` are eligible. On all subsequent rounds,
        the previous round's suit is excluded so Ancient Joker never picks the same suit twice in a row.
        """
        target_ante = ante if ante is not None else self.state.ante
        return get_ancient_joker_suit(
            rng=self.rng,
            ante=target_ante,
            current_suit=current_suit,
            state=self.state,
        )

    def predict_ancient_joker_suits(
        self,
        antes: Sequence[int] = range(1, 9),
        rounds_per_ante: int = 3,
        skips_per_ante: dict[int, int] | None = None,
    ) -> dict[int, list[Suit]]:
        """Predict Ancient Joker suits across `antes` without mutating main RNG.

        Args:
            antes: Sequence of ante numbers (default 1..8).
            rounds_per_ante: Default 3 (Small, Big, Boss).
            skips_per_ante: Optional mapping `{ante: num_skips}` (0, 1, or 2 blinds skipped in that ante).
                Because `reset_ancient_card()` is called at the end of defeated blinds, skipping a blind
                reduces the number of calls to `'anc{ante}'` in that ante and carries the un-reset suit forward!
        """
        forked = self.fork()
        result: dict[int, list[Suit]] = {}
        skips = skips_per_ante or {}
        for a in antes:
            num_skips = skips.get(a, 0)
            calls = max(1, rounds_per_ante - num_skips)
            result[a] = [forked.get_ancient_joker_suit(ante=a) for _ in range(calls)]
        return result

    # -----------------------------------------------------------------------
    # 3. The Idol Card Selection (`reset_idol_card`)
    # -----------------------------------------------------------------------

    def get_idol_card(
        self,
        ante: int | None = None,
        round_num: int | None = None,
    ) -> IdolTarget:
        """Determine the card (rank + suit) The Idol selects for the round (`'idol' + str(ante)`).

        Supports both stateful round-by-round progression (`round_num=None`) and direct jumping
        to a specific round within the ante (`round_num=1|2|3`) without evaluating earlier rounds.
        """
        target_ante = ante if ante is not None else self.state.ante
        return get_idol_card(
            rng=self.rng,
            deck=self.state.deck,
            ante=target_ante,
            state=self.state,
            round_num=round_num,
        )

    def predict_idol_cards_for_ante(self, ante: int, rounds: int = 3) -> list[IdolTarget]:
        """Predict The Idol target cards across `rounds` (Small, Big, Boss) in `ante` without mutating main RNG."""
        forked = self.fork()
        return [forked.get_idol_card(ante=ante) for _ in range(rounds)]

    def predict_idol_cards_across_antes(
        self,
        antes: Sequence[int] = range(1, 9),
        rounds_per_ante: int = 3,
    ) -> dict[int, list[IdolTarget]]:
        """Predict The Idol target cards across multiple antes without mutating main RNG."""
        forked = self.fork()
        result: dict[int, list[IdolTarget]] = {}
        for a in antes:
            result[a] = [forked.get_idol_card(ante=a) for _ in range(rounds_per_ante)]
        return result

    # -----------------------------------------------------------------------
    # 4. To Do List Poker Hand Selection (`'to_do'`)
    # -----------------------------------------------------------------------

    def get_to_do_list_hand(
        self,
        is_initial_spawn: bool = False,
        current_hand: PokerHand | str | None = None,
        visible_hands: Sequence[PokerHand | str] | None = None,
    ) -> PokerHand:
        """Determine the poker hand To Do List selects (`'to_do'`).

        - On initial creation (`is_initial_spawn=True` or when `state.to_do_hand is None`),
          draws from all `visible_hands` (9 hands by default).
        - At the end of each round (`is_initial_spawn=False` with a current hand set),
          excludes `current_hand` from `visible_hands` and draws from the remaining 8 hands.
        """
        return get_to_do_list_hand(
            rng=self.rng,
            current_hand=current_hand,
            visible_hands=visible_hands,
            state=self.state,
            is_initial_spawn=is_initial_spawn,
        )

    def predict_to_do_list_hands(
        self,
        rounds: int = 8,
        initial_hand: PokerHand | str | None = None,
        secret_hand_unlocks: dict[int, Sequence[PokerHand | str]] | None = None,
    ) -> list[PokerHand]:
        """Predict a sequence of To Do List poker hands across `rounds` without mutating main RNG.

        Args:
            rounds: Number of selections to predict (step 0 is initial spawn if `initial_hand is None`,
                followed by `end_of_round` resets).
            initial_hand: Optional starting hand already on To Do List.
            secret_hand_unlocks: Optional mapping `{step_index: [PokerHand.FIVE_OF_A_KIND, ...]}` unlocking
                secret poker hands before that step's roll.
        """
        forked = self.fork()
        if initial_hand is not None:
            forked.state.to_do_hand = (
                initial_hand
                if isinstance(initial_hand, PokerHand)
                else PokerHand(initial_hand)
            )
        unlocks = secret_hand_unlocks or {}
        hands: list[PokerHand] = []
        for step in range(rounds):
            if step in unlocks:
                for secret_hand in unlocks[step]:
                    forked.unlock_secret_hand(secret_hand)
            h = forked.get_to_do_list_hand(
                is_initial_spawn=(step == 0 and initial_hand is None)
            )
            hands.append(h)
        return hands

    # -----------------------------------------------------------------------
    # 5 & 6. Cartomancer & Vagabond Tarot Predictions (`'Tarotcar{ante}'`, `'Tarotvag{ante}'`)
    # -----------------------------------------------------------------------

    def predict_cartomancer_tarot(
        self,
        ante: int | None = None,
        active_tarots: set[str | TarotName] | None = None,
        has_showman: bool = False,
    ) -> TarotPrediction:
        """Predict which Tarot card Cartomancer will give when a blind is selected (`'Tarotcar' + str(ante)`)."""
        target_ante = ante if ante is not None else self.state.ante
        return predict_cartomancer_tarot(
            rng=self.rng,
            ante=target_ante,
            state=self.state,
            active_tarots=active_tarots,
            has_showman=has_showman,
        )

    def batch_predict_cartomancer_tarots(
        self,
        antes: Sequence[int] = range(1, 9),
        triggers_per_ante: int = 3,
        active_tarots: set[str | TarotName] | None = None,
    ) -> dict[int, list[TarotPrediction]]:
        """Predict Cartomancer Tarot cards across multiple antes without mutating main RNG."""
        forked = self.fork()
        result: dict[int, list[TarotPrediction]] = {}
        for a in antes:
            result[a] = [
                forked.predict_cartomancer_tarot(ante=a, active_tarots=active_tarots)
                for _ in range(triggers_per_ante)
            ]
        return result

    def predict_vagabond_tarot(
        self,
        ante: int | None = None,
        active_tarots: set[str | TarotName] | None = None,
        has_showman: bool = False,
    ) -> TarotPrediction:
        """Predict which Tarot card Vagabond will give when a hand is played with <= $4 (`'Tarotvag' + str(ante)`)."""
        target_ante = ante if ante is not None else self.state.ante
        return predict_vagabond_tarot(
            rng=self.rng,
            ante=target_ante,
            state=self.state,
            active_tarots=active_tarots,
            has_showman=has_showman,
        )

    def batch_predict_vagabond_tarots(
        self,
        antes: Sequence[int] = range(1, 9),
        triggers_per_ante: int = 4,
        active_tarots: set[str | TarotName] | None = None,
    ) -> dict[int, list[TarotPrediction]]:
        """Predict Vagabond Tarot cards across multiple antes without mutating main RNG."""
        forked = self.fork()
        result: dict[int, list[TarotPrediction]] = {}
        for a in antes:
            result[a] = [
                forked.predict_vagabond_tarot(ante=a, active_tarots=active_tarots)
                for _ in range(triggers_per_ante)
            ]
        return result

    # -----------------------------------------------------------------------
    # 7. Marble Joker / "Stone Joker" Card Prediction (`'marb_fr'`)
    # -----------------------------------------------------------------------

    def predict_stone_joker_card(self, add_to_deck: bool = True) -> StoneCardResult:
        """Predict the Stone Card (and its hidden suit + rank) added on blind select (`'marb_fr'`).

        In Balatro (`card.lua:2580-2601`), Marble Joker ("Stone Joker" in prompt) pulls a card front
        from `G.P_CARDS` using stream `'marb_fr'` and applies the `m_stone` (Stone Card) enhancement.
        If a different enhancement (e.g. Empress, Magician, Lovers) or Vampire is later used on that card,
        its hidden `suit` and `rank` are revealed.
        """
        return predict_stone_joker_card(
            rng=self.rng,
            state=self.state,
            add_to_deck=add_to_deck,
        )

    def predict_marble_joker_card(self, add_to_deck: bool = True) -> StoneCardResult:
        """Alias for `predict_stone_joker_card`."""
        return self.predict_stone_joker_card(add_to_deck=add_to_deck)

    def batch_predict_stone_joker_cards(
        self, count: int = 6, add_to_deck: bool = False
    ) -> list[StoneCardResult]:
        """Predict the next `count` Stone Cards generated by Marble/Stone Joker without mutating main RNG."""
        forked = self.fork()
        return [
            forked.predict_stone_joker_card(add_to_deck=add_to_deck)
            for _ in range(count)
        ]

    # -----------------------------------------------------------------------
    # 8. Order of Boss Blinds (`get_new_boss`, stream `'boss'`)
    # -----------------------------------------------------------------------

    def get_next_boss_blind(
        self, ante: int | None = None, is_reroll: bool = False
    ) -> BossBlindPrediction:
        """Roll the next Boss Blind for `ante` (updates `state.bosses_used` and `state.current_boss`)."""
        target_ante = ante if ante is not None else self.state.ante
        return get_next_boss_blind(
            rng=self.rng,
            ante=target_ante,
            state=self.state,
            is_reroll=is_reroll,
        )

    def reroll_boss_blind(self, ante: int | None = None) -> BossBlindPrediction:
        """Reroll the current Boss Blind (via Director's Cut, Retcon, or Boss Tag)."""
        return self.get_next_boss_blind(ante=ante, is_reroll=True)

    def predict_boss_blinds(
        self,
        max_ante: int = 8,
        rerolls_at_step: dict[int, int] | None = None,
        minus_ante_vouchers_at_step: dict[int, int] | Sequence[int] | None = None,
        rerolls_after_voucher_at_step: dict[int, int] | None = None,
        mutate_state: bool = False,
    ) -> list[BossBlindPrediction]:
        """Predict the full sequence of Boss Blinds in a seed, accounting for all edge cases.

        Edge cases supported (`common_events.lua:2338-2383`, `card.lua:2065-2075`):
        1. Minimum ante locks (`v.boss.min <= math.max(1, ante)`).
        2. Showdown bosses at multiples of `win_ante` (`ante % 8 == 0 and ante >= 2`),
           and Normal bosses at all other antes (including `ante <= 1`, e.g. `ante = 0` after
           redeeming Hieroglyph in Ante 1, because `ante < 2` forces normal bosses even though `0 % 8 == 0`!).
        3. `bosses_used` cycle tracking: a boss cannot repeat until every eligible boss in its pool
           (Normal or Showdown) has been rolled `min_use` times.
        4. Boss rerolls (`rerolls_at_step={step_index: num_rerolls}`): Director's Cut, Retcon, or Boss Tag
           rolls a new boss at the current `ante`. Because the replaced boss already incremented
           `bosses_used`, it cannot immediately reappear unless it was the last boss in the cycle.
        5. `-1 ante` vouchers (`Hieroglyph` / `Petroglyph`, via `minus_ante_vouchers_at_step`):
           - Redeeming Hieroglyph or Petroglyph during step `s` (1-based step index) immediately decrements
             `G.GAME.round_resets.ante` by 1 (`ease_ante(-1)`).
           - If the player rerolls the boss *after* buying Hieroglyph/Petroglyph in that same ante
             (`rerolls_after_voucher_at_step={s: count}`), `get_new_boss()` rolls using the **decremented**
             `ante` (e.g., buying Hieroglyph in Ante 8 drops `ante` to 7; rerolling the boss in Ante 8
             after buying Hieroglyph rolls a Normal boss for `ante=7` instead of a Showdown boss!).
           - When the current ante's boss is defeated, `ease_ante(1)` increments `ante` back to the ante
             number just played, so `get_new_boss()` rolls another boss for that repeated ante!
        """
        runner = self if mutate_state else self.fork()
        rerolls_before = rerolls_at_step or {}
        rerolls_after = rerolls_after_voucher_at_step or {}

        if isinstance(minus_ante_vouchers_at_step, dict):
            voucher_drops = dict(minus_ante_vouchers_at_step)
        elif minus_ante_vouchers_at_step is not None:
            voucher_drops = {}
            for s in minus_ante_vouchers_at_step:
                voucher_drops[s] = voucher_drops.get(s, 0) + 1
        else:
            voucher_drops = {}

        predictions: list[BossBlindPrediction] = []
        current_ante = 1
        step = 1

        while current_ante <= max_ante:
            runner.state.ante = current_ante
            initial_pred = runner.get_next_boss_blind(ante=current_ante, is_reroll=False)
            predictions.append(initial_pred)

            for _ in range(rerolls_before.get(step, 0)):
                rr_pred = runner.reroll_boss_blind(ante=current_ante)
                predictions.append(rr_pred)

            drop = voucher_drops.get(step, 0)
            if drop > 0:
                current_ante -= drop
                runner.state.ante = current_ante

                for _ in range(rerolls_after.get(step, 0)):
                    rr_after = runner.reroll_boss_blind(ante=current_ante)
                    predictions.append(rr_after)

            current_ante += 1
            step += 1

        return predictions

    # -----------------------------------------------------------------------
    # 9. Order of Skip Tags (`get_next_tag_key`, stream `'Tag{ante}'`)
    # -----------------------------------------------------------------------

    def get_next_tag(
        self,
        ante: int | None = None,
        blind_type: str = "Small",
        roll_orbital_choice: bool = False,
    ) -> TagPrediction:
        """Roll the next Skip Tag for `ante` (`'Tag' + str(ante)`)."""
        target_ante = ante if ante is not None else self.state.ante
        return get_next_tag(
            rng=self.rng,
            ante=target_ante,
            blind_type=blind_type,
            state=self.state,
            roll_orbital_choice=roll_orbital_choice,
        )

    def predict_ante_tags(
        self,
        ante: int | None = None,
        include_orbital_hands: bool = False,
        mutate_state: bool = False,
    ) -> tuple[TagPrediction, TagPrediction]:
        """Predict the `(Small Blind Tag, Big Blind Tag)` pair for `ante`."""
        runner = self if mutate_state else self.fork()
        target_ante = ante if ante is not None else runner.state.ante

        small_tag = get_next_tag(
            rng=runner.rng,
            ante=target_ante,
            blind_type="Small",
            state=runner.state,
        )
        big_tag = get_next_tag(
            rng=runner.rng,
            ante=target_ante,
            blind_type="Big",
            state=runner.state,
        )

        if include_orbital_hands:
            orb = get_orbital_tag_hands_for_ante(
                rng=runner.rng,
                state=runner.state,
            )
            if small_tag.key == "tag_orbital":
                small_tag.orbital_hand = orb["Small"]
            if big_tag.key == "tag_orbital":
                big_tag.orbital_hand = orb["Big"]

        return small_tag, big_tag

    def predict_skip_tags(
        self,
        max_ante: int = 8,
        minus_ante_vouchers_at_step: dict[int, int] | Sequence[int] | None = None,
        include_orbital_hands: bool = False,
        mutate_state: bool = False,
    ) -> list[tuple[TagPrediction, TagPrediction]]:
        """Predict the full sequence of `(Small Tag, Big Tag)` pairs across a run."""
        runner = self if mutate_state else self.fork()

        if isinstance(minus_ante_vouchers_at_step, dict):
            voucher_drops = dict(minus_ante_vouchers_at_step)
        elif minus_ante_vouchers_at_step is not None:
            voucher_drops = {}
            for s in minus_ante_vouchers_at_step:
                voucher_drops[s] = voucher_drops.get(s, 0) + 1
        else:
            voucher_drops = {}

        results: list[tuple[TagPrediction, TagPrediction]] = []
        current_ante = 1
        step = 1

        while current_ante <= max_ante:
            runner.state.ante = current_ante
            pair = runner.predict_ante_tags(
                ante=current_ante,
                include_orbital_hands=include_orbital_hands,
                mutate_state=True,
            )
            results.append(pair)

            drop = voucher_drops.get(step, 0)
            if drop > 0:
                current_ante -= drop

            current_ante += 1
            step += 1

        return results

    # -----------------------------------------------------------------------
    # 10. Order of Vouchers (`get_next_voucher_key`, `'Voucher{ante}'` & `'Voucher_fromtag'`)
    # -----------------------------------------------------------------------

    def get_next_voucher(
        self,
        ante: int | None = None,
        from_tag: bool = False,
        add_to_shop: bool = True,
    ) -> VoucherPrediction:
        """Roll the next Voucher (`'Voucher' + str(ante)` for shop, or `'Voucher_fromtag'` for Voucher Tag)."""
        target_ante = ante if ante is not None else self.state.ante
        if not from_tag and add_to_shop:
            self.state.shop_vouchers.clear()
        return get_next_voucher(
            rng=self.rng,
            ante=target_ante,
            from_tag=from_tag,
            state=self.state,
            add_to_shop_vouchers=add_to_shop,
        )

    def predict_voucher_from_tag(
        self,
        ante: int | None = None,
        add_to_shop: bool = True,
    ) -> VoucherPrediction:
        """Roll a bonus Voucher from taking a Voucher Tag skip (`_from_tag = true`, stream `'Voucher_fromtag'`)."""
        return self.get_next_voucher(
            ante=ante,
            from_tag=True,
            add_to_shop=add_to_shop,
        )

    def redeem_voucher(
        self,
        voucher_key_or_name: str | VoucherName,
        apply_ante_effect: bool = False,
    ) -> str:
        """Redeem (purchase) a voucher, updating `state.used_vouchers` and unlocking its Tier-2 upgrade."""
        v_key = normalize_voucher_key(voucher_key_or_name)
        self.state.used_vouchers.add(v_key)
        if v_key in self.state.shop_vouchers:
            self.state.shop_vouchers.remove(v_key)
        if apply_ante_effect and v_key in ("v_hieroglyph", "v_petroglyph"):
            self.state.ante -= 1
        return v_key

    def predict_voucher_sequence(
        self,
        max_ante: int = 8,
        purchase_plan: bool | dict[int, bool | Sequence[str | VoucherName]] = False,
        voucher_tags_at_step: dict[int, int] | None = None,
        minus_ante_vouchers_at_step: dict[int, int] | Sequence[int] | None = None,
        mutate_state: bool = False,
    ) -> list[VoucherPrediction]:
        """Predict the sequence of Vouchers across a run, adapting dynamically to purchase order and Voucher Tags."""
        runner = self if mutate_state else self.fork()
        tag_counts = voucher_tags_at_step or {}

        if isinstance(minus_ante_vouchers_at_step, dict):
            explicit_drops = dict(minus_ante_vouchers_at_step)
        elif minus_ante_vouchers_at_step is not None:
            explicit_drops = {}
            for s in minus_ante_vouchers_at_step:
                explicit_drops[s] = explicit_drops.get(s, 0) + 1
        else:
            explicit_drops = {}

        def _should_buy(step_idx: int, v_pred: VoucherPrediction) -> bool:
            if isinstance(purchase_plan, bool):
                return purchase_plan
            rule = purchase_plan.get(step_idx, False)
            if isinstance(rule, bool):
                return rule
            normalized_targets = {normalize_voucher_key(x) for x in rule}
            return v_pred.key in normalized_targets

        history: list[VoucherPrediction] = []
        current_ante = 1
        step = 1

        while current_ante <= max_ante:
            runner.state.ante = current_ante
            ante_drop_this_step = explicit_drops.get(step, 0)

            shop_v = runner.get_next_voucher(
                ante=current_ante, from_tag=False, add_to_shop=True
            )
            history.append(shop_v)

            if _should_buy(step, shop_v):
                runner.redeem_voucher(shop_v.key)
                if (
                    shop_v.key in ("v_hieroglyph", "v_petroglyph")
                    and step not in explicit_drops
                ):
                    ante_drop_this_step += 1

            for _ in range(tag_counts.get(step, 0)):
                tag_v = runner.predict_voucher_from_tag(
                    ante=current_ante, add_to_shop=True
                )
                history.append(tag_v)
                if _should_buy(step, tag_v):
                    runner.redeem_voucher(tag_v.key)
                    if (
                        tag_v.key in ("v_hieroglyph", "v_petroglyph")
                        and step not in explicit_drops
                    ):
                        ante_drop_this_step += 1

            current_ante -= ante_drop_this_step
            current_ante += 1
            step += 1

        return history

    # -----------------------------------------------------------------------
    # Existing Event Predictions & Roll Solvers
    # -----------------------------------------------------------------------

    def predict_certificate(
        self,
        ante: int | None = None,
        add_to_deck: bool = True,
        ante_scoped: bool = True,
    ) -> CertificateResult:
        """Query deterministic Certificate joker outcome."""
        target_ante = ante if ante is not None else self.state.ante
        return predict_certificate(
            rng=self.rng,
            ante=target_ante,
            state=self.state,
            add_to_state=add_to_deck,
            ante_scoped=ante_scoped,
        )

    def evaluate_lucky_hit(self) -> dict[str, bool]:
        """Predict dynamic mid-round triggers for a scored Lucky Card."""
        modifier = self.state.probability_modifier
        res = evaluate_lucky_hit(
            rng=self.rng,
            probability_modifier=modifier,
        )
        return res.to_dict()

    def get_mail_in_rebate_target(
        self,
        ante: int | None = None,
        round_num: int | None = None,
    ) -> Rank:
        """Query dynamically chosen Mail-in Rebate target rank for the current deck (`'mail' + str(ante)`).

        Supports both stateful round-by-round progression (`round_num=None`) and direct jumping
        to a specific round within the ante (`round_num=1|2|3`) without evaluating earlier rounds.
        """
        target_ante = ante if ante is not None else self.state.ante
        return get_mail_in_rebate_target(
            rng=self.rng,
            deck=self.state.deck,
            ante=target_ante,
            state=self.state,
            round_num=round_num,
        )

    def evaluate_mail_in_rebate_discard(
        self,
        discarded_cards: Sequence[PlayingCard],
        target_rank: Rank | None = None,
    ) -> RebateEvaluation:
        """Evaluate payout from discarding candidates with Mail-in Rebate active."""
        rank = (
            target_rank
            if target_rank is not None
            else self.get_mail_in_rebate_target()
        )
        return evaluate_mail_in_rebate_discard(
            target_rank=rank,
            discarded_cards=discarded_cards,
        )

    def evaluate_wheel_of_fortune(self) -> bool:
        """Resolve The Wheel of Fortune (1 in 4 base chance)."""
        return self.events.wheel_of_fortune(self.state.probability_modifier)

    def evaluate_space_joker(self, stream_key: str = "space_joker") -> bool:
        """Resolve Space Joker hand level upgrade (1 in 4 base chance)."""
        return self.events.space_joker(
            self.state.probability_modifier, stream_key=stream_key
        )

    def evaluate_bloodstone(self, base_denominator: int = 2) -> bool:
        """Resolve Bloodstone extra mult check (1 in 2 base chance)."""
        return self.events.bloodstone(
            oops_multiplier=self.state.probability_modifier,
            base_denominator=base_denominator,
        )

    def evaluate_glass_shatter(self) -> bool:
        """Resolve Glass Card shatter upon scoring (1 in 4 base chance)."""
        return self.events.glass_card_shatter(self.state.probability_modifier)

    def evaluate_business_card(self, stream_key: str = "business_card") -> bool:
        """Resolve Business Card $2 payout on face card scoring (1 in 2 base chance)."""
        return self.events.business_card(
            self.state.probability_modifier, stream_key=stream_key
        )

    def evaluate_gros_michel(self) -> bool:
        """Resolve Gros Michel extinction check at end of round (1 in 6 base chance)."""
        return self.events.gros_michel_extinct(self.state.probability_modifier)

    def evaluate_cavendish(self) -> bool:
        """Resolve Cavendish extinction check at end of round (1 in 1000 base chance)."""
        return self.events.cavendish_extinct(self.state.probability_modifier)

    # -----------------------------------------------------------------------
    # Batch Lookup Functions
    # -----------------------------------------------------------------------

    def batch_predict_certificates(
        self, antes: list[int], ante_scoped: bool = True
    ) -> list[CertificateResult]:
        """Predict Certificate outcomes across multiple future antes without mutating deck."""
        results: list[CertificateResult] = []
        if ante_scoped:
            for ante in antes:
                forked_rng = self.rng.fork()
                res = predict_certificate(
                    rng=forked_rng,
                    ante=ante,
                    add_to_state=False,
                    ante_scoped=True,
                )
                results.append(res)
        else:
            forked_rng = self.rng.fork()
            for ante in antes:
                res = predict_certificate(
                    rng=forked_rng,
                    ante=ante,
                    add_to_state=False,
                    ante_scoped=False,
                )
                results.append(res)
        return results

    def batch_evaluate_lucky_hits(self, count: int) -> list[dict[str, bool]]:
        """Evaluate a sequence of consecutive Lucky Card scoring checks."""
        return [self.evaluate_lucky_hit() for _ in range(count)]

    def batch_evaluate_wheel_of_fortune(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Wheel of Fortune usages."""
        return [self.evaluate_wheel_of_fortune() for _ in range(count)]

    def batch_evaluate_space_joker(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Space Joker played hand checks."""
        return [self.evaluate_space_joker() for _ in range(count)]

    def batch_evaluate_glass_shatter(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Glass Card scoring checks."""
        return [self.evaluate_glass_shatter() for _ in range(count)]

    def batch_evaluate_business_card(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Business Card face card scoring checks."""
        return [self.evaluate_business_card() for _ in range(count)]

    # -----------------------------------------------------------------------
    # Stream Inspection, Advancing, & Checkpointing
    # -----------------------------------------------------------------------

    def advance_stream(self, key: str, steps: int = 1) -> float:
        """Arbitrarily advance a named PRNG stream channel."""
        return self.rng.advance(key, steps)

    def get_stream_state(self, key: str) -> float | None:
        """Inspect the raw LCG state of a PRNG stream channel."""
        return self.rng.inspect_state(key)

    def get_stream_call_count(self, key: str) -> int:
        """Get the number of times a PRNG stream channel has been called."""
        return self.rng.get_channel_call_count(key)

    def checkpoint(self) -> dict[str, Any]:
        """Save a complete snapshot checkpoint of the run state and RNG pipeline."""
        return {
            "seed": self.seed,
            "rng": self.rng.checkpoint(),
            "state": self.state.checkpoint(),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore the run state and RNG pipeline from a checkpoint snapshot."""
        self.seed = snap["seed"]
        self.rng.restore(snap["rng"])
        self.state.restore(snap["state"])
        self.events = EventResolver(self.rng)

    def fork(self) -> BalatroRun:
        """Create an independent deep copy of this run simulation."""
        cloned = BalatroRun(self.seed, validate=False)
        cloned.rng = self.rng.fork()
        cloned.state = self.state.clone()
        cloned.events = EventResolver(cloned.rng)
        return cloned

    def clone(self) -> BalatroRun:
        """Alias for `fork()`."""
        return self.fork()
