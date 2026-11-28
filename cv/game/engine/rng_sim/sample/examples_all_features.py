"""Runnable Reference Examples for `game.engine.rng_sim`.

Run from the repository root:
    python -m game.engine.rng_sim.sample.examples_all_features

Covers:
0. Mid-Ante Deck Updates (Stateful Round-by-Round Progression vs Direct `round_num` Jump)
   & `game.gamestate.GameState` / `game.balatro_models` Integration
1. Castle Suit Selection (`'cas{ante}'`) + Checkered / Stone Deck Edge Cases
2. Ancient Joker Suit Selection (`'anc{ante}'`) + Blind Skip Edge Case
3. The Idol Card Selection (`'idol{ante}'`) + Deck Thinning / Stone Card Edge Cases
4. To Do List Poker Hand Selection (`'to_do'`) + Secret Hand Unlock Edge Case
5. Cartomancer (`'Tarotcar{ante}'`) & Vagabond (`'Tarotvag{ante}'`) + Held Tarot / Showman Edge Cases
6. Marble Joker / "Stone Joker" (`'marb_fr'`) Hidden Suit & Rank + Re-Enhancement Reveal
7. Boss Blind Queue (`'boss'`) + Rerolls, Hieroglyph/Petroglyph (`-1 ante`), Ante 8 Normal Boss Exploit
8. Skip Tag Queue (`'Tag{ante}'`) + Ante 1 `min_ante=2` Resampling, Hieroglyph, Orbital Tag (`'orbital'`)
9. Voucher Queue (`'Voucher{ante}'` vs `'Voucher_fromtag'`) + Purchase Plans, Voucher Tag Global Queue, Zodiac Deck
10. Certificate, Lucky Cards, Mail-in Rebate, Wheel of Fortune, Space Joker, Bloodstone, Glass, Business Card
"""

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


def example_0_mid_ante_deck_changes_and_gamestate_sync() -> None:
    print("=" * 76)
    print("0. MID-ANTE DECK CHANGES & GAMESTATE / BALATRO_MODELS INTEGRATION")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # Pattern A: Stateful Round-by-Round Progression (No wasted evaluations!)
    # -------------------------------------------------------------------------
    # Because `run.rng` is stateful, calling `get_castle_suit(ante=1)` in Round 1
    # advances `'cas1'` from call 0 -> 1. When the deck changes in Round 1 (or in
    # the shop after Round 1), mutating `run.state.deck` (or calling `run.update_deck(...)`)
    # and calling `get_castle_suit(ante=1)` in Round 2 advances `'cas1'` from call 1 -> 2
    # directly against the NEW deck—Round 1 is never re-generated or thrown away!
    run = BalatroRun(seed="HELLO")

    r1_castle = run.get_castle_suit(ante=1)
    r1_idol = run.get_idol_card(ante=1)
    r1_rebate = run.get_mail_in_rebate_target(ante=1)
    print(
        f"[Stateful R1] Castle={r1_castle.value}, "
        f"Idol={r1_idol.rank.name} of {r1_idol.suit.value}, "
        f"Rebate={r1_rebate.name} (cas1 calls={run.get_stream_call_count('cas1')})"
    )

    # Suppose after Round 1, the player converts all Spades in their deck into Hearts:
    for c in run.state.deck.cards:
        if c.suit == Suit.SPADES:
            c.suit = Suit.HEARTS

    # Now query Round 2 directly (advances 'cas1' call 1 -> 2 on the updated deck):
    r2_castle = run.get_castle_suit(ante=1)
    r2_idol = run.get_idol_card(ante=1)
    print(
        f"[Stateful R2 after Spades->Hearts] Castle={r2_castle.value}, "
        f"Idol={r2_idol.rank.name} of {r2_idol.suit.value} "
        f"(cas1 calls={run.get_stream_call_count('cas1')})"
    )

    # -------------------------------------------------------------------------
    # Pattern B: Direct Jump to `round_num` (Stateless / Snapshot Jump)
    # -------------------------------------------------------------------------
    # Why does jumping directly to Round `n` work without evaluating Rounds `1..n-1`?
    # In Balatro's PRNG, `pseudoseed('cas1')` advances its 64-bit float LCG state
    # (`st = advance(st)`) purely as a function of the call count—completely independent
    # of `len(valid_cards)`! The deck size only affects the ephemeral `Tw223(pseed)`
    # seeded for that specific round's draw.
    # Passing `round_num=2` (or `3`) steps the float LCG to call `n-1` in O(1) math
    # and evaluates `pseudorandom_element` only once against the current deck!
    fresh_run = BalatroRun(seed="HELLO", deck=run.state.deck.clone())
    r2_castle_jump = fresh_run.get_castle_suit(ante=1, round_num=2)
    r2_idol_jump = fresh_run.get_idol_card(ante=1, round_num=2)
    print(
        f"[Direct Jump round_num=2]          Castle={r2_castle_jump.value}, "
        f"Idol={r2_idol_jump.rank.name} of {r2_idol_jump.suit.value}"
    )

    # -------------------------------------------------------------------------
    # Pattern C: Syncing Directly with `game.gamestate.GameState`
    # -------------------------------------------------------------------------
    gs = EngineGameState(ante=1, round=1)
    sim = BalatroRun.from_gamestate(seed="HELLO", gs=gs)
    # Remove all Diamonds from `gs.full_deck` mid-ante and sync:
    gs.full_deck = [c for c in gs.full_deck if c.suit != Suit.DIAMONDS]
    sim.sync_from_gamestate(gs)
    print(
        f"[Synced from Engine GameState (no Diamonds)] "
        f"Castle R1={sim.get_castle_suit(ante=1, round_num=1).value}\n"
    )


def example_1_castle() -> None:
    print("=" * 76)
    print("1. CASTLE SUIT SELECTION ('cas' + str(ante))")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    suits_a1 = run.predict_castle_suits_for_ante(ante=1, rounds=3)
    print(
        f"Standard 52-card Deck (Ante 1 Small/Big/Boss): {[s.value for s in suits_a1]}"
    )

    # Edge Case 1: Checkered Deck (26 Spades, 26 Hearts)
    checkered_run = BalatroRun(seed="HELLO", deck=DeckType.CHECKERED_DECK)
    print(
        f"Checkered Deck (Ante 1 Small/Big/Boss):        "
        f"{[s.value for s in checkered_run.predict_castle_suits_for_ante(ante=1, rounds=3)]}"
    )

    # Edge Case 2: All Stone Cards -> falls back to Spades ('Spades')
    stone_deck = Deck(
        [
            Card(Suit.HEARTS, Rank.ACE, enhancement=Enhancement.STONE),
            Card(Suit.DIAMONDS, Rank.KING, enhancement=Enhancement.STONE),
        ]
    )
    stone_run = BalatroRun(seed="HELLO", deck=stone_deck)
    print(
        f"All-Stone Deck Fallback (Ante 1):              {stone_run.get_castle_suit(ante=1).value}\n"
    )


def example_2_ancient_joker() -> None:
    print("=" * 76)
    print("2. ANCIENT JOKER SUIT SELECTION ('anc' + str(ante))")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    normal = run.predict_ancient_joker_suits(antes=[1, 2], rounds_per_ante=3)
    print(
        f"No Blinds Skipped -> Ante 1: {[s.value for s in normal[1]]}, Ante 2: {[s.value for s in normal[2]]}"
    )

    # Edge Case: Skipping 1 blind in Ante 1 means `reset_ancient_card` is only called twice
    # in Ante 1, carrying Hearts into Ante 2 Round 1 instead of Spades!
    skipped = run.predict_ancient_joker_suits(
        antes=[1, 2], rounds_per_ante=3, skips_per_ante={1: 1}
    )
    print(
        f"1 Skip in Ante 1  -> Ante 1: {[s.value for s in skipped[1]]}, Ante 2: {[s.value for s in skipped[2]]}\n"
    )


def example_3_idol() -> None:
    print("=" * 76)
    print("3. THE IDOL CARD SELECTION ('idol' + str(ante))")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    idols_a1 = run.predict_idol_cards_for_ante(ante=1, rounds=3)
    for idx, target in enumerate(idols_a1, start=1):
        print(
            f"Ante 1 Round {idx}: {target.rank.name} of {target.suit.value} (sort_id={target.card_id})"
        )

    # Edge Case: Deck thinned to only 2x Ace of Spades + Stone Cards
    thin_deck = Deck(
        [
            PlayingCard(suit=Suit.SPADES, rank=Rank.ACE),
            PlayingCard(suit=Suit.SPADES, rank=Rank.ACE),
            PlayingCard(suit=Suit.HEARTS, rank=Rank.TWO, enhancement=Enhancement.STONE),
        ]
    )
    thin_run = BalatroRun(seed="HELLO", deck=thin_deck)
    t = thin_run.get_idol_card(ante=1)
    print(f"Thinned Deck (only A of Spades + Stone): {t.rank.name} of {t.suit.value}\n")


def example_4_to_do_list() -> None:
    print("=" * 76)
    print("4. TO DO LIST POKER HAND SELECTION ('to_do')")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    hands = run.predict_to_do_list_hands(rounds=6)
    print(f"Default 9 Visible Hands (6 steps): {[h.value for h in hands]}")

    # Edge Case: Unlocking secret hands ('Five of a Kind', 'Flush Five') before step 2
    hands_secret = run.predict_to_do_list_hands(
        rounds=6,
        secret_hand_unlocks={2: [PokerHand.FIVE_OF_A_KIND, PokerHand.FLUSH_FIVE]},
    )
    print(f"Secret Hands Unlocked at Step 2:   {[h.value for h in hands_secret]}\n")


def example_5_cartomancer_and_vagabond() -> None:
    print("=" * 76)
    print("5 & 6. CARTOMANCER ('Tarotcar{ante}') & VAGABOND ('Tarotvag{ante}')")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    c1 = run.predict_cartomancer_tarot(ante=1)
    v1 = run.predict_vagabond_tarot(ante=1)
    print(f"Cartomancer (Ante 1, no held Tarots): {c1.tarot_name.value} ({c1.key})")
    print(f"Vagabond    (Ante 1, no held Tarots): {v1.tarot_name.value} ({v1.key})")

    # Edge Case 1: Holding 'The Moon' blocks Cartomancer from generating 'The Moon',
    # causing it to resample on `'Tarotcar1_resample2'`!
    run_held = BalatroRun(seed="HELLO")
    c1_held = run_held.predict_cartomancer_tarot(ante=1, active_tarots={TarotName.MOON})
    print(
        f"Cartomancer (holding The Moon):       {c1_held.tarot_name.value} "
        f"(resamples={c1_held.resample_count})"
    )

    # Edge Case 2: Owning Showman allows duplicate Tarots!
    run_showman = BalatroRun(seed="HELLO")
    run_showman.add_joker(JokerName.SHOWMAN)
    c1_showman = run_showman.predict_cartomancer_tarot(
        ante=1, active_tarots={TarotName.MOON}
    )
    print(
        f"Cartomancer (The Moon + Showman):     {c1_showman.tarot_name.value} "
        f"(resamples={c1_showman.resample_count})\n"
    )


def example_6_stone_joker() -> None:
    print("=" * 76)
    print("7. MARBLE / STONE JOKER ('marb_fr') HIDDEN SUIT & RANK")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    cards = run.batch_predict_stone_joker_cards(count=4, add_to_deck=False)
    for i, sc in enumerate(cards, start=1):
        print(
            f"Stone Card #{i}: enhancement={sc.card.enhancement.value}, "
            f"hidden={sc.hidden_rank.name} of {sc.hidden_suit.value} ({sc.front_key})"
        )

    # Edge Case: Enhancing the generated Stone Card with The Empress reveals its hidden base card
    added = run.predict_stone_joker_card(add_to_deck=True)
    run.state.deck.convert_card(added.card, new_enhancement=Enhancement.MULT)
    print(
        f"After The Empress on Stone Card #1: is_suit(Spades)={added.card.is_suit(Suit.SPADES)}, "
        f"is_rank(6)={added.card.is_rank(Rank.SIX)}\n"
    )


def example_7_boss_blinds() -> None:
    print("=" * 76)
    print("8. BOSS BLIND QUEUE ('boss') & EDGE CASES")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    standard = run.predict_boss_blinds(max_ante=8)
    print("Standard Antes 1..8:")
    for b in standard:
        tag = "SHOWDOWN" if b.is_showdown else "Normal"
        print(f"  Ante {b.ante}: {b.name:<18} ({b.key}, {tag})")

    # Edge Case: Buying Hieroglyph in Ante 8 AND rerolling the boss in Ante 8 after buying Hieroglyph!
    # Because Hieroglyph drops `ante` to 7 immediately, rerolling the Ante 8 boss rolls a NORMAL boss (`ante=7`)!
    hiero_a8 = run.predict_boss_blinds(
        max_ante=8,
        minus_ante_vouchers_at_step={8: 1},
        rerolls_after_voucher_at_step={8: 1},
    )
    print("\nEdge Case - Hieroglyph in Ante 8 + Boss Reroll in Ante 8:")
    for b in hiero_a8[7:]:
        kind = "SHOWDOWN" if b.is_showdown else "Normal"
        rr = " [REROLL]" if b.is_reroll else ""
        print(f"  Ante {b.ante}{rr}: {b.name:<18} ({b.key}, {kind})")
    print()


def example_8_skip_tags() -> None:
    print("=" * 76)
    print("9. SKIP TAG QUEUE ('Tag{ante}') & ORBITAL TAG ('orbital')")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    tags = run.predict_skip_tags(max_ante=4, include_orbital_hands=True)
    for pair in tags:
        small, big = pair
        print(
            f"  Ante {small.ante}: Small={small.skip_tag.value:<16} (resamples={small.resample_count}), "
            f"Big={big.skip_tag.value:<16} (resamples={big.resample_count})"
        )
    print()


def example_9_vouchers() -> None:
    print("=" * 76)
    print("10. VOUCHER QUEUE ('Voucher{ante}' vs 'Voucher_fromtag')")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")

    # Compare buying NO vouchers vs buying ALL vouchers:
    unbought = run.predict_voucher_sequence(max_ante=6, purchase_plan=False)
    bought = run.predict_voucher_sequence(max_ante=6, purchase_plan=True)
    print("No Vouchers Bought (Antes 1..6): ", [v.voucher_name.value for v in unbought])
    print("All Vouchers Bought (Antes 1..6):", [v.voucher_name.value for v in bought])

    # Taking 2 Voucher Tags in Ante 1 pulls from global `'Voucher_fromtag'`:
    with_tags = run.predict_voucher_sequence(
        max_ante=2,
        purchase_plan={1: [VoucherName.PLANET_MERCHANT]},
        voucher_tags_at_step={1: 2},
    )
    print("With 2 Voucher Tags in Ante 1:")
    for v in with_tags:
        src = "TAG ('Voucher_fromtag')" if v.from_tag else f"SHOP ('{v.stream_key}')"
        print(
            f"  Ante {v.ante} [{src:<22}]: {v.voucher_name.value} (resamples={v.resample_count})"
        )
    print()


def example_10_certificate_and_probabilities() -> None:
    print("=" * 76)
    print("11. CERTIFICATE, LUCKY CARDS & 1-IN-N PROBABILITY SOLVERS")
    print("=" * 76)

    run = BalatroRun(seed="HELLO")
    run.set_state(ante=6, round_num=2)
    cert = run.predict_certificate()
    print(
        f"Certificate (Ante 6): {cert.rank.name} of {cert.suit.value}, Seal={cert.seal.value}"
    )
    print(f"Next 3 Lucky Card Rolls: {run.batch_evaluate_lucky_hits(3)}")
    print(f"Next 5 Wheel of Fortune Rolls: {run.batch_evaluate_wheel_of_fortune(5)}")


def main() -> None:
    example_0_mid_ante_deck_changes_and_gamestate_sync()
    example_1_castle()
    example_2_ancient_joker()
    example_3_idol()
    example_4_to_do_list()
    example_5_cartomancer_and_vagabond()
    example_6_stone_joker()
    example_7_boss_blinds()
    example_8_skip_tags()
    example_9_vouchers()
    example_10_certificate_and_probabilities()


if __name__ == "__main__":
    main()
