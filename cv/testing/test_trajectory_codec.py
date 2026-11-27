"""Round-trip and sparse-delta tests for `game/trajectory_codec.py`."""

from __future__ import annotations

import copy
import json
import unittest

from game.balatro_models import (
    Action,
    ActionType,
    BlindName,
    BoosterPack,
    BoosterSize,
    BoosterType,
    Deck,
    Edition,
    Enhancement,
    HandCard,
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
from game.gamestate import (
    _TRACKED_STATE_FIELDS,
    GameState,
    RunTrajectory,
    on_action,
)
from game.trajectory_codec import (
    rebuild_gamestate,
    rebuild_trajectory,
    serialize_trajectory,
)


def _assert_states_equivalent(tc: unittest.TestCase, expected: GameState, actual: GameState) -> None:
    """Asserts full field-by-field equivalence between two GameState instances."""
    tc.assertEqual(actual.step_idx, expected.step_idx)
    tc.assertEqual(actual.frame_idx, expected.frame_idx)
    tc.assertAlmostEqual(actual.timestamp, expected.timestamp, places=9)

    for attr in _TRACKED_STATE_FIELDS:
        tc.assertEqual(
            getattr(actual, attr),
            getattr(expected, attr),
            msg=f"Mismatch on GameState.{attr} at step {expected.step_idx}",
        )

    tc.assertEqual(actual.inventory.joker_slots, expected.inventory.joker_slots)
    tc.assertEqual(actual.inventory.consumable_slots, expected.inventory.consumable_slots)
    tc.assertEqual(actual.inventory.jokers, expected.inventory.jokers)
    for ej, aj in zip(expected.inventory.jokers, actual.inventory.jokers):
        tc.assertEqual(aj.id, ej.id)
        tc.assertEqual(aj.state, ej.state)
    tc.assertEqual(actual.inventory.consumables, expected.inventory.consumables)

    tc.assertEqual(actual.hand.max_size, expected.hand.max_size)
    tc.assertEqual(actual.hand.cards, expected.hand.cards)
    for ehc, ahc in zip(expected.hand.cards, actual.hand.cards):
        tc.assertEqual(ahc.card.id, ehc.card.id)
        tc.assertEqual(ahc.card.debuffed, ehc.card.debuffed)
        tc.assertEqual(ahc.card.destroyed, ehc.card.destroyed)
        tc.assertEqual(ahc._real_card, ehc._real_card)


class TestTrajectoryCodec(unittest.TestCase):
    """Verifies sparse delta serialization and full GameState reconstruction."""

    def _step(
        self,
        state: GameState,
        traj: RunTrajectory,
        action: Action,
        mutate=None,
    ) -> None:
        state.frame_idx = action.frame_idx
        state.timestamp = action.timestamp
        if mutate is not None:
            mutate(state)
        on_action(state, action)
        traj.record_step(state, action)

    def test_sparse_delta_omits_unchanged_fields_and_captures_expressive_details(self):
        state = GameState(
            seed="SPARSE1",
            deck=Deck.BLUE_DECK,
            stake=Stake.GOLD,
            ante=1,
            round=1,
            current_blind=BlindName.SMALL,
            location=Location.SELECT,
            apply_deck_stake_on_init=True,
        )
        traj = RunTrajectory()
        traj.record_step(state, action=None)

        # Step 1: Select Blind
        self._step(state, traj, Action(type=ActionType.SELECT_BLIND, frame_idx=60, timestamp=1.0))

        # Step 2: Populate hand & play a pair; reroll_cost does NOT change here.
        c1 = PlayingCard(
            rank=Rank.ACE,
            suit=Suit.SPADES,
            enhancement=Enhancement.MULT,
            seal=Seal.RED,
            edition=Edition.FOIL,
            id=101,
        )
        c2 = PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS, id=102)
        c3 = PlayingCard(rank=Rank.KING, suit=Suit.CLUBS, id=103)
        hand_cards = [
            HandCard(card=c1, selected=True, slot_uid=1),
            HandCard(card=c2, selected=True, slot_uid=2),
            HandCard(card=c3, selected=False, slot_uid=3),
        ]

        def _set_hand_and_score(st: GameState) -> None:
            st.hand.cards = list(hand_cards)
            st.current_chips = 180

        self._step(
            state,
            traj,
            Action(
                type=ActionType.PLAY_HAND,
                frame_idx=180,
                timestamp=3.0,
                selected_cards=[c1, c2],
            ),
            mutate=_set_hand_and_score,
        )

        # Step 3: End round -> Cash out -> Shop with sticker/edition joker
        self._step(
            state,
            traj,
            Action(type=ActionType.END_ROUND, frame_idx=240, timestamp=4.0),
            mutate=lambda st: setattr(st, "current_chips", 350),
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.CASH_OUT, frame_idx=300, timestamp=5.0),
        )

        sticker_joker = Joker(
            name=JokerName.BLUEPRINT,
            edition=Edition.POLYCHROME,
            eternal=True,
            perishable=True,
            perish_tally=3,
            rental=True,
            extra_value=4,
            state={"copied": "Joker"},
            id=77,
        )
        plain_joker = Joker(name=JokerName.JOKER, id=78)
        self._step(
            state,
            traj,
            Action(type=ActionType.BUY_CARD, frame_idx=360, timestamp=6.0, target_idx=0),
            mutate=lambda st: setattr(st, "shop_cards", [sticker_joker, plain_joker]),
        )

        payload = json.loads(json.dumps(serialize_trajectory(state, traj)))
        steps = payload["steps"]

        # Step 2 (PLAY_HAND): action.selected_cards lists played cards (with edition/seal/enhancement),
        # hand_diff lists remaining cards in hand, and unchanged reroll_cost/ante are NOT written.
        play_step = steps[2]
        self.assertIn("hand_diff", play_step)
        self.assertEqual(len(play_step["hand_diff"]), 1)
        self.assertEqual(len(play_step["action"]["selected_cards"]), 2)
        self.assertEqual(play_step["action"]["selected_cards"][0]["edition"], "foil")
        self.assertEqual(play_step["action"]["selected_cards"][0]["seal"], "red")
        self.assertEqual(play_step["action"]["selected_cards"][0]["enhancement"], "mult")
        self.assertNotIn("reroll_cost", play_step.get("field_diffs", {}))
        self.assertNotIn("ante", play_step)

        # Step 5 (BUY_CARD): jokers_diff lists sticker/edition fields on sticker_joker, omits defaults on plain_joker.
        buy_step = steps[5]
        self.assertIn("jokers_diff", buy_step)
        j0 = buy_step["jokers_diff"][0]
        self.assertEqual(j0["name"], "Blueprint")
        self.assertEqual(j0["edition"], "polychrome")
        self.assertTrue(j0["eternal"])
        self.assertTrue(j0["perishable"])
        self.assertEqual(j0["perish_tally"], 3)
        self.assertTrue(j0["rental"])
        self.assertEqual(j0["extra_value"], 4)
        self.assertEqual(j0["state"], {"copied": "Joker"})

        # Verify full round-trip equivalence across every step
        rebuilt_traj = rebuild_trajectory(payload)
        self.assertEqual(len(rebuilt_traj.deltas), len(traj.deltas))
        for i in range(len(traj.deltas)):
            _assert_states_equivalent(self, traj.get_state(i), rebuilt_traj.get_state(i))

        rebuilt_final = rebuild_gamestate(payload)
        _assert_states_equivalent(self, state, rebuilt_final)

    def test_full_multi_ante_trajectory_roundtrip_equivalence(self):
        state = GameState(
            seed="ROUNDTRIP",
            deck=Deck.PLASMA_DECK,
            stake=Stake.PURPLE,
            ante=1,
            round=1,
            current_blind=BlindName.SMALL,
            location=Location.SELECT,
            apply_deck_stake_on_init=True,
        )
        traj = RunTrajectory(keyframe_interval=4)
        traj.record_step(state, action=None)

        # Skip small blind with a tag
        state.small_blind_tag = SkipTag.COUPON_TAG
        self._step(state, traj, Action(type=ActionType.SKIP_BLIND, frame_idx=30, timestamp=0.5))

        # Select big blind -> discard -> play -> end round -> cash out
        self._step(state, traj, Action(type=ActionType.SELECT_BLIND, frame_idx=60, timestamp=1.0))

        d_card = PlayingCard(rank=Rank.TWO, suit=Suit.DIAMONDS, id=1)
        p_card = PlayingCard(rank=Rank.QUEEN, suit=Suit.HEARTS, enhancement=Enhancement.GLASS, id=2)
        fd_card = HandCard(
            card=PlayingCard(rank=Rank.NONE, suit=Suit.NONE),
            facing_back=True,
            slot_uid=9,
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.DISCARD, frame_idx=90, timestamp=1.5, selected_cards=[d_card]),
            mutate=lambda st: setattr(
                st.hand,
                "cards",
                [HandCard(card=d_card, selected=True, slot_uid=1), HandCard(card=p_card, slot_uid=2), fd_card],
            ),
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.PLAY_HAND, frame_idx=120, timestamp=2.0, selected_cards=[p_card]),
            mutate=lambda st: setattr(st, "current_chips", 1000),
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.END_ROUND, frame_idx=150, timestamp=2.5),
            mutate=lambda st: setattr(st, "boss_blind", BlindName.THE_HOOK),
        )
        self._step(state, traj, Action(type=ActionType.CASH_OUT, frame_idx=180, timestamp=3.0))

        # Shop: buy consumable, reroll, redeem voucher, open pack, use in pack, next round
        tarot = Tarot(name=TarotName.FOOL, edition=Edition.NEGATIVE)
        planet = Planet(name=PlanetName.JUPITER)
        spectral = Spectral(name=SpectralName.ECTOPLASM)
        voucher = Voucher(name=VoucherName.OVERSTOCK)
        pack = BoosterPack(
            booster_type=BoosterType.ARCANA,
            size=BoosterSize.JUMBO,
            cost=6,
            choose=1,
            extra=5,
            couponed=True,
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.BUY_CARD, frame_idx=210, timestamp=3.5, target_idx=0),
            mutate=lambda st: (
                setattr(st, "shop_cards", [tarot]),
                setattr(st, "shop_vouchers", [voucher]),
                setattr(st, "shop_packs", [pack]),
            ),
        )
        self._step(state, traj, Action(type=ActionType.REROLL_SHOP, frame_idx=240, timestamp=4.0))
        self._step(state, traj, Action(type=ActionType.REDEEM_VOUCHER, frame_idx=270, timestamp=4.5, target_idx=0))
        self._step(state, traj, Action(type=ActionType.OPEN_PACK, frame_idx=300, timestamp=5.0, target_idx=0))
        self._step(
            state,
            traj,
            Action(type=ActionType.USE_IN_PACK, frame_idx=330, timestamp=5.5, target_idx=0),
            mutate=lambda st: setattr(st, "pack_cards", [planet, spectral]),
        )
        self._step(state, traj, Action(type=ActionType.USE_CONSUMABLE, frame_idx=360, timestamp=6.0, target_idx=0))
        self._step(state, traj, Action(type=ActionType.NEXT_ROUND, frame_idx=390, timestamp=6.5))

        traj.impute_retroactive_boss_blinds()

        json_str = json.dumps(serialize_trajectory(state, traj))
        reloaded = json.loads(json_str)

        rebuilt_traj = rebuild_trajectory(reloaded, keyframe_interval=4)
        self.assertEqual(len(rebuilt_traj.deltas), len(traj.deltas))
        for i in range(len(traj.deltas)):
            _assert_states_equivalent(self, traj.get_state(i), rebuilt_traj.get_state(i))

        # Check step_idx selection on rebuild_gamestate
        _assert_states_equivalent(self, traj.get_state(3), rebuild_gamestate(reloaded, step_idx=3))
        _assert_states_equivalent(self, state, rebuild_gamestate(reloaded))

    def test_incremental_collection_and_mapping_diffs_roundtrip(self):
        state = GameState(
            seed="INCRDIFF",
            deck=Deck.RED_DECK,
            stake=Stake.WHITE,
            ante=1,
            round=1,
            current_blind=BlindName.SMALL,
            location=Location.SELECT,
            apply_deck_stake_on_init=True,
        )
        traj = RunTrajectory(keyframe_interval=3)
        traj.record_step(state, action=None)

        # Pick real deck cards with unique IDs from the initialized 52-card deck
        c0 = copy.deepcopy(state.full_deck[0])
        c1 = copy.deepcopy(state.full_deck[1])
        c2 = copy.deepcopy(state.full_deck[2])
        c3 = copy.deepcopy(state.full_deck[3])

        # Step 1: Select Blind
        self._step(state, traj, Action(type=ActionType.SELECT_BLIND, frame_idx=30, timestamp=0.5))

        # Step 2: Discard c0 (checks incremental discard_pile_diff)
        self._step(
            state,
            traj,
            Action(type=ActionType.DISCARD, frame_idx=60, timestamp=1.0, selected_cards=[c0]),
            mutate=lambda st: setattr(
                st.hand,
                "cards",
                [
                    HandCard(card=copy.deepcopy(c0), selected=True, slot_uid=1),
                    HandCard(card=copy.deepcopy(c1), slot_uid=2),
                    HandCard(card=copy.deepcopy(c2), slot_uid=3),
                    HandCard(card=copy.deepcopy(c3), slot_uid=4),
                ],
            ),
        )

        # Step 3: Play c1 (first append to played_cards_this_round, played_cards_this_ante,
        # hand_play_counts, discard_pile)
        def _select_c1_and_score(st: GameState) -> None:
            for hc in st.hand.cards:
                hc.selected = hc.slot_uid == 2
            st.current_chips = 100

        self._step(
            state,
            traj,
            Action(type=ActionType.PLAY_HAND, frame_idx=90, timestamp=1.5, selected_cards=[c1]),
            mutate=_select_c1_and_score,
        )

        # Step 4: Play c2 (pure append on top of existing played_cards & discard_pile)
        def _select_c2_and_score(st: GameState) -> None:
            for hc in st.hand.cards:
                hc.selected = hc.slot_uid == 3
            st.current_chips = 200

        self._step(
            state,
            traj,
            Action(type=ActionType.PLAY_HAND, frame_idx=120, timestamp=2.0, selected_cards=[c2]),
            mutate=_select_c2_and_score,
        )

        # Step 5: End round -> clears discard_pile,
        # enhances c3, removes one card, adds one card, and levels up a poker hand.
        c3_enhanced = copy.deepcopy(c3)
        c3_enhanced.enhancement = Enhancement.GLASS
        removed_id = state.full_deck[10].id
        extra_card = PlayingCard(rank=Rank.ACE, suit=Suit.SPADES, enhancement=Enhancement.LUCKY, id=99)

        def _mutate_deck_and_levels(st: GameState) -> None:
            st.hand_levels[PokerHand.FLUSH] = st.hand_levels.get(PokerHand.FLUSH, 1) + 1
            for idx, dc in enumerate(st.full_deck):
                if dc.id == c3_enhanced.id:
                    st.full_deck[idx] = copy.deepcopy(c3_enhanced)
            st.full_deck = [dc for dc in st.full_deck if dc.id != removed_id]
            st.full_deck.append(copy.deepcopy(extra_card))

        self._step(
            state,
            traj,
            Action(type=ActionType.END_ROUND, frame_idx=150, timestamp=2.5),
            mutate=_mutate_deck_and_levels,
        )

        # Step 6: Cash out to shop
        self._step(state, traj, Action(type=ActionType.CASH_OUT, frame_idx=180, timestamp=3.0))

        # Step 7: Enter Boss Blind (The Club -> debuffs all Clubs in full_deck and clears played_cards_this_round)
        expected_club_ids = [dc.id for dc in state.full_deck if dc.suit == Suit.CLUBS]
        non_club_id = next(dc.id for dc in state.full_deck if dc.suit != Suit.CLUBS)

        def _enter_club_boss(st: GameState) -> None:
            st.location = Location.SELECT
            st.pending_blind_advance = False
            st.current_blind = BlindName.THE_CLUB
            st.boss_blind = BlindName.THE_CLUB
            for dc in st.full_deck:
                if dc.id == non_club_id:
                    dc.bonus_chips += 5

        self._step(
            state,
            traj,
            Action(type=ActionType.SELECT_BLIND, frame_idx=210, timestamp=3.5),
            mutate=_enter_club_boss,
        )

        # Step 8: Defeat Boss Blind -> clears debuffs back to False -> undebuffed_ids & played_cards_this_ante=[]
        def _defeat_club_boss(st: GameState) -> None:
            st.current_chips = 600
            for dc in st.full_deck:
                if dc.id == non_club_id:
                    dc.bonus_chips += 5

        self._step(
            state,
            traj,
            Action(type=ActionType.END_ROUND, frame_idx=240, timestamp=4.0),
            mutate=_defeat_club_boss,
        )

        payload = json.loads(json.dumps(serialize_trajectory(state, traj)))
        steps = payload["steps"]

        # Step 4 (second Play Hand): played_cards_diff, discard_pile_diff, played_cards_this_ante
        # should only serialize newly added card(s), NOT full snapshots, and hand_play_counts_diff
        # should only serialize the changed hand key.
        step4 = steps[4]
        self.assertIsInstance(step4["played_cards_diff"], dict)
        self.assertEqual(len(step4["played_cards_diff"]["added"]), 1)
        self.assertIsInstance(step4["discard_pile_diff"], dict)
        self.assertEqual(len(step4["discard_pile_diff"]["added"]), 1)
        ante_diff = step4["field_diffs"]["played_cards_this_ante"]
        self.assertIsInstance(ante_diff, dict)
        self.assertEqual(len(ante_diff["added"]), 1)
        self.assertIsInstance(step4["hand_play_counts_diff"], dict)
        self.assertEqual(len(step4["hand_play_counts_diff"]), 1)

        # Step 5 (End Round): full_deck_diff should contain compact incremental diff keys,
        # hand_levels_diff should only have Flush, and cleared discard_pile should be [].
        step5 = steps[5]
        deck_diff5 = step5["full_deck_diff"]
        self.assertIsInstance(deck_diff5, dict)
        self.assertEqual(deck_diff5["removed_ids"], [removed_id])
        self.assertEqual(len(deck_diff5["updated"]), 1)
        self.assertEqual(deck_diff5["updated"][0]["id"], c3_enhanced.id)
        self.assertEqual(deck_diff5["updated"][0]["enhancement"], "glass")
        self.assertEqual(len(deck_diff5["added"]), 1)
        self.assertEqual(deck_diff5["added"][0]["id"], 99)
        self.assertEqual(step5["hand_levels_diff"], {PokerHand.FLUSH.value: 2})
        self.assertEqual(step5["discard_pile_diff"], [])

        # Step 7 & 8: debuffed_ids + cleared played_cards on Boss entry,
        # undebuffed_ids + cleared played_cards_this_ante on Boss clear
        self.assertEqual(steps[7]["full_deck_diff"]["debuffed_ids"], expected_club_ids)
        self.assertEqual(len(steps[7]["full_deck_diff"]["updated"]), 1)
        self.assertEqual(steps[7]["played_cards_diff"], [])
        self.assertEqual(steps[8]["full_deck_diff"]["undebuffed_ids"], expected_club_ids)
        self.assertEqual(len(steps[8]["full_deck_diff"]["updated"]), 1)
        self.assertEqual(steps[8]["field_diffs"]["played_cards_this_ante"], [])

        # Full round-trip equivalence at every single step
        rebuilt_traj = rebuild_trajectory(payload, keyframe_interval=3)
        self.assertEqual(len(rebuilt_traj.deltas), len(traj.deltas))
        for i in range(len(traj.deltas)):
            _assert_states_equivalent(self, traj.get_state(i), rebuilt_traj.get_state(i))
        _assert_states_equivalent(self, state, rebuild_gamestate(payload))

    def test_telemetry_to_trajectory_run_splitting_and_roundtrip(self):
        from pathlib import Path
        from tools.telemetry_to_trajectory import (
            load_telemetry_entries,
            split_telemetry_runs,
            select_run,
            build_trajectory_from_telemetry,
        )

        # 1. Synthetic multi-run + Clearance Sale pack pricing (no double discount)
        base_state = {
            "seed": "TEST1234",
            "deck_name": "Red Deck",
            "stake": 1,
            "location": "shop",
            "ante": 1,
            "round": 1,
            "money": 20,
            "vouchers": ["v_clearance_sale"],
            "shop_packs": [{"name": "Jumbo Standard Pack", "key": "p_standard_jumbo_1", "choose": 1, "extra": 5, "cost": 4}],
        }
        e1 = {"run_id": "r1", "step": 1, "timestamp": 10.0, "action": "Open Pack", "action_metadata": {"target_idx": 0}, "state": base_state}
        e2 = {"run_id": "r1", "step": 2, "timestamp": 11.0, "action": "Skip Pack", "action_metadata": {}, "state": {**base_state, "location": "pack", "money": 16, "shop_packs": []}}
        e3 = {"run_id": "r2", "step": 1, "timestamp": 20.0, "action": "Select Blind", "action_metadata": {}, "state": {**base_state, "location": "select", "ante": 1, "round": 0, "money": 4, "vouchers": []}}
        runs = split_telemetry_runs([e1, e2, e3])
        self.assertEqual(len(runs), 2)
        st, traj, warns = build_trajectory_from_telemetry(runs[0], reconcile_truth=False)
        self.assertEqual(warns, [])
        self.assertEqual(st.money, 16)
        from game.trajectory_codec import SCHEMA_VERSION
        payload = serialize_trajectory(st, traj)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        _assert_states_equivalent(self, st, rebuild_gamestate(payload))

        # 2. Local telemetry run A8CQP141 if present
        local_p = Path(__file__).resolve().parent.parent / "assets" / "local" / "run_telemetry.jsonl"
        if local_p.exists() and local_p.stat().st_size > 0:
            entries = load_telemetry_entries(local_p)
            local_runs = split_telemetry_runs(entries)
            chosen = select_run(local_runs, seed="A8CQP141")
            st_l, traj_l, warns_l = build_trajectory_from_telemetry(chosen, reconcile_truth=False)
            self.assertEqual(warns_l, [])
            p_l = serialize_trajectory(st_l, traj_l)
            self.assertEqual(p_l["num_steps"], 15)
            rebuilt_l = rebuild_trajectory(p_l)
            _assert_states_equivalent(self, st_l, rebuilt_l.get_state(len(rebuilt_l.deltas) - 1))


if __name__ == "__main__":
    unittest.main()
