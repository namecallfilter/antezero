"""The adapter between the video pipeline and AnteZero's simulator.

Run from the `cv` folder: `python -m unittest testing.test_antezero_bridge`.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import antezero_sim
except ImportError:
    antezero_sim = None

# What `antezero_bridge.video` made of the first ten minutes of youtube.com/watch?v=wInI1W_EaEw.
RECORDING = REPO_ROOT / "testing" / "antezero_fixtures" / "wInI1W_EaEw_0-600.bal"


def random_game(seed: str, steps: int) -> tuple["antezero_sim.Game", list[str]]:
    rng = random.Random(seed)
    game = antezero_sim.Game(seed, "b_red", 1)
    codes = []
    while len(codes) < steps and not game.is_terminal():
        codes.append(rng.choice(game.legal_codes()))
        game.step(codes[-1])
    return game, codes


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
class TestBal(unittest.TestCase):
    def test_lines_are_the_codes(self) -> None:
        from antezero_bridge import bal

        for code in ("select", "play 21", "discard 1", "use 2 0", "use 0 6", "pick 1 3", "buy_use 0 0",
                     "sell_consumable 1", "sell_joker 4", "sort 0", "sort 1", "move_joker 3 0", "end_shop"):
            self.assertEqual(bal.code(bal.line(code)), code)
        self.assertEqual(bal.line("play 21"), "play 0 2 4")
        self.assertEqual(bal.line("sell_consumable 1"), "sell_cons 1")

    def test_simulator_replays_what_is_written(self) -> None:
        from antezero_bridge import bal

        for seed in ("BRIDGE1", "BRIDGE2", "BRIDGE3"):
            _, codes = random_game(seed, 300)
            bal.check(bal.script(seed, "b_red", 1, codes))
        with self.assertRaises(ValueError):
            bal.check(bal.script("BRIDGE1", "b_red", 1, ["select", "cash_out"]))

    def test_recording_from_video_ends_with_the_deck_on_screen(self) -> None:
        """The video's last frame is its deck view: 47 cards, counted here by suit and by rank."""
        from antezero_bridge import bal

        rows = RECORDING.read_text().splitlines()
        bal.check(RECORDING.read_text())
        _, seed, _, back, _, stake = rows[0].split()
        game = antezero_sim.Game(seed, back, int(stake))
        game.steamodded = True
        for row in rows[1:]:
            game.step(bal.legal_code(game, row))
        state = json.loads(game.state_json())
        cards = [c["card"] for area in ("deck", "hand", "discard") for c in state[area]]
        self.assertEqual(Counter(c[0] for c in cards), {"S": 15, "H": 12, "C": 12, "D": 8})
        self.assertEqual(
            Counter(c[2] for c in cards),
            {"A": 5, "K": 4, "Q": 4, "J": 3, "T": 4, "9": 2, "8": 7, "7": 4, "6": 1, "5": 3, "4": 4, "3": 3, "2": 3},
        )
        self.assertEqual((state["ante"], state["round"], state["dollars"]), (3, 7, 52))


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
class TestReconcile(unittest.TestCase):
    def test_a_misread_play_is_repaired_from_the_screen(self) -> None:
        from antezero_bridge import reconcile

        game = antezero_sim.Game("BRIDGE1", "b_red", 1)
        game.step("select")
        true = "play 3"
        after = game.clone()
        after.step(true)
        screen = reconcile.observe(after, hand=True)
        # one card too many, and an action that is not legal at all
        self.assertEqual(reconcile.repair(game, "play 7", screen), true)
        self.assertEqual(reconcile.repair(game, "play 0", screen), true)

    def test_no_action_fits_a_screen_from_another_game(self) -> None:
        from antezero_bridge import reconcile

        game = antezero_sim.Game("BRIDGE1", "b_red", 1)
        game.step("select")
        screen = reconcile.observe(game) | {"dollars": 999}
        self.assertIsNone(reconcile.repair(game, "play 3", screen))


if __name__ == "__main__":
    unittest.main()
