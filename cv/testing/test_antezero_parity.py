"""AnteZero's simulator against the real game's telemetry logs (`testing/telemetry_data/`).

Run from the `cv` folder: `python -m unittest testing.test_antezero_parity`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import antezero_sim  # noqa: F401
except ImportError:
    antezero_sim = None

LOGS = sorted((REPO_ROOT / "testing" / "telemetry_data").glob("*.jsonl"))


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
@unittest.skipIf(not LOGS, "no telemetry logs in testing/telemetry_data")
class TestAnteZeroParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from antezero_bridge import telemetry

        cls.logs = {path.name: telemetry.load(path) for path in LOGS}

    def test_start_state_matches(self) -> None:
        """From the seed, deck and stake alone the simulator is in the game's state, RNG streams included."""
        from antezero_bridge import telemetry

        for name, entries in self.logs.items():
            with self.subTest(log=name):
                game = telemetry.start_game(entries)
                settled = entries[1] if entries[0]["action"] == "Setup Test Suite" else entries[0]
                self.assertEqual(telemetry.compare(telemetry.dump(game), settled["state"]), [])

    def test_a_card_turned_over_is_a_difference(self) -> None:
        """A card that is the same card but lies the other way up does not pass as equal."""
        import copy

        from antezero_bridge import telemetry

        for name, entries in self.logs.items():
            with self.subTest(log=name):
                sim = telemetry.dump(telemetry.start_game(entries))
                settled = entries[1] if entries[0]["action"] == "Setup Test Suite" else entries[0]
                for area, at in (("draw_pile", 0), ("jokers", 2)):
                    lua = copy.deepcopy(settled["state"])
                    card = lua[area][at]
                    card["facing_back"] = not card["facing_back"]
                    diffs = telemetry.compare(sim, lua)
                    self.assertEqual(len(diffs), 1, diffs)
                    self.assertRegex(diffs[0], rf"^{area}\[{at}\] \S+ facing: sim (back game front|front game back)$")

    def test_a_joker_turned_face_down_is_loaded_from_the_log(self) -> None:
        """Amber Acorn turns the jokers over; a logged state with one face down resyncs as any other."""
        import copy

        from antezero_bridge import resync, telemetry

        for name, entries in self.logs.items():
            with self.subTest(log=name):
                game = telemetry.start_game(entries)
                settled = entries[1] if entries[0]["action"] == "Setup Test Suite" else entries[0]
                lua = copy.deepcopy(settled["state"])
                lua["jokers"][2]["facing_back"] = True
                card = entries[0]["state"]["deck"][0]
                resync.resync(game, lua, card["sort_id"] - card["playing_card"])
                self.assertEqual(telemetry.dump(game)["jokers"][2]["facing"], "back")

    def test_transitions_match_but_for_steamodded(self) -> None:
        """Every logged action is legal, and each state the simulator gets wrong is a known difference of the
        log's Steamodded build from vanilla (and from the Steamodded of today, see `replay.steamodded_cause`)."""
        from antezero_bridge import bal, replay

        for name, entries in self.logs.items():
            with self.subTest(log=name):
                run = replay.replay(entries, resync_on_divergence=True)
                self.assertIsNone(run.error)
                causes = Counter(
                    replay.steamodded_cause(entries, s) for s in run.steps if s.status == "diverged"
                )
                self.assertNotIn(None, causes, [s for s in run.steps if s.status == "diverged"])
                self.assertGreaterEqual(run.count("match") + run.count("transient"), len(entries) - sum(causes.values()))
                for code in run.codes:
                    bal.line(code)

    @unittest.expectedFailure
    def test_open_loop(self) -> None:
        """Given only the actions, the simulator follows the whole game (fails: it does not model the log's old Steamodded build)."""
        from antezero_bridge import replay

        for name, entries in self.logs.items():
            with self.subTest(log=name):
                run = replay.replay(entries)
                self.assertIsNone(run.error)
                self.assertIsNone(run.first_divergence)


if __name__ == "__main__":
    unittest.main()
