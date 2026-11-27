"""The adapter between the video pipeline and AnteZero's simulator.

Run from the `cv` folder: `python -m unittest testing.test_antezero_bridge`.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any
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
# White Stake, Red Deck: the first hand holds a straight (`play 31`, 324 chips) and a flush
# (`play 158`, 308 chips). Either wins the blind, so the screen after it differs in the score alone.
CASE8 = ("CASE8", "play 31", "play 158")
# The first pack of this run holds The Soul, whose joker vanilla rolls a rarity for and Steamodded does not.
SOUL = ("RULES2541", ["skip", "pick 0 96", "pick 1 0", "select"])


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
            bal.check(bal.script(seed, "b_red", 1, codes, steamodded=False))
        with self.assertRaises(ValueError):
            bal.check(bal.script("BRIDGE1", "b_red", 1, ["select", "cash_out"], steamodded=False))

    def test_a_recording_is_replayed_by_the_rules_it_was_played_by(self) -> None:
        from antezero_bridge import bal

        seed, codes = SOUL
        rolled = {}
        for steamodded in (False, True):
            text = bal.script(seed, "b_red", 1, codes, steamodded=steamodded)
            bal.check(text)
            game = bal.start(text)
            self.assertIs(game.steamodded, steamodded)
            for row in text.splitlines()[1:]:
                game.step(bal.legal_code(game, row))
            rolled[steamodded] = "rng.rarity1sou" in game.canonical()
        self.assertEqual(rolled, {False: True, True: False})

    def test_a_vanilla_header_is_the_one_older_recordings_have(self) -> None:
        from antezero_bridge import bal

        self.assertEqual(bal.script("OLD", "b_red", 1, ["select"], steamodded=False), "seed OLD deck b_red stake 1\nselect\n")
        self.assertEqual(bal.script("NEW", "b_red", 1, ["select"], steamodded=True),
                         "seed NEW deck b_red stake 1 rules steamodded\nselect\n")
        with self.assertRaises(ValueError):
            bal.check("seed OLD deck b_red stake 1 rules other\nselect\n")

    def test_a_comment_on_a_line_is_not_part_of_its_action(self) -> None:
        from antezero_bridge import bal

        text = bal.script("BRIDGE1", "b_red", 1, ["select", "play 21"], steamodded=False, comments={1: "unconfirmed"})
        self.assertEqual(text.splitlines()[2], "play 0 2 4  # unconfirmed")
        self.assertEqual(bal.code(text.splitlines()[2]), "play 21")
        bal.check(text)

    def test_recording_from_video_ends_with_the_deck_on_screen(self) -> None:
        """The video's last frame is its deck view: 47 cards, counted here by suit and by rank."""
        from antezero_bridge import bal

        text = RECORDING.read_text()
        bal.check(text)
        game = bal.start(text)
        self.assertTrue(game.steamodded, "the streamer's game ran under Steamodded")
        for row in text.splitlines()[1:]:
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


def _screen(game: "antezero_sim.Game") -> tuple[dict[str, Any], dict[str, Any]]:
    """The HUD numbers and the cards that perfect classifiers read in `game`."""
    from antezero_bridge import reconcile

    view = reconcile.predicted(game)
    return {name: view[name] for name in reconcile.HUD_NUMBERS}, {area: view[area] for area in reconcile.CARDS}


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
class TestSettle(unittest.TestCase):
    def setUp(self) -> None:
        seed, self.straight, self.flush = CASE8
        self.game = antezero_sim.Game(seed, "b_red", 1)
        self.game.step("select")
        self.plays = [code for code in self.game.legal_codes() if code.startswith("play")]
        self.after = self.game.clone()
        self.after.step(self.straight)
        self.hud, self.cards = _screen(self.after)

    def test_the_score_on_screen_tells_two_winning_plays_apart(self) -> None:
        from antezero_bridge import reconcile

        wrong = self.game.clone()
        wrong.step(self.flush)
        differing = {name for name in reconcile.HUD_NUMBERS if _screen(wrong)[0][name] != self.hud[name]}
        self.assertEqual(differing, {"chips"})
        self.assertEqual(_screen(wrong)[1], self.cards)

        settled = reconcile.settle(self.game, self.plays, self.flush, self.hud, lambda state: self.cards)
        self.assertEqual((settled.code, settled.outcome), (self.straight, "repaired"))
        settled = reconcile.settle(self.game, self.plays, self.straight, self.hud, lambda state: self.cards)
        self.assertEqual((settled.code, settled.outcome, settled.rivals), (self.straight, "read", 0))

    def test_an_unread_score_that_would_tell_them_apart_settles_nothing(self) -> None:
        from antezero_bridge import reconcile

        hud = {name: value for name, value in self.hud.items() if name != "chips"}
        for read in (self.straight, self.flush):
            with self.assertRaisesRegex(reconcile.Unresolved, "chips was not read"):
                reconcile.settle(self.game, self.plays, read, hud, lambda state: self.cards)

    def test_a_read_that_does_not_fit_is_not_replaced_by_a_guess(self) -> None:
        """A pair with any three of the other cards scores the same; without the hand on screen nothing says which three."""
        from antezero_bridge import reconcile

        pair = "play 211"  # S_A, D_K and D_T with the pair of twos
        after = self.game.clone()
        after.step(pair)
        hud, cards = _screen(after)
        blind = {area: cards[area] for area in ("jokers", "consumables")}
        settled = reconcile.settle(self.game, self.plays, "play 199", hud, lambda state: cards)
        self.assertEqual((settled.code, settled.outcome), (pair, "repaired"))
        with self.assertRaisesRegex(reconcile.Unresolved, "does not fit the screen"):
            reconcile.settle(self.game, self.plays, "play 0", hud, lambda state: blind)
        settled = reconcile.settle(self.game, self.plays, pair, hud, lambda state: blind)
        self.assertEqual((settled.code, settled.outcome), (pair, "read"))
        self.assertGreater(settled.rivals, 0, "the record rests on the read, and says so")

    def test_a_screen_is_the_simulators_only_in_every_number_read(self) -> None:
        from antezero_bridge import reconcile

        self.assertTrue(reconcile.shows(self.after, self.hud))
        self.assertFalse(reconcile.shows(self.after, self.hud | {"chips": 308}))
        self.assertFalse(reconcile.shows(self.game, self.hud))
        few = dict(list(self.hud.items())[: reconcile.MIN_AGREEMENT - 1])
        self.assertFalse(reconcile.shows(self.after, few))
        self.assertTrue(reconcile.shows(self.after, dict(list(self.hud.items())[: reconcile.MIN_AGREEMENT])))


class SimScreens:
    """The screens of a game played in the simulator: frame `k` is the screen before its click `k`."""

    def __init__(self, seed: str, codes: list[str]) -> None:
        game = antezero_sim.Game(seed, "b_red", 1)
        self.truth = codes
        self.screens = [_screen(game)]
        for code in codes:
            game.step(code)
            self.screens.append(_screen(game))
        # What a test makes the classifiers get wrong: reads of clicks, HUD numbers by frame.
        self.misreads: dict[int, str] = {}
        self.unread: dict[int, str] = {}
        self.wrong: dict[int, dict[str, Any]] = {}

    def clicks(self) -> list[SimpleNamespace]:
        from antezero_bridge import video
        from game.balatro_models import ActionType

        types = {verb: kind for kind, verbs in video.BUTTONS.items() for verb in verbs}
        types |= {"sort 0": ActionType.SORT_RANK, "sort 1": ActionType.SORT_SUIT}
        return [SimpleNamespace(action_type=types.get(code, types[code.split()[0]]), target_frame=at, frame_id=at)
                for at, code in enumerate(self.truth)]

    def hud(self, at: int) -> dict[str, Any]:
        hud = self.screens[at][0] | self.wrong.get(at, {})
        return {name: value for name, value in hud.items() if name != self.unread.get(at)}

    def read(self, game: "antezero_sim.Game", click: SimpleNamespace, legal: list[str]) -> str:
        return self.misreads.get(click.target_frame, self.truth[click.target_frame])

    def cards(self, at: int, state: dict[str, Any]) -> dict[str, Any]:
        return self.screens[at][1]

    def missed_sells(self, at: int, game: "antezero_sim.Game") -> list[str]:
        return []


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
class TestWalk(unittest.TestCase):
    def walk(self, seed: str, screens: SimScreens) -> Any:
        from antezero_bridge import video

        walk = video.Walk()
        video.follow(antezero_sim.Game(seed, "b_red", 1), screens.clicks(), screens, walk)
        return walk

    def test_true_screens_are_walked_to_the_last_click(self) -> None:
        for seed in ("BRIDGE1", "BRIDGE2", "BRIDGE3"):
            _, codes = random_game(seed, 300)
            walk = self.walk(seed, SimScreens(seed, codes))
            # the last click has no screen after it, so it is recorded only if its button has one action
            self.assertEqual(walk.codes[: len(codes) - 1], codes[:-1])
            self.assertGreater(len(walk.codes), 5)
            self.assertNotIn("repaired", walk.outcomes)

    def test_a_misread_play_is_repaired_by_the_score(self) -> None:
        seed, straight, flush = CASE8
        screens = SimScreens(seed, ["select", straight, "cash_out"])
        screens.misreads[1] = flush
        walk = self.walk(seed, screens)
        self.assertEqual(walk.codes, ["select", straight, "cash_out"])
        self.assertEqual(walk.outcomes["repaired"], 1)
        self.assertIsNone(walk.stopped)

    def test_the_walk_stops_before_a_play_the_screen_cannot_settle(self) -> None:
        seed, straight, flush = CASE8
        for read in (straight, flush):
            screens = SimScreens(seed, ["select", straight, "cash_out"])
            screens.misreads[1] = read
            screens.unread[2] = "chips"
            walk = self.walk(seed, screens)
            self.assertEqual(walk.codes, ["select"])
            self.assertIn("chips was not read", walk.stopped)

    def test_the_walk_stops_at_a_screen_that_is_not_the_simulators(self) -> None:
        seed, straight, _ = CASE8
        screens = SimScreens(seed, ["select", straight, "cash_out", "end_shop", "select"])
        screens.wrong[3] = {"dollars": screens.screens[3][0]["dollars"] + 1}
        walk = self.walk(seed, screens)
        self.assertEqual(walk.codes, ["select", straight])
        self.assertIn("is not the simulator's game after cash_out", walk.stopped)

        screens = SimScreens("OTHER", ["select"])
        screens.wrong[0] = {"dollars": 99}
        walk = self.walk("OTHER", screens)
        self.assertEqual(walk.codes, [])
        self.assertIn("the first screen is not a new run", walk.stopped)


if __name__ == "__main__":
    unittest.main()
