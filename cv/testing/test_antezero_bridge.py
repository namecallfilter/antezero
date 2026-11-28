"""The adapter between the video pipeline and AnteZero's simulator.

Run from the `cv` folder: `python -m unittest testing.test_antezero_bridge`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
# Its first shop, with $10: Runner for $5 and Splash for $3.
CASE8_SHOP = ["select", "play 31", "cash_out"]
# White Stake, Red Deck: after this the shop has a Buffoon Pack, which deals no hand, and an
# Arcana Pack, which does, at one price.
PACKS = ("PACK63", ["select", "play 31", "cash_out"])
# Jokers by their injection codes, and what they sell for.
JOKER, CAVENDISH, BLUEPRINT = ("inject_joker 0 0 0", 1), ("inject_joker 60 0 0", 2), ("inject_joker 122 0 0", 5)
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


def played(seed: str, codes: list[str]) -> "antezero_sim.Game":
    game = antezero_sim.Game(seed, "b_red", 1)
    for code in codes:
        game.inject(code) if code.startswith("inject") else game.step(code)
    return game


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
            self.assertEqual(game.canonical(), antezero_sim.Game.from_recording(text, len(codes)).canonical())
        self.assertEqual(rolled, {False: True, True: False})

    def test_a_vanilla_header_is_the_one_older_recordings_have(self) -> None:
        from antezero_bridge import bal

        self.assertEqual(bal.script("OLD", "b_red", 1, ["select"], steamodded=False), "seed OLD deck b_red stake 1\nselect\n")
        self.assertEqual(bal.script("NEW", "b_red", 1, ["select"], steamodded=True),
                         "seed NEW deck b_red stake 1 rules steamodded\nselect\n")

    def test_a_header_is_read_as_the_simulator_reads_it(self) -> None:
        from antezero_bridge import bal

        commented = "# a note\n\nseed NEW deck b_red stake 1 rules steamodded  # from a video\nselect\n"
        bal.check(commented)
        self.assertTrue(bal.start(commented).steamodded)
        self.assertFalse(bal.start("seed OLD deck b_red stake 1  # no rules named\n").steamodded)
        for header in ("seed OLD deck b_red stake 1 rules other", "seed OLD deck b_red stake 1 rules", "select"):
            for read in (bal.check, bal.start):
                with self.assertRaises(ValueError, msg=(header, read.__name__)):
                    read(header + "\nselect\n")

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
        # before the clip's last click, a $5 reroll that no later screen confirms
        self.assertEqual((state["ante"], state["round"], state["dollars"]), (3, 7, 57))


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


@dataclass
class SimScreen:
    """A screen of a game in the simulator, as perfect classifiers read it (`reconcile.Screen`)."""

    hud: dict[str, Any]
    seen: dict[str, Any]
    # The joker bar, for a screen on which a sale's click was missed.
    bar: Counter | None = None

    @classmethod
    def of(cls, game: "antezero_sim.Game") -> "SimScreen":
        from antezero_bridge import reconcile

        view = reconcile.predicted(game)
        return cls({name: view[name] for name in reconcile.HUD_NUMBERS}, {area: view[area] for area in reconcile.CARDS})

    def cards(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        return self.seen

    def sales(self, game: "antezero_sim.Game") -> list[str]:
        from antezero_bridge import video

        return [] if self.bar is None else video.missed_sells(game, self.hud.get("dollars"), lambda: self.bar)


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
class TestSettle(unittest.TestCase):
    def setUp(self) -> None:
        seed, self.straight, self.flush = CASE8
        self.game = played(seed, ["select"])
        self.plays = [code for code in self.game.legal_codes() if code.startswith("play")]
        self.won = played(seed, ["select", self.straight])
        self.screen = SimScreen.of(self.won)

    def test_the_score_on_screen_tells_two_winning_plays_apart(self) -> None:
        from antezero_bridge import reconcile

        wrong = SimScreen.of(played(CASE8[0], ["select", self.flush]))
        self.assertEqual({name for name in reconcile.HUD_NUMBERS if wrong.hud[name] != self.screen.hud[name]}, {"chips"})
        self.assertEqual(wrong.seen, self.screen.seen)

        settled = reconcile.settle(self.game, self.plays, self.flush, self.screen, ("cash_out",))
        self.assertEqual((settled.code, settled.outcome), (self.straight, "repaired"))
        settled = reconcile.settle(self.game, self.plays, self.straight, self.screen, ("cash_out",))
        self.assertEqual((settled.code, settled.outcome, settled.rivals), (self.straight, "read", 0))

    def test_an_unread_score_that_would_tell_them_apart_settles_nothing(self) -> None:
        from antezero_bridge import reconcile

        del self.screen.hud["chips"]
        for read in (self.straight, self.flush):
            with self.assertRaisesRegex(reconcile.Unresolved, "chips was not read"):
                reconcile.settle(self.game, self.plays, read, self.screen, ("cash_out",))

    def test_a_read_that_does_not_fit_is_not_replaced_by_a_guess(self) -> None:
        """A pair with any three of the other cards scores the same; without the hand on screen nothing says which three."""
        from antezero_bridge import reconcile

        pair = "play 211"  # S_A, D_K and D_T with the pair of twos
        screen = SimScreen.of(played(CASE8[0], ["select", pair]))
        settled = reconcile.settle(self.game, self.plays, "play 199", screen, ("play",))
        self.assertEqual((settled.code, settled.outcome), (pair, "repaired"))

        del screen.seen["hand"]
        for unread in (None, "play 0"):
            with self.assertRaisesRegex(reconcile.Unresolved, "does not fit the screen"):
                reconcile.settle(self.game, self.plays, unread, screen, ("play",))
        settled = reconcile.settle(self.game, self.plays, pair, screen, ("play",))
        self.assertEqual((settled.code, settled.outcome), (pair, "read"))
        self.assertGreater(settled.rivals, 0, "the record rests on the read, and says so")

    def test_cards_read_wrong_do_not_replace_a_read_that_fits_the_hud(self) -> None:
        from antezero_bridge import reconcile

        pair, other = "play 211", "play 199"
        screen = SimScreen.of(played(CASE8[0], ["select", pair]))
        # the detector sees most of the hand another play would leave, and a card that is in neither
        screen.seen["hand"] = sorted([*SimScreen.of(played(CASE8[0], ["select", other])).seen["hand"][:-1], "X_X"])
        settled = reconcile.settle(self.game, self.plays, pair, screen, ("play",))
        self.assertEqual((settled.code, settled.outcome), (pair, "read"))
        self.assertGreater(settled.rivals, 0)
        with self.assertRaises(reconcile.Unresolved):
            reconcile.settle(self.game, self.plays, "play 0", screen, ("play",))

    def test_a_click_that_did_nothing_is_told_by_the_next_click(self) -> None:
        from antezero_bridge import reconcile

        shop = played(CASE8[0], CASE8_SHOP)
        left = played(CASE8[0], [*CASE8_SHOP, "end_shop"])
        self.assertEqual(SimScreen.of(shop).hud, SimScreen.of(left).hud)
        # still in the shop: its Next Round button is clicked again
        settled = reconcile.settle(shop, ["end_shop"], "end_shop", SimScreen.of(shop), ("end_shop",))
        self.assertEqual((settled.code, settled.outcome), (None, "rejected"))
        settled = reconcile.settle(shop, ["end_shop"], "end_shop", SimScreen.of(left), ("select", "skip"))
        self.assertEqual((settled.code, settled.outcome, settled.rivals), ("end_shop", "forced", 0))
        with self.assertRaisesRegex(reconcile.Unresolved, "legal after no result"):
            reconcile.settle(shop, ["end_shop"], "end_shop", SimScreen.of(left), ("cash_out",))

    def test_a_click_no_screen_tells_from_nothing_is_marked(self) -> None:
        from antezero_bridge import reconcile

        start = antezero_sim.Game(CASE8[0], "b_red", 1)
        for screen in (SimScreen.of(start), SimScreen.of(played(CASE8[0], ["skip"]))):
            settled = reconcile.settle(start, ["skip"], "skip", screen, ("select",))
            self.assertEqual((settled.code, settled.outcome, settled.rivals), ("skip", "forced", 1))

    def test_a_pack_that_deals_no_hand_is_told_from_one_that_does(self) -> None:
        from antezero_bridge import reconcile

        seed, codes = PACKS
        shop = played(seed, codes)
        opened = {code: played(seed, [*codes, code]) for code in ("open 0", "open 1")}
        self.assertEqual({code: bool(SimScreen.of(game).seen["hand"]) for code, game in opened.items()},
                         {"open 0": False, "open 1": True})
        self.assertEqual(SimScreen.of(opened["open 0"]).hud, SimScreen.of(opened["open 1"]).hud)
        for true, wrong in (("open 0", "open 1"), ("open 1", "open 0")):
            screen = SimScreen.of(opened[true])
            settled = reconcile.settle(shop, ["open 0", "open 1"], wrong, screen, ("skip_pack",))
            self.assertEqual((settled.code, settled.outcome), (true, "repaired"))
            settled = reconcile.settle(shop, ["open 0", "open 1"], true, screen, ("skip_pack",))
            self.assertEqual((settled.code, settled.outcome, settled.rivals), (true, "read", 0))

    def test_a_sale_missed_beside_a_purchase_is_not_taken_for_another_action(self) -> None:
        from antezero_bridge import reconcile

        buys = ["buy 0", "buy 1"]
        # Runner bought and Blueprint sold, both for $5: the money is where it was
        shop = played(CASE8[0], [*CASE8_SHOP, BLUEPRINT[0]])
        after = played(CASE8[0], [*CASE8_SHOP, BLUEPRINT[0], "buy 0", "sell_joker 0"])
        screen = SimScreen.of(after)
        self.assertEqual(screen.hud, SimScreen.of(shop).hud)
        screen.bar = Counter(screen.seen["jokers"])
        settled = reconcile.settle(shop, buys, "buy 0", screen, ("end_shop",))
        self.assertEqual((settled.code, settled.outcome, settled.sales), ("buy 0", "read", ("sell_joker 0",)))

        # Runner bought for $5 and Cavendish sold for $2: the money of buying Splash for $3
        shop = played(CASE8[0], [*CASE8_SHOP, CAVENDISH[0]])
        after = played(CASE8[0], [*CASE8_SHOP, CAVENDISH[0], "buy 0", "sell_joker 0"])
        screen = SimScreen.of(after)
        self.assertEqual(screen.hud, SimScreen.of(played(CASE8[0], [*CASE8_SHOP, CAVENDISH[0], "buy 1"])).hud)
        screen.bar = Counter(screen.seen["jokers"])
        settled = reconcile.settle(shop, buys, "buy 0", screen, ("end_shop",))
        self.assertEqual((settled.code, settled.outcome, settled.sales), ("buy 0", "read", ("sell_joker 0",)))

    def test_a_screen_is_the_simulators_only_in_every_number_read(self) -> None:
        from antezero_bridge import reconcile

        hud = self.screen.hud
        self.assertTrue(reconcile.shows(self.won, hud))
        self.assertFalse(reconcile.shows(self.won, hud | {"chips": 308}))
        self.assertFalse(reconcile.shows(self.game, hud))
        self.assertFalse(reconcile.shows(self.won, dict(list(hud.items())[: reconcile.MIN_AGREEMENT - 1])))
        self.assertTrue(reconcile.shows(self.won, dict(list(hud.items())[: reconcile.MIN_AGREEMENT])))

    def test_a_large_score_is_compared_as_the_game_prints_it(self) -> None:
        from antezero_bridge import reconcile

        for chips, steamodded, vanilla in ((123_400_000_000, 1.23e11, 1.234e11), (899_400_000_000, 8.99e11, 8.994e11),
                                           (99_999_999_999_999, 1.00e14, 1.000e14)):
            for shown in (steamodded, vanilla):
                self.assertEqual(reconcile.agreement({"chips": chips}, {"chips": shown}), 1, (chips, shown))
        self.assertEqual(reconcile.agreement({"chips": 899_400_000_000}, {"chips": 9.00e11}), 0)
        self.assertEqual(reconcile.agreement({"chips": 99_999_999_999}, {"chips": 1e11}), 0)


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
class TestMissedSells(unittest.TestCase):
    def test_a_joker_gone_from_the_bar_was_sold_if_the_money_agrees(self) -> None:
        from antezero_bridge import video

        game = played(CASE8[0], [*CASE8_SHOP, JOKER[0], CAVENDISH[0]])
        money = game.dollars + CAVENDISH[1]
        self.assertEqual(video.missed_sells(game, money, lambda: Counter(["j_joker"])), ["sell_joker 1"])
        self.assertEqual(video.missed_sells(game, money + 1, lambda: Counter(["j_joker"])), [])
        self.assertEqual(video.missed_sells(game, money, lambda: Counter(["j_joker", "j_cavendish"])), [])

    def test_no_sale_is_inferred_of_a_joker_held_twice(self) -> None:
        from antezero_bridge import video

        game = played(CASE8[0], [*CASE8_SHOP, JOKER[0], JOKER[0]])
        self.assertEqual(video.missed_sells(game, game.dollars + JOKER[1], lambda: Counter(["j_joker"])), [])


class SimScreens:
    """The screens of a game played in the simulator: frame `k` is the screen before its click `k`."""

    def __init__(self, seed: str, codes: list[str]) -> None:
        game = antezero_sim.Game(seed, "b_red", 1)
        self.truth = codes
        self.screens = [SimScreen.of(game)]
        for code in codes:
            game.step(code)
            self.screens.append(SimScreen.of(game))
        # What a test makes the classifiers read of a click, by its frame.
        self.misreads: dict[int, str | None] = {}

    def clicks(self) -> list[SimpleNamespace]:
        from antezero_bridge import video
        from game.balatro_models import ActionType

        types = {verb: kind for kind, verbs in video.BUTTONS.items() for verb in verbs}
        types |= {"sort 0": ActionType.SORT_RANK, "sort 1": ActionType.SORT_SUIT}
        return [SimpleNamespace(action_type=types.get(code, types[code.split()[0]]), target_frame=at, frame_id=at)
                for at, code in enumerate(self.truth)]

    def at(self, click: SimpleNamespace) -> SimScreen:
        return self.screens[click.target_frame]

    def read(self, game: "antezero_sim.Game", click: SimpleNamespace, legal: list[str]) -> str | None:
        return self.misreads.get(click.target_frame, self.truth[click.target_frame])


@unittest.skipIf(antezero_sim is None, "antezero_sim is not installed")
class TestWalk(unittest.TestCase):
    def walk(self, seed: str, screens: SimScreens, clicks: list[SimpleNamespace] | None = None) -> Any:
        from antezero_bridge import video

        walk = video.Walk()
        video.follow(antezero_sim.Game(seed, "b_red", 1), clicks or screens.clicks(), screens, walk)
        return walk

    def test_true_screens_are_walked_to_the_last_click_which_no_screen_follows(self) -> None:
        for seed in ("BRIDGE1", "BRIDGE2", "BRIDGE3"):
            _, codes = random_game(seed, 300)
            walk = self.walk(seed, SimScreens(seed, codes))
            self.assertEqual(walk.codes, codes[:-1])
            self.assertGreater(len(walk.codes), 5)
            self.assertIsNone(walk.stopped)
            self.assertNotIn("repaired", walk.outcomes)

    def test_a_misread_play_is_repaired_by_the_score(self) -> None:
        seed, straight, flush = CASE8
        screens = SimScreens(seed, ["select", straight, "cash_out", "end_shop"])
        screens.misreads[1] = flush
        walk = self.walk(seed, screens)
        self.assertEqual(walk.codes, ["select", straight, "cash_out"])
        self.assertEqual(walk.outcomes["repaired"], 1)
        self.assertIsNone(walk.stopped)

    def test_the_walk_stops_before_a_play_the_screen_cannot_settle(self) -> None:
        seed, straight, flush = CASE8
        for read in (straight, flush):
            screens = SimScreens(seed, ["select", straight, "cash_out", "end_shop"])
            screens.misreads[1] = read
            del screens.screens[2].hud["chips"]
            walk = self.walk(seed, screens)
            self.assertEqual(walk.codes, ["select"])
            self.assertIn("chips was not read", walk.stopped)

    def test_the_walk_stops_at_a_screen_that_is_not_the_simulators(self) -> None:
        seed, straight, _ = CASE8
        screens = SimScreens(seed, ["select", straight, "cash_out", "end_shop", "select"])
        screens.screens[3].hud["dollars"] += 1
        walk = self.walk(seed, screens)
        self.assertEqual(walk.codes, ["select", straight])
        self.assertIn("is not the simulator's game after cash_out", walk.stopped)

        screens = SimScreens("OTHER", ["select", "play 1"])
        screens.screens[0].hud["dollars"] = 99
        walk = self.walk("OTHER", screens)
        self.assertEqual(walk.codes, [])
        self.assertIn("the first screen is not a new run", walk.stopped)

    def test_the_walk_stops_at_a_click_whose_button_is_not_legal(self) -> None:
        from game.balatro_models import ActionType

        seed, straight, _ = CASE8
        screens = SimScreens(seed, ["select", straight, "cash_out", "end_shop"])
        clicks = screens.clicks()
        clicks[1].action_type = ActionType.NEXT_ROUND
        walk = self.walk(seed, screens, clicks)
        self.assertEqual(walk.codes, [])
        self.assertIn("legal after no result", walk.stopped)

        clicks = screens.clicks()
        clicks[0].action_type = ActionType.NEXT_ROUND
        walk = self.walk(seed, screens, clicks)
        self.assertEqual(walk.codes, [])
        self.assertIn("no legal", walk.stopped)

    def test_a_click_the_classifiers_read_nothing_of_is_recorded_only_if_the_screen_settles_it(self) -> None:
        seed, straight, _ = CASE8
        screens = SimScreens(seed, ["select", straight, "cash_out", "end_shop"])
        screens.misreads[1] = None
        self.assertEqual(self.walk(seed, screens).codes, ["select", straight, "cash_out"])

        screens = SimScreens(seed, ["select", "play 211", "play 1"])
        screens.misreads[1] = None
        del screens.screens[2].seen["hand"]
        walk = self.walk(seed, screens)
        self.assertEqual(walk.codes, ["select"])
        self.assertIn("does not fit the screen", walk.stopped)


if __name__ == "__main__":
    unittest.main()
