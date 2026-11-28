"""Stage 2 with the simulator in the loop (prototype): the clicks `main.py` found in a video,
as a `.bal` recording.

    python -m antezero_bridge.video <video> <actions json> --seed S --deck b_magic --stake 8 \
        --rules steamodded --out run.bal [--device cuda]

The simulator holds the game. A click whose button has one legal action needs no classifier.
For the others the hand detector says which cards are raised and the item detector which
slot the button sits at, and the screen before the next click accepts or repairs the action
(`reconcile.settle`). A click is recorded only if that screen then shows the simulator's game
in every HUD number read: the walk stops at the first click for which it does not, or whose
button is not legal, and the recording is what came before it. The last click has no screen
after it and is not recorded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import antezero_sim

from antezero_bridge import bal, reconcile
from classifiers.common.video_utils import extract_video_frames
from classifiers.inference import batch_recognize_hud, recognize_hand_cards, recognize_shop
from game.balatro_models import ActionType as A
from post_processing import filter_premature_next_rounds, plan_action_pipeline, remove_duplicates

BUTTONS = {
    A.SELECT_BLIND: ("select",), A.SKIP_BLIND: ("skip",), A.REROLL_BOSS: ("reroll_boss",),
    A.PLAY_HAND: ("play",), A.DISCARD: ("discard",), A.CASH_OUT: ("cash_out",),
    A.NEXT_ROUND: ("end_shop",), A.REROLL_SHOP: ("reroll",), A.BUY_CARD: ("buy",),
    A.BUY_AND_USE: ("buy_use",), A.REDEEM_VOUCHER: ("redeem",), A.OPEN_PACK: ("open",),
    A.USE_IN_PACK: ("pick",), A.SKIP_PACK: ("skip_pack",), A.USE_CONSUMABLE: ("use",),
    A.SELL_CARD: ("sell_joker", "sell_consumable"), A.SORT_RANK: ("sort",), A.SORT_SUIT: ("sort",),
}
# The detector's shelf each button's slot is in, and the attribute a shelf's items come back in.
SHELVES = {"buy": "shop", "buy_use": "shop", "redeem": "voucher", "open": "booster", "pick": "pack",
           "sell_joker": "joker_bar", "sell_consumable": "consumeables", "use": "consumeables"}
SHELF_ITEMS = {"shop": "shop_items", "voucher": "vouchers", "booster": "boosters", "pack": "pack_items",
               "joker_bar": "joker_bar", "consumeables": "consumeables"}
# The OCR's field of each HUD number.
OCR_FIELDS = {"dollars": "hud_money", "chips": "hud_round_score", "hands_left": "hud_hands",
              "discards_left": "hud_discards", "ante": "hud_ante", "round": "hud_round"}


@dataclass
class Walk:
    codes: list[str] = field(default_factory=list)
    # How each click was settled: "forced" (its button has one legal action and the next screen
    # does not contradict it), "read" (the classifiers' read is the action that fits the next
    # screen best), "repaired" (another action fits it better), "bridged" (a missed click was
    # filled in), "inferred" (a sale the detector missed, seen on the joker bar), "rejected" (the
    # next screen shows that nothing happened).
    outcomes: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)
    # Recorded clicks that the next screen fits no better than some other result (their place in
    # `codes`, and how many such results): they rest on the classifiers alone.
    unconfirmed: dict[int, int] = field(default_factory=dict)
    # Why the walk stopped before the video's last click, if it did.
    stopped: str | None = None
    seconds: Counter = field(default_factory=Counter)
    # Clicks whose result needed the cards on the next screen read, beside its HUD.
    card_reads: int = 0
    # HUD numbers read on the screens of the recorded clicks; each is the simulator's.
    hud_reads: int = 0


RANKS = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}


def hud_numbers(hud: Any) -> dict[str, Any]:
    """The HUD numbers the OCR read on a frame, by their names in the simulator's state."""
    values = {name: hud.values.get(OCR_FIELDS[name]) for name in reconcile.HUD_NUMBERS}
    return {name: value for name, value in values.items() if value is not None}


def _slot(box: list[int] | None, items: list[Any], side: bool) -> int | None:
    """The item a button belongs to: buy and use buttons hang under their card, sell buttons at its right."""
    if box is None or not items:
        return None
    bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def far(item: Any) -> float:
        x1, y1, x2, y2 = item.box
        ax = x2 + 1.2 * (x2 - x1) if side else (x1 + x2) / 2
        return abs(bx - ax) + 0.25 * abs(by - (y1 + y2) / 2) + (1000 if side and bx < x1 else 0)

    return min(range(len(items)), key=lambda i: far(items[i]))


def missed_sells(game: antezero_sim.Game, money: Any, bar: Callable[[], Counter]) -> list[str]:
    """The sales whose click the detector missed: jokers that are gone from the joker bar, if the money agrees.

    `bar()` reads the bar's jokers. Nothing is inferred for a joker the game holds twice: the bar
    does not say which of the two was sold.
    """
    if money is None or money == game.dollars:
        return []
    state = json.loads(game.state_json())
    if not state["jokers"]:
        return []
    owned = Counter(c["key"] for c in state["jokers"])
    gone = owned - bar()
    if bar() - owned or not gone or any(owned[key] > 1 for key in gone):
        return []
    ahead = game.clone()
    codes = []
    for key in gone:
        code = f"sell_joker {[c['key'] for c in json.loads(ahead.state_json())['jokers']].index(key)}"
        if code not in ahead.legal_codes():
            return []
        ahead.step(code)
        codes.append(code)
    return codes if ahead.dollars == money else []


class Screen:
    """What the classifiers read on one frame of a video (`reconcile.Screen`). Each detector looks at a part of it once."""

    def __init__(self, frame: tuple[Any, int, float], hud: dict[str, Any], walk: Walk, device: str) -> None:
        self.frame, self.hud, self.walk, self.device = frame, hud, walk, device
        self._shelves: dict[str, list[Any]] = {}
        self._hands: dict[str, list[Any]] = {}

    def shelf(self, name: str) -> list[Any]:
        """The items the detector sees on one shelf."""
        if name not in self._shelves:
            start = time.perf_counter()
            found = recognize_shop(self.frame, shelves=[name], device=self.device)
            self._shelves[name] = getattr(found, SHELF_ITEMS[name])
            self.walk.seconds["item detector"] += time.perf_counter() - start
        return self._shelves[name]

    def hand(self, in_pack: bool) -> list[Any]:
        """The hand cards the detector sees, from the left."""
        roi = "pack_hand" if in_pack else "card_hand"
        if roi not in self._hands:
            start = time.perf_counter()
            cards = recognize_hand_cards(self.frame, device=self.device, roi_key=roi).cards
            self._hands[roi] = sorted(cards, key=lambda c: c.box[0])
            self.walk.seconds["hand detector"] += time.perf_counter() - start
        return self._hands[roi]

    def cards(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        """The cards on a screen that shows one of `states`: the jokers, the consumables and, if any of them has one, the hand."""
        self.walk.card_reads += 1
        seen = {"jokers": sorted(item.key for item in self.shelf("joker_bar")),
                "consumables": sorted(item.key for item in self.shelf("consumeables"))}
        dealt = [state for state in states if state["hand"]]
        if dealt:
            cards = self.hand(dealt[0]["pack_cards"] is not None)
            seen["hand"] = sorted(f"{c.suit[0]}_{RANKS.get(c.rank, c.rank)}" for c in cards if not c.face_down)
        return seen

    def sales(self, game: antezero_sim.Game) -> list[str]:
        return missed_sells(game, self.hud.get("dollars"), lambda: Counter(item.key for item in self.shelf("joker_bar")))

    def read(self, game: antezero_sim.Game, action: Any, legal: list[str]) -> str | None:
        """The classifiers' read of a click that has several legal actions; None if they do not see what it applied to."""
        state = json.loads(game.state_json())
        in_pack = state["pack_cards"] is not None
        mask, raised = 0, None
        if any(code.split()[-1] != "0" for code in legal if reconcile.verb(code) in ("play", "discard", "use", "pick", "buy_use")):
            cards = self.hand(in_pack)
            if len(cards) == len(state["hand"]):
                raised = sum(1 << i for i, card in enumerate(cards) if card.selected)
            else:
                self.walk.notes.append(f"frame {action.frame_id}: hand not seen as {len(state['hand'])} cards")
            mask = raised or 0
        button = reconcile.verb(legal[0])
        if button in ("play", "discard"):
            return None if raised is None else f"{button} {mask}"
        if button == "sort":
            return "sort 0" if action.action_type == A.SORT_RANK else "sort 1"

        best: tuple[float, str] | None = None
        for name in sorted({reconcile.verb(code) for code in legal}):
            items = sorted(self.shelf(SHELVES[name]), key=lambda item: item.box[0])
            side = name in ("sell_joker", "sell_consumable", "use")
            slot = _slot(action.button_box, items, side)
            if slot is None:
                continue
            x1, _, x2, _ = items[slot].box
            gap = abs((action.button_box[0] + action.button_box[2]) / 2 - (x2 + 1.2 * (x2 - x1) if side else (x1 + x2) / 2))
            code = f"{name} {slot}" + (f" {mask}" if name in ("use", "pick", "buy_use") else "")
            if best is None or gap < best[0]:
                best = (gap, code)
        return best[1] if best else None


class Screens:
    """What the classifiers read on the frames just before the clicks of a video."""

    def __init__(self, video: Path, clicks: list[Any], walk: Walk, device: str) -> None:
        start = time.perf_counter()
        images = extract_video_frames(video, [click.target_frame for click in clicks])
        walk.seconds["frame extraction"] = time.perf_counter() - start
        start = time.perf_counter()
        frames = {click.target_frame: (images[click.target_frame], click.target_frame, 0.0) for click in clicks}
        huds = batch_recognize_hud(list(frames.values()), fields=tuple(OCR_FIELDS.values()), device=device)
        self.screens = {out.frame_index: Screen(frames[out.frame_index], hud_numbers(out), walk, device) for out in huds}
        walk.seconds["HUD OCR"] = time.perf_counter() - start

    def at(self, click: Any) -> Screen:
        return self.screens[click.target_frame]

    def read(self, game: antezero_sim.Game, click: Any, legal: list[str]) -> str | None:
        return self.at(click).read(game, click, legal)


def follow(game: antezero_sim.Game, clicks: list[Any], screens: Any, walk: Walk) -> None:
    """Plays in `game` what the clicks did and records it in `walk`, up to the first click that `screens` does not settle.

    `screens.at(click)` is the `reconcile.Screen` just before a click, and `screens.read` the
    classifiers' read of one.
    """
    if not clicks:
        return
    hud = screens.at(clicks[0]).hud
    if not reconcile.shows(game, hud):
        walk.stopped = f"frame {clicks[0].frame_id}: the first screen is not a new run of this deck and stake (HUD {hud})"
        return
    walk.hud_reads += len(hud)
    # The screen just before the next click shows a click's result.
    for click, following in zip(clicks, clicks[1:]):
        name = click.action_type.value
        base, bridge, legal = reconcile.playable(game, BUTTONS[click.action_type])
        if not legal:
            walk.stopped = f"frame {click.frame_id}: no legal {name} in {game.phase}"
            return
        guess = legal[0] if len(legal) == 1 else screens.read(base, click, legal)
        screen = screens.at(following)
        try:
            settled = reconcile.settle(base, legal, guess, screen, BUTTONS[following.action_type])
        except reconcile.Unresolved as e:
            walk.stopped = f"frame {click.frame_id}: {name}: {e}"
            return
        if settled.code is None and bridge is not None:
            walk.stopped = f"frame {click.frame_id}: {name} is legal only after a missed {bridge}, and did nothing"
            return
        steps = [(code, "bridged") for code in (bridge,) if code is not None]
        steps += [(code, settled.outcome) for code in (settled.code,) if code is not None]
        steps += [(code, "inferred") for code in settled.sales]
        result = game.clone()
        for code, _ in steps:
            result.step(code)
        if not reconcile.shows(result, screen.hud):
            walk.stopped = (f"frame {click.frame_id}: {name}: the next screen (HUD {screen.hud}) "
                            f"is not the simulator's game after {settled.code}")
            return
        walk.hud_reads += len(screen.hud)

        if settled.code is None:
            walk.outcomes["rejected"] += 1
            walk.notes.append(f"frame {click.frame_id}: {name} did nothing")
        elif settled.outcome == "repaired":
            walk.notes.append(f"frame {click.frame_id}: read {guess}, the screen fits {settled.code}")
        for code, outcome in steps:
            if outcome == "inferred":
                walk.notes.append(f"frame {click.frame_id}: a joker is gone on the next screen and the money fits: {code}")
            elif outcome == settled.outcome and settled.rivals:
                walk.unconfirmed[len(walk.codes)] = settled.rivals
            walk.codes.append(code)
            walk.outcomes[outcome] += 1
        game = result
    walk.notes.append(f"frame {clicks[-1].frame_id}: {clicks[-1].action_type.value} is the last click: "
                      "no screen after it shows what it did, and it is not recorded")


def walk_clicks(video: Path, actions: dict[str, Any], seed: str, back: str, stake: int, steamodded: bool,
                device: str = "cpu") -> Walk:
    game = antezero_sim.Game(seed, back, stake)
    game.steamodded = steamodded
    walk = Walk()
    frames = filter_premature_next_rounds(remove_duplicates(actions["frames"]), actions["key"])
    clicks = plan_action_pipeline({"key": actions["key"], "frames": frames})["scheduled_actions"]
    follow(game, clicks, Screens(video, clicks, walk, device), walk)
    return walk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("actions", type=Path, help="the actions JSON of main.py --skip-post-processing")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--deck", required=True, help="the deck's key, e.g. b_magic")
    parser.add_argument("--stake", type=int, required=True, help="1 (white) to 8 (gold)")
    parser.add_argument("--rules", choices=("steamodded", "vanilla"), required=True,
                        help="whether the game in the video ran under Steamodded")
    parser.add_argument("--device", default="cpu", help="the torch device of the classifiers, e.g. cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    steamodded = args.rules == "steamodded"
    walk = walk_clicks(args.video, json.loads(args.actions.read_text()), args.seed, args.deck, args.stake, steamodded,
                       args.device)
    comments = {at: f"unconfirmed: {rivals} other results fit the next screen" for at, rivals in walk.unconfirmed.items()}
    text = bal.script(args.seed, args.deck, args.stake, walk.codes, steamodded=steamodded, comments=comments)
    bal.check(text)
    if walk.stopped:
        text += f"# stopped at {walk.stopped}\n"
    args.out.write_text(text)
    print(f"{len(walk.codes)} actions -> {args.out} in {time.perf_counter() - start:.1f} s")
    print(f"stopped at {walk.stopped}" if walk.stopped else "walked to the video's last click, which is not recorded")
    print("clicks:", dict(walk.outcomes), "| needed the cards on screen read:", walk.card_reads,
          "| unconfirmed reads:", len(walk.unconfirmed))
    print(f"HUD numbers read on the screens of the recorded clicks, each the simulator's: {walk.hud_reads}")
    print("seconds:", {name: round(value, 1) for name, value in walk.seconds.items()})
    for note in walk.notes:
        print(" ", note)


if __name__ == "__main__":
    main()
