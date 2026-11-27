"""Stage 2 with the simulator in the loop (prototype): the clicks `main.py` found in a video,
as a `.bal` recording.

    python -m antezero_bridge.video <video> <actions json> --seed S --deck b_magic --stake 8 \
        --rules steamodded --out run.bal

The simulator holds the game. A click whose button has one legal action needs no classifier.
For the others the hand detector says which cards are raised and the item detector which
slot the button sits at, and the screen before the next click accepts or repairs the action
(`reconcile.settle`). A click is recorded only if that screen then shows the simulator's game
in every HUD number read: the walk stops at the first click for which it does not, and the
recording is what came before it.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
# The detector's shelf each button's slot is in.
SHELVES = {"buy": "shop_items", "buy_use": "shop_items", "redeem": "vouchers", "open": "boosters",
           "pick": "pack_items", "sell_joker": "joker_bar", "sell_consumable": "consumeables",
           "use": "consumeables"}
# Clicks the detector can miss without the game stopping: each has one outcome.
BRIDGES = ("cash_out", "end_shop", "skip_pack", "select")
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
    # next screen shows that nothing happened), "dropped" (no legal action for the button).
    outcomes: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)
    # Recorded reads that the next screen fits no better than some other action (their place in
    # `codes`, and how many such actions): they rest on the classifiers alone.
    unconfirmed: dict[int, int] = field(default_factory=dict)
    # Why the walk stopped before the video's last click, if it did.
    stopped: str | None = None
    seconds: Counter = field(default_factory=Counter)
    # Clicks whose result needed the cards on the next screen read, beside its HUD.
    card_reads: int = 0
    # HUD numbers read on the screens of the recorded clicks; each is the simulator's.
    hud_reads: int = 0


RANKS = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}


def look(frame: Any, state: dict[str, Any], walk: Walk) -> dict[str, Any]:
    """The cards on screen: the jokers, the consumables and, in a round or a pack that shows it, the hand."""
    in_pack = state["pack_cards"] is not None
    walk.card_reads += 1
    start = time.perf_counter()
    bars = recognize_shop(frame, mode="bars", device="cpu")
    seen = {"jokers": sorted(item.key for item in bars.joker_bar),
            "consumables": sorted(item.key for item in bars.consumeables)}
    walk.seconds["item detector"] += time.perf_counter() - start
    if state["hand"]:
        start = time.perf_counter()
        cards = recognize_hand_cards(frame, device="cpu", roi_key="pack_hand" if in_pack else "card_hand").cards
        seen["hand"] = sorted(f"{c.suit[0]}_{RANKS.get(c.rank, c.rank)}" for c in cards if not c.face_down)
        walk.seconds["hand detector"] += time.perf_counter() - start
    return seen


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


def _raised(frame: Any, hand_size: int, in_pack: bool) -> int | None:
    """The mask of the raised hand cards, or None if the detector does not see the simulator's hand."""
    cards = recognize_hand_cards(frame, device="cpu", roi_key="pack_hand" if in_pack else "card_hand").cards
    cards = sorted(cards, key=lambda c: c.box[0])
    if len(cards) != hand_size:
        return None
    return sum(1 << i for i, card in enumerate(cards) if card.selected)


def read(game: antezero_sim.Game, action: Any, frame: Any, legal: list[str], walk: Walk) -> str:
    """The classifiers' read of a click that has several legal actions."""
    state = json.loads(game.state_json())
    in_pack = state["pack_cards"] is not None
    mask = 0
    if any(code.split()[-1] != "0" for code in legal if reconcile.verb(code) in ("play", "discard", "use", "pick", "buy_use")):
        start = time.perf_counter()
        seen = _raised(frame, len(state["hand"]), in_pack)
        walk.seconds["hand detector"] += time.perf_counter() - start
        if seen is None:
            walk.notes.append(f"frame {action.frame_id}: hand not seen as {len(state['hand'])} cards")
        mask = seen or 0
    button = reconcile.verb(legal[0])
    if button in ("play", "discard"):
        return f"{button} {mask}"
    if button == "sort":
        return "sort 0" if action.action_type == A.SORT_RANK else "sort 1"

    start = time.perf_counter()
    shelves = recognize_shop(frame, mode="pack" if in_pack else "shop", device="cpu")
    walk.seconds["item detector"] += time.perf_counter() - start
    best: tuple[float, str] | None = None
    for name in {reconcile.verb(code) for code in legal}:
        items = sorted(getattr(shelves, SHELVES[name]), key=lambda item: item.box[0])
        side = name in ("sell_joker", "sell_consumable", "use")
        slot = _slot(action.button_box, items, side)
        if slot is None:
            continue
        x1, _, x2, _ = items[slot].box
        gap = abs((action.button_box[0] + action.button_box[2]) / 2 - (x2 + 1.2 * (x2 - x1) if side else (x1 + x2) / 2))
        code = f"{name} {slot}" + (f" {mask}" if name in ("use", "pick", "buy_use") else "")
        if best is None or gap < best[0]:
            best = (gap, code)
    return best[1] if best else legal[0]


def missed_sells(game: antezero_sim.Game, frame: Any, money: Any, walk: Walk) -> list[str]:
    """The sales whose click the detector missed: jokers that are gone from the bar, if the money agrees."""
    state = json.loads(game.state_json())
    if money is None or money == state["dollars"] or not state["jokers"]:
        return []
    start = time.perf_counter()
    bar = Counter(item.key for item in recognize_shop(frame, mode="joker_only", device="cpu").joker_bar)
    walk.seconds["item detector"] += time.perf_counter() - start
    owned = Counter(c["key"] for c in state["jokers"])
    if bar - owned or not owned - bar:
        return []
    ahead = game.clone()
    codes = []
    for key in (owned - bar).elements():
        code = f"sell_joker {[c['key'] for c in json.loads(ahead.state_json())['jokers']].index(key)}"
        if code not in ahead.legal_codes():
            return []
        ahead.step(code)
        codes.append(code)
    return codes if ahead.dollars == money else []


class Screens:
    """What the classifiers read on the frames just before the clicks of a video."""

    def __init__(self, video: Path, clicks: list[Any], walk: Walk) -> None:
        self.walk = walk
        start = time.perf_counter()
        self.images = extract_video_frames(video, [click.target_frame for click in clicks])
        walk.seconds["frame extraction"] = time.perf_counter() - start
        start = time.perf_counter()
        self.huds = {out.frame_index: hud_numbers(out) for out in batch_recognize_hud(
            [self._frame(c.target_frame) for c in clicks], fields=tuple(OCR_FIELDS.values()), device="cpu")}
        walk.seconds["HUD OCR"] = time.perf_counter() - start

    def _frame(self, at: int) -> tuple[Any, int, float]:
        return (self.images[at], at, 0.0)

    def hud(self, at: int) -> dict[str, Any]:
        return self.huds[at]

    def read(self, game: antezero_sim.Game, click: Any, legal: list[str]) -> str:
        return read(game, click, self._frame(click.target_frame), legal, self.walk)

    def cards(self, at: int, state: dict[str, Any]) -> dict[str, Any]:
        return look(self._frame(at), state, self.walk)

    def missed_sells(self, at: int, game: antezero_sim.Game) -> list[str]:
        return missed_sells(game, self._frame(at), self.huds[at].get("dollars"), self.walk)


def follow(game: antezero_sim.Game, clicks: list[Any], screens: Any, walk: Walk) -> None:
    """Plays in `game` what the clicks did and records it in `walk`, up to the first click that `screens` does not settle."""
    if not clicks:
        return
    hud = screens.hud(clicks[0].target_frame)
    if not reconcile.shows(game, hud):
        walk.stopped = f"frame {clicks[0].frame_id}: the first screen is not a new run of this seed, deck and stake (HUD {hud})"
        return
    walk.hud_reads += len(hud)
    for i, click in enumerate(clicks):
        buttons = BUTTONS[click.action_type]
        # A click the detector missed is played first if the clicked button is legal only after it.
        result, steps = game.clone(), []
        legal = [code for button in buttons for code in reconcile.candidates(game, button)]
        if not legal:
            for bridge in BRIDGES:
                if reconcile.candidates(game, bridge) == [bridge]:
                    ahead = game.clone()
                    ahead.step(bridge)
                    legal = [code for button in buttons for code in reconcile.candidates(ahead, button)]
                    if legal:
                        result, steps = ahead, [(bridge, "bridged")]
                        break
        if not legal:
            walk.outcomes["dropped"] += 1
            walk.notes.append(f"frame {click.frame_id}: no legal {click.action_type.value} in {game.phase}")
            continue
        guess = legal[0] if len(legal) == 1 else screens.read(result, click, legal)
        if i + 1 == len(clicks):
            if len(legal) > 1:
                walk.stopped = f"frame {click.frame_id}: no screen after the last click shows what it did"
                return
            settled = reconcile.Settled(guess, "forced")
        else:
            # The screen just before the next click shows this click's result.
            following = clicks[i + 1].target_frame
            hud = screens.hud(following)
            try:
                settled = reconcile.settle(result, legal, guess, hud, lambda state, at=following: screens.cards(at, state))
            except reconcile.Unresolved as e:
                walk.stopped = f"frame {click.frame_id}: {click.action_type.value}: {e}"
                return
        if settled.code is None:
            walk.notes.append(f"frame {click.frame_id}: {click.action_type.value} did nothing")
        else:
            result.step(settled.code)
            steps.append((settled.code, settled.outcome))
        if i + 1 < len(clicks):
            for code in screens.missed_sells(following, result):
                result.step(code)
                steps.append((code, "inferred"))
            if not reconcile.shows(result, hud):
                walk.stopped = (f"frame {click.frame_id}: {click.action_type.value}: the next screen (HUD {hud}) "
                                f"is not the simulator's game after {settled.code}")
                return
            walk.hud_reads += len(hud)

        if settled.code is None:
            walk.outcomes["rejected"] += 1
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


def walk_clicks(video: Path, actions: dict[str, Any], seed: str, back: str, stake: int, steamodded: bool) -> Walk:
    game = antezero_sim.Game(seed, back, stake)
    game.steamodded = steamodded
    walk = Walk()
    frames = filter_premature_next_rounds(remove_duplicates(actions["frames"]), actions["key"])
    clicks = plan_action_pipeline({"key": actions["key"], "frames": frames})["scheduled_actions"]
    follow(game, clicks, Screens(video, clicks, walk), walk)
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    steamodded = args.rules == "steamodded"
    walk = walk_clicks(args.video, json.loads(args.actions.read_text()), args.seed, args.deck, args.stake, steamodded)
    comments = {at: f"unconfirmed: {rivals} other actions fit the next screen" for at, rivals in walk.unconfirmed.items()}
    text = bal.script(args.seed, args.deck, args.stake, walk.codes, steamodded=steamodded, comments=comments)
    bal.check(text)
    if walk.stopped:
        text += f"# stopped at {walk.stopped}\n"
    args.out.write_text(text)
    print(f"{len(walk.codes)} actions -> {args.out} in {time.perf_counter() - start:.1f} s")
    print(f"stopped at {walk.stopped}" if walk.stopped else "walked to the video's last click")
    print("clicks:", dict(walk.outcomes), "| needed the cards on screen read:", walk.card_reads,
          "| unconfirmed reads:", len(walk.unconfirmed))
    print(f"HUD numbers read on the screens of the recorded clicks, each the simulator's: {walk.hud_reads}")
    print("seconds:", {name: round(value, 1) for name, value in walk.seconds.items()})
    for note in walk.notes:
        print(" ", note)


if __name__ == "__main__":
    main()
