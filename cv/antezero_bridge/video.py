"""Stage 2 with the simulator in the loop (prototype): the clicks `main.py` found in a video,
as a `.bal` recording.

    python -m antezero_bridge.video <video> <actions json> --seed S --deck b_magic --stake 8 --out run.bal

The simulator holds the game. A click whose button has one legal action needs no classifier.
For the others the hand detector says which cards are raised and the item detector which
slot the button sits at, and the HUD read before the next click accepts or repairs the action.
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
HUD_FIELDS = {"hud_money": "dollars", "hud_hands": "hands_left", "hud_discards": "discards_left",
              "hud_ante": "ante", "hud_round": "round"}


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
    seconds: Counter = field(default_factory=Counter)
    # Clicks settled by the HUD alone, and those that also needed the cards on screen read.
    card_reads: int = 0
    # HUD numbers read just before each click, and how many differ from the simulator's.
    hud_reads: int = 0
    hud_misses: int = 0


RANKS = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}


def look(frame: Any, state: dict[str, Any], walk: Walk) -> dict[str, Any]:
    """The cards on screen: the jokers, the consumables and, in a round or a pack that shows it, the hand."""
    in_pack = state["pack_cards"] is not None
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


def predicted(game: antezero_sim.Game) -> dict[str, Any]:
    """What `look` and the HUD OCR should give in `game`."""
    state = json.loads(game.state_json())
    return {"dollars": state["dollars"], "hands_left": state["hands_left"], "discards_left": state["discards_left"],
            "ante": state["ante"], "round": state["round"],
            "jokers": sorted(c["key"] for c in state["jokers"]),
            "consumables": sorted(c["key"] for c in state["consumables"]),
            "hand": sorted(c["card"] for c in state["hand"] if c["facing"] == "front"), "state": state}


def agreement(expected: dict[str, Any], screen: dict[str, Any]) -> float:
    """How many of the things read off the screen are as expected (a hand counts by the share of its cards)."""
    score = 0.0
    for key, value in screen.items():
        if key == "hand":
            both = Counter(value) & Counter(expected["hand"])
            score += sum(both.values()) / max(len(value), len(expected["hand"]), 1)
        else:
            score += expected[key] == value
    return score


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


def missed_sells(game: antezero_sim.Game, frame: Any, money: Any, walk: Walk) -> None:
    """Plays the sales whose click the detector missed: jokers that are gone from the bar, if the money agrees."""
    state = json.loads(game.state_json())
    if money is None or money == state["dollars"] or not state["jokers"]:
        return
    start = time.perf_counter()
    bar = Counter(item.key for item in recognize_shop(frame, mode="joker_only", device="cpu").joker_bar)
    walk.seconds["item detector"] += time.perf_counter() - start
    owned = Counter(c["key"] for c in state["jokers"])
    if bar - owned or not owned - bar:
        return
    ahead = game.clone()
    codes = []
    for key in (owned - bar).elements():
        code = f"sell_joker {[c['key'] for c in json.loads(ahead.state_json())['jokers']].index(key)}"
        if code not in ahead.legal_codes():
            return
        ahead.step(code)
        codes.append(code)
    if ahead.dollars != money:
        return
    for code in codes:
        game.step(code)
        walk.codes.append(code)
        walk.outcomes["inferred"] += 1
        walk.notes.append(f"frame {frame[1]}: a joker is gone and the money fits: {code}")


def walk_clicks(video: Path, actions: dict[str, Any], seed: str, back: str, stake: int) -> Walk:
    game = antezero_sim.Game(seed, back, stake)
    # The streamer's game runs under Steamodded, which rounds xMult differently from vanilla.
    game.steamodded = True
    walk = Walk()
    frames = filter_premature_next_rounds(remove_duplicates(actions["frames"]), actions["key"])
    clicks = plan_action_pipeline({"key": actions["key"], "frames": frames})["scheduled_actions"]

    start = time.perf_counter()
    images = extract_video_frames(video, [click.target_frame for click in clicks])
    walk.seconds["frame extraction"] = time.perf_counter() - start
    start = time.perf_counter()
    huds = {out.frame_index: out for out in batch_recognize_hud(
        [(images[c.target_frame], c.target_frame, 0.0) for c in clicks], fields=tuple(HUD_FIELDS), device="cpu")}
    walk.seconds["HUD OCR"] = time.perf_counter() - start

    for i, click in enumerate(clicks):
        buttons = BUTTONS[click.action_type]
        frame = (images[click.target_frame], click.target_frame, 0.0)
        missed_sells(game, frame, huds[click.target_frame].values.get("hud_money"), walk)
        expected = predicted(game)
        for key, name in HUD_FIELDS.items():
            value = huds[click.target_frame].values.get(key)
            if value is not None:
                walk.hud_reads += 1
                walk.hud_misses += value != expected[name]
                if value != expected[name]:
                    walk.notes.append(f"frame {click.frame_id}: HUD {name} {value}, simulator {expected[name]}")
        legal = [code for button in buttons for code in reconcile.candidates(game, button)]
        if not legal:
            for bridge in BRIDGES:
                if reconcile.candidates(game, bridge) == [bridge]:
                    ahead = game.clone()
                    ahead.step(bridge)
                    if any(reconcile.candidates(ahead, button) for button in buttons):
                        game.step(bridge)
                        walk.codes.append(bridge)
                        walk.outcomes["bridged"] += 1
                        legal = [code for button in buttons for code in reconcile.candidates(game, button)]
                        break
        if not legal:
            walk.outcomes["dropped"] += 1
            walk.notes.append(f"frame {click.frame_id}: no legal {click.action_type.value} in {game.phase}")
            continue
        guess = legal[0] if len(legal) == 1 else read(game, click, frame, legal, walk)
        code = guess if guess in legal else min(legal, key=lambda other: reconcile.distance(guess, other))
        if i + 1 < len(clicks):
            # The screen just before the next click shows this click's result. `None` stands
            # for a click that did nothing (a button the detector took for pressed).
            start = time.perf_counter()
            after = {}
            for option in [None, *legal]:
                ahead = game.clone()
                if option is not None:
                    ahead.step(option)
                after[option] = predicted(ahead)
            walk.seconds["simulator"] += time.perf_counter() - start
            following = clicks[i + 1].target_frame
            values = huds[following].values
            screen = {name: values[key] for key, name in HUD_FIELDS.items() if values.get(key) is not None}
            scores = {option: agreement(expected, screen) for option, expected in after.items()}
            best = [option for option in after if scores[option] == max(scores.values())]
            if len({json.dumps([after[o][k] for k in ("jokers", "consumables", "hand")]) for o in best}) > 1:
                cards = look((images[following], following, 0.0), after[code]["state"], walk)
                walk.card_reads += 1
                scores = {option: scores[option] + agreement(after[option], cards) for option in best}
                best = [option for option in best if scores[option] == max(scores.values())]
            if code not in best:
                if None in best and len(best) == 1:
                    walk.outcomes["rejected"] += 1
                    walk.notes.append(f"frame {click.frame_id}: {click.action_type.value} did nothing")
                    continue
                options = [option for option in best if option is not None]
                was, code = code, min(options, key=lambda other: reconcile.distance(guess, other))
                walk.notes.append(f"frame {click.frame_id}: read {was}, the screen fits {code}")
                outcome = "repaired"
            else:
                outcome = "forced" if len(legal) == 1 else "read"
        else:
            outcome = "forced" if len(legal) == 1 else "read"
        game.step(code)
        walk.codes.append(code)
        walk.outcomes[outcome] += 1
    return walk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("actions", type=Path, help="the actions JSON of main.py --skip-post-processing")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--deck", required=True, help="the deck's key, e.g. b_magic")
    parser.add_argument("--stake", type=int, required=True, help="1 (white) to 8 (gold)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    walk = walk_clicks(args.video, json.loads(args.actions.read_text()), args.seed, args.deck, args.stake)
    text = bal.script(args.seed, args.deck, args.stake, walk.codes)
    bal.check(text)
    args.out.write_text(text)
    print(f"{len(walk.codes)} actions -> {args.out} in {time.perf_counter() - start:.1f} s")
    print("clicks:", dict(walk.outcomes), "| needed the cards on screen read:", walk.card_reads)
    print(f"HUD numbers read before the clicks: {walk.hud_reads}, of which {walk.hud_misses} differ from the simulator")
    print("seconds:", {name: round(value, 1) for name, value in walk.seconds.items()})
    for note in walk.notes:
        print(" ", note)


if __name__ == "__main__":
    main()
