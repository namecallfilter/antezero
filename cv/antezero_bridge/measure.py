"""Measures, on `.bal` recordings of real play, how much of an action the simulator recovers
from the screen, and how many misread actions it repairs.

    python -m antezero_bridge.measure <folder of .bal files> [--games N] [--error 0.05]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import time


from antezero_bridge import bal, reconcile


def actions(path: Path):
    """Each action of a recording with the game it is taken in."""
    rows = [row.split("#")[0].strip() for row in path.read_text().splitlines()]
    rows = [row for row in rows if row]
    game = bal.start(rows[0])
    for row in rows[1:]:
        if row == "endless":
            game.endless = True
            continue
        if row.startswith("dbg_"):
            return
        code = bal.legal_code(game, row)
        # Drags are not clicks: the order of the jokers and of the hand is read off the screen.
        if not code.startswith("move_"):
            yield game, code
        game.step(code)


def misread(code: str, legal: list[str], rng: random.Random) -> str:
    """A classifier's mistake: one hand card too many or too few, or the slot next to the right one."""
    near = [other for other in legal if other != code and reconcile.distance(code, other) == 1]
    return rng.choice(near) if near else code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--error", type=float, default=0.05, help="share of actions read wrong")
    args = parser.parse_args()

    rng = random.Random(0)
    # per button: actions, legal actions of the button, and how often the HUD alone, then the
    # HUD and the next hand, then those and the cards in every area, leave only the true action
    seen = defaultdict(lambda: [0, 0, 0, 0, 0])
    wrong = repaired = unrepaired = broke = total = 0
    start = time.perf_counter()
    for path in sorted(args.folder.glob("*.bal"))[: args.games]:
        for game, code in actions(path):
            button = reconcile.verb(code)
            legal = reconcile.candidates(game, button)
            after = game.clone()
            after.step(code)
            by_hud = reconcile.consistent(game, legal, reconcile.observe(after))
            by_hand = reconcile.consistent(game, by_hud, reconcile.observe(after, hand=True))
            by_all = reconcile.consistent(game, by_hand, reconcile.observe(after, hand=True, items=True))
            row = seen[button]
            row[0] += 1
            row[1] += len(legal)
            row[2] += by_hud == [code]
            row[3] += by_hand == [code]
            row[4] += by_all == [code]

            total += 1
            read = misread(code, legal, rng) if rng.random() < args.error else code
            wrong += read != code
            fixed = min(by_all, key=lambda other: reconcile.distance(read, other))
            if read != code:
                repaired += fixed == code
                unrepaired += fixed != code
            else:
                broke += fixed != code

    print(f"{total} actions in {time.perf_counter() - start:.0f} s")
    print(f"{'button':<16}{'actions':>8}{'legal/action':>14}{'HUD alone':>11}{'+ hand':>9}{'+ items':>9}")
    rows = sorted(seen.items(), key=lambda kv: -kv[1][0])
    rows.append(("all", [sum(column) for column in zip(*seen.values())]))
    for button, (n, legal, hud, hand, items) in rows:
        print(f"{button:<16}{n:>8}{legal / n:>14.1f}{hud / n:>11.1%}{hand / n:>9.1%}{items / n:>9.1%}")
    print(f"misread {wrong} of {total} ({wrong / total:.1%}); repaired {repaired}, not repaired {unrepaired}; "
          f"correct reads changed {broke}; wrong after repair {(unrepaired + broke) / total:.2%}")


if __name__ == "__main__":
    main()
