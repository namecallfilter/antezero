"""Plays a telemetry log's actions in AnteZero's simulator and scores every state against the log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import antezero_sim

from antezero_bridge.resync import Unsupported, resync
from antezero_bridge.telemetry import action_code, compare, dump, reorders, start_game


@dataclass
class Step:
    step: int
    action: str
    code: str | None
    # "match"; "transient" when the log's snapshot was taken before the game had finished the
    # previous action and the next entry matches again; "diverged" otherwise.
    status: str
    diffs: list[str]


@dataclass
class Replay:
    steps: list[Step] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    error: str | None = None

    def count(self, status: str) -> int:
        return sum(s.status == status for s in self.steps)

    @property
    def first_divergence(self) -> Step | None:
        return next((s for s in self.steps if s.status == "diverged"), None)


def _settles(game: antezero_sim.Game, code: str | None, following: dict[str, Any] | None) -> bool:
    """Whether the simulator, left as it is, agrees with the log again one entry later."""
    if following is None:
        return False
    ahead = game.clone()
    try:
        if code is not None:
            ahead.step(code)
        for move in reorders(dump(ahead), following["state"]):
            ahead.step(move)
    except (RuntimeError, ValueError):
        return False
    return not compare(dump(ahead), following["state"])


def replay(entries: list[dict[str, Any]], resync_on_divergence: bool = False) -> Replay:
    """Compares each entry's state with the simulator, then plays the entry's action.

    The simulator is given the seed, deck, stake and the actions only. With
    `resync_on_divergence` the logged state is loaded into it whenever the two differ, so
    every later transition is still tested on its own.
    """
    game = start_game(entries)
    card = entries[0]["state"]["deck"][0]
    offset = card["sort_id"] - card["playing_card"]
    out = Replay()
    for i, entry in enumerate(entries):
        lua = entry["state"]
        try:
            for move in reorders(dump(game), lua):
                game.step(move)
                out.codes.append(move)
        except (RuntimeError, ValueError) as e:
            out.error = f"step {entry['step']}: reorder not possible: {e}"
            break
        diffs = compare(dump(game), lua)
        code = action_code(entry)
        status = "match"
        if diffs:
            following = entries[i + 1] if i + 1 < len(entries) else None
            status = "transient" if _settles(game, code, following) else "diverged"
        out.steps.append(Step(entry["step"], entry["action"], code, status, diffs))
        if status == "diverged" and resync_on_divergence:
            try:
                resync(game, lua, offset)
            except Unsupported as e:
                out.error = f"step {entry['step']}: cannot resync: {e}"
                break
        if code is None:
            continue
        try:
            game.step(code)
        except (RuntimeError, ValueError) as e:
            out.error = f"step {entry['step']} {entry['action']} ({code}): {e}"
            break
        out.codes.append(code)
    return out


# Spectral cards whose random suit or rank Steamodded draws from its own tables.
_SMODS_PICKS = {"c_sigil", "c_ouija", "c_grim", "c_familiar", "c_incantation"}


def steamodded_cause(entries: list[dict[str, Any]], step: Step) -> str | None:
    """Which known difference between Steamodded and the vanilla game explains a divergence, if any."""
    previous = entries[step.step - 2]
    fields = {d.split(":")[0] for d in step.diffs}
    if previous["action_metadata"].get("key") in _SMODS_PICKS and fields <= {"hand", "draw_pile"}:
        return "spectral suit or rank"
    blind_started = previous["state"]["blind"] != entries[step.step - 1]["state"]["blind"]
    seal_stream = {f for f in fields if f.startswith("rng stdseal")}
    if blind_started and fields <= {"hand", "draw_pile"} | seal_stream:
        return "blind start"
    return None
