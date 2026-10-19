"""Diff two state streams step by step with the canon.rs rules.

Each input is JSONL where a line is either a bare ``H.state`` dump (as
written by ``replay --oracle-out``) or a replay record with a ``state`` field
(as written by ``replay -o``). Prints the first divergence (or all with
``--all``); streams of different lengths, steps whose history lines differ
and rejected steps (no state) count as divergences too. Exits with status 1
on any divergence::

    uv run python -m antezero_bridge.compare game.jsonl oracle.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .canon import diff, from_lua


@dataclass
class Step:
    """One state of a stream (``line`` / ``error`` only for replay records)."""

    line: str | None
    state: dict | None
    error: str | None = None


def load(path: Path) -> list[Step]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        j = json.loads(ln)
        if isinstance(j, dict) and "step" in j:  # a replay record
            state = j.get("state") if isinstance(j.get("state"), dict) else None
            out.append(Step(j.get("line"), state, j.get("error") or (None if state else "no state")))
        else:
            out.append(Step(None, j))
    return out


def divergence(a: Step, b: Step) -> list[tuple[str, str, str]]:
    """The differences between two steps (empty if they match)."""
    if a.line is not None and b.line is not None and a.line != b.line:
        return [("<history line>", a.line, b.line)]
    if a.state is None or b.state is None:
        return [("<state>", a.error or "ok", b.error or "ok")]
    return diff(from_lua(a.state), from_lua(b.state))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--all", action="store_true", help="report every diverging step")
    ap.add_argument("--max", type=int, default=40, help="max fields shown per step")
    a = ap.parse_args(argv)
    sa, sb = load(a.a), load(a.b)
    n = min(len(sa), len(sb))
    bad = 0
    for i in range(n):
        d = divergence(sa[i], sb[i])
        if d:
            bad += 1
            print(f"DIVERGED at step {i}: {sa[i].line or sb[i].line or ''}")
            for k, x, y in d[: a.max]:
                print(f"  {k}: A={x} B={y}")
            if len(d) > a.max:
                print(f"  ... {len(d) - a.max} more")
            if not a.all:
                break
    if len(sa) != len(sb):
        bad += 1
        print(f"length differs: {len(sa)} vs {len(sb)} states")
    if not bad:
        print(f"no divergence over {n} steps")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
