"""Regenerate antezero_bridge/_codes.py from sim/src/data_gen.rs (read-only).

The sim's history files encode injections by enum index (``inject_joker 17
0 0`` = ``ALL_JOKERS[17]``); this script extracts the index -> game key
tables so the bridge can translate them without building the Rust crate.

Usage (from the repository root)::

    uv run python bridge/scripts/gen_codes.py           # rewrite _codes.py
    uv run python bridge/scripts/gen_codes.py --check   # exit 1 if stale
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sim" / "src" / "data_gen.rs"
OUT = ROOT / "bridge" / "src" / "antezero_bridge" / "_codes.py"

TABLES = [
    # (enum, ALL_ array, DEFS array, python name)
    ("Joker", "ALL_JOKERS", "JOKER_DEFS", "JOKERS"),
    ("Tarot", "ALL_TAROTS", "TAROT_DEFS", "TAROTS"),
    ("Planet", "ALL_PLANETS", "PLANET_DEFS", "PLANETS"),
    ("Spectral", "ALL_SPECTRALS", "SPECTRAL_DEFS", "SPECTRALS"),
    ("Voucher", "ALL_VOUCHERS", "VOUCHER_DEFS", "VOUCHERS"),
    ("Back", None, "BACK_DEFS", "BACKS"),
]


def block(text: str, header_re: str) -> str:
    m = re.search(header_re, text)
    if not m:
        raise SystemExit(f"not found: {header_re}")
    start = m.end()
    depth = 1
    i = start
    while depth:
        c = text[i]
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def render(text: str) -> str:
    """``_codes.py`` for the contents of ``data_gen.rs``."""
    lines = [
        '"""Index -> game key tables generated from sim/src/data_gen.rs by',
        "bridge/scripts/gen_codes.py. Do not edit by hand.",
        '"""',
        "",
    ]
    for enum, all_name, defs_name, py in TABLES:
        body = block(text, rf"pub enum {enum} \{{")
        disc = {name: int(v) for name, v in re.findall(r"(\w+)\s*=\s*(\d+)", body)}
        defs = block(text, rf"pub const {defs_name}: \[\w+; \d+\] = \[")
        keys = re.findall(r'key: "([^"]+)"', defs)
        if all_name:
            arr = block(text, rf"pub const {all_name}: \[{enum}; \d+\] = \[")
            order = re.findall(rf"{enum}::(\w+)", arr)
            out = [keys[disc[v]] for v in order]
        else:
            out = [keys[disc[n]] for n in sorted(disc, key=disc.get)]
        lines.append(f"{py}: tuple[str, ...] = (")
        lines += [f'    "{k}",' for k in out]
        lines.append(")")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--check"]
    src = Path(args[0]) if args else SRC
    want = render(src.read_text(encoding="utf-8"))
    if "--check" in sys.argv[1:]:
        if OUT.read_text(encoding="utf-8") != want:
            raise SystemExit(f"{OUT} is stale; rerun without --check")
        print(f"{OUT} is up to date")
        return
    OUT.write_text(want, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
