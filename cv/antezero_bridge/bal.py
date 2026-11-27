"""AnteZero's `.bal` recordings (`sim/src/balatro/script.rs`): a header line and one action per line."""

from __future__ import annotations

import antezero_sim

# Verbs a recording spells differently from an action code.
_VERBS = {"sell_consumable": "sell_cons"}
_SORTS = {"sort 0": "sort_rank", "sort 1": "sort_suit"}
# Codes whose last operand is a mask of hand positions, which a recording lists one by one.
_MASKED = {"play", "discard", "use", "buy_use", "pick"}


def line(code: str) -> str:
    """The recording line of an action code of `antezero_sim.Game`."""
    if code in _SORTS:
        return _SORTS[code]
    verb, *operands = code.split()
    if verb in _MASKED:
        mask = int(operands.pop())
        operands += [str(i) for i in range(mask.bit_length()) if mask >> i & 1]
    return " ".join([_VERBS.get(verb, verb), *operands])


def script(seed: str, back: str, stake: int, codes: list[str]) -> str:
    return "\n".join([f"seed {seed} deck {back} stake {stake}", *map(line, codes)]) + "\n"


def check(text: str) -> None:
    """Raises `ValueError` unless the simulator replays every line of the recording."""
    actions = sum(bool(row.split("#")[0].strip()) for row in text.splitlines()) - 1
    antezero_sim.StartPool().add(text, [actions])
