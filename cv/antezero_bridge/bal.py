"""AnteZero's `.bal` recordings (`sim/src/balatro/script.rs`): a header line and one action per line."""

from __future__ import annotations

import antezero_sim

# Verbs a recording spells differently from an action code.
_VERBS = {"sell_consumable": "sell_cons"}
_SORTS = {"sort 0": "sort_rank", "sort 1": "sort_suit"}
# Codes whose last operand is a mask of hand positions, which a recording lists one by one.
_MASKED = {"play", "discard", "use", "buy_use", "pick"}
# What a header ends in if its run played by Steamodded's rules; vanilla's otherwise.
_STEAMODDED = ["rules", "steamodded"]


def line(code: str) -> str:
    """The recording line of an action code of `antezero_sim.Game`."""
    if code in _SORTS:
        return _SORTS[code]
    verb, *operands = code.split()
    if verb in _MASKED:
        mask = int(operands.pop())
        operands += [str(i) for i in range(mask.bit_length()) if mask >> i & 1]
    return " ".join([_VERBS.get(verb, verb), *operands])


def script(seed: str, back: str, stake: int, codes: list[str], *, steamodded: bool,
           comments: dict[int, str] | None = None) -> str:
    """The recording of `codes` played in a new run, which followed Steamodded's rules or vanilla's.

    `comments` are written after the lines of the codes at their places.
    """
    rows = [" ".join(["seed", seed, "deck", back, "stake", str(stake), *(_STEAMODDED if steamodded else [])])]
    for at, code in enumerate(codes):
        rows.append(line(code) + (f"  # {comments[at]}" if comments and at in comments else ""))
    return "\n".join(rows) + "\n"


def start(text: str) -> antezero_sim.Game:
    """The new run of a recording's header, playing by the rules it names; `ValueError` if it is not a header."""
    return antezero_sim.Game.from_recording(text, 0)


def check(text: str) -> None:
    """Raises `ValueError` unless the simulator replays every line of the recording."""
    actions = sum(bool(row.split("#")[0].strip()) for row in text.splitlines()) - 1
    try:
        antezero_sim.StartPool().add(text, [actions])
    except ValueError as e:
        # A finished run replays to its last line and has no start point after it.
        if str(e) != f"the run is over after line {actions}":
            raise


def code(row: str) -> str:
    """The action code of a recording line (the inverse of `line`)."""
    words = row.split("#")[0].split()
    for action, name in _SORTS.items():
        if words == [name]:
            return action
    name, *operands = words
    action = next((a for a, n in _VERBS.items() if n == name), name)
    if action in _MASKED:
        slots = 0 if action in ("play", "discard") else 1
        mask = sum(1 << int(i) for i in operands[slots:])
        operands = [*operands[:slots], str(mask)]
    return " ".join([action, *operands])


def legal_code(game: antezero_sim.Game, row: str) -> str:
    """The code that plays a recording line in `game`.

    A recording keeps the hand cards that were highlighted when a card was used or picked; the
    game ignores them for a card that takes no targets, and `Game.step` refuses them.
    """
    action = code(row)
    name, *operands = action.split()
    if name in ("use", "buy_use", "pick") and operands[-1] != "0":
        probe = game.clone()
        try:
            probe.step(action)
        except RuntimeError:
            return " ".join([name, *operands[:-1], "0"])
    return action
