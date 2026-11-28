"""The canonical state of the game's dumps."""

from antezero_bridge import canon


def _hand_card(flag, **fields):
    return {"ability": {"set": "Default", "forced_selection": flag}, **fields}


def test_a_card_with_the_forced_flag_reads_as_held_only_when_highlighted():
    hand = [
        _hand_card(True, highlighted=1),
        _hand_card(True, highlighted=False),
        _hand_card(None, highlighted=1),
        # a dump of a mod before 0.3.1 has no highlight places
        _hand_card(True),
        _hand_card(True, highlighted=True),
        _hand_card(True, highlighted=0),
        _hand_card(True, highlighted=1.5),
        _hand_card(True, highlighted="inf"),
        _hand_card(True, highlighted=float("inf")),
        _hand_card(True, highlighted=10**400),
    ]
    state = canon.from_lua({"hand": hand})
    forced = [state[f"hand.{i}.forced"] for i in range(len(hand))]
    assert forced == ["true", "flagged", "false", "?", "?", "?", "?", "?", "?", "true"]
