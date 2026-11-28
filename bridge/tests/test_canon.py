"""The canonical state of the game's dumps."""

from antezero_bridge import canon


def _hand_card(flag, **fields):
    return {"ability": {"set": "Default", "forced_selection": flag}, **fields}


def test_a_card_with_the_forced_flag_reads_as_held_only_when_highlighted():
    hand = [
        _hand_card(True, highlighted=True),
        _hand_card(True, highlighted=False),
        _hand_card(None, highlighted=True),
        # a dump of a mod before 0.3.0 has no highlights
        _hand_card(True),
        _hand_card(True, highlighted=1),
    ]
    state = canon.from_lua({"hand": hand})
    forced = [state[f"hand.{i}.forced"] for i in range(len(hand))]
    assert forced == ["true", "flagged", "false", "?", "?"]
