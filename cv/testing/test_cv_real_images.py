"""Real-image integration test suite scaffolding for all CV detectors.

Evaluates:
  - `cv_detect_hand_cards`
  - `cv_detect_shop_cards`
  - `cv_detect_pack_cards`
  - `cv_detect_joker_bar`

Screenshots and expected outputs are user-supplied in:
  testing/real_image_fixtures/ground_truth.json
"""

from __future__ import annotations

import difflib
import json
import re
import unittest
from pathlib import Path
from typing import Any

import cv2

import game.balatro_models as bm
import game.engine.cv_stubs as cv_stubs
import game.gamestate as gs
from testing.visualize_cv_failures import visualize_cv_failure


FIXTURES_DIR = Path(__file__).resolve().parent / "real_image_fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

REQUIRED_SECTIONS = ("hand", "shop", "pack", "joker_bar")


def _norm(value: Any) -> str:
    """Normalizes enum names and human labels for flexible ground-truth comparison."""
    text = value.value if hasattr(value, "value") else str(value)
    return re.sub(r"[^a-z0-9]", "", text.strip().lower())


def _val_str(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _is_none_val(value: Any) -> bool:
    return value is None or _norm(value) in ("", "none")


def _fmt_hand_card(item: bm.HandCard | dict[str, Any]) -> str:
    if isinstance(item, dict):
        facing_back = bool(item.get("facing_back", False))
        rank = item.get("rank")
        suit = item.get("suit")
        enh = item.get("enhancement")
        ed = item.get("edition")
        seal = item.get("seal")
        selected = item.get("selected")
        debuffed = item.get("debuffed")
    else:
        facing_back = bool(item.facing_back)
        rank = _val_str(item.card.rank)
        suit = _val_str(item.card.suit)
        enh = _val_str(item.card.enhancement)
        ed = _val_str(item.card.edition)
        seal = _val_str(item.card.seal)
        selected = bool(item.selected)
        debuffed = bool(item.card.debuffed)

    if facing_back:
        base = "Face Down"
    elif rank and suit:
        base = f"{rank} of {suit}"
    else:
        base = rank or suit or "Card"

    tags: list[str] = []
    if not _is_none_val(enh):
        tags.append(_val_str(enh))
    if not _is_none_val(ed):
        tags.append(_val_str(ed))
    if not _is_none_val(seal):
        tags.append(f"{_val_str(seal)} seal")
    if selected:
        tags.append("selected")
    if debuffed:
        tags.append("debuffed")
    return f"{base} [{', '.join(tags)}]" if tags else base


def _fmt_shop_item(item: Any) -> str:
    if isinstance(item, dict):
        item_type = item.get("type")
        name = item.get("name")
        booster_type = item.get("booster_type")
        size = item.get("size")
        rank = item.get("rank")
        suit = item.get("suit")
        enh = item.get("enhancement")
        ed = item.get("edition")
        seal = item.get("seal")
        flags = [f for f in ("eternal", "perishable", "rental", "debuffed") if item.get(f)]
    else:
        item_type = type(item).__name__
        name = getattr(item, "name", None)
        booster_type = getattr(item, "booster_type", None)
        size = getattr(item, "size", None)
        rank = getattr(item, "rank", None)
        suit = getattr(item, "suit", None)
        enh = getattr(item, "enhancement", None)
        ed = getattr(item, "edition", None)
        seal = getattr(item, "seal", None)
        flags = [f for f in ("eternal", "perishable", "rental", "debuffed") if getattr(item, f, False)]

    if booster_type or size:
        base = f"{_val_str(size or '')} {_val_str(booster_type or '')} Booster".strip()
    elif rank or suit:
        base = f"{_val_str(rank or '?')} of {_val_str(suit or '?')}"
    elif name:
        base = f"{name} ({item_type})" if item_type and item_type != "Joker" else str(name)
    else:
        base = str(item_type or "Unknown")

    tags: list[str] = []
    if not _is_none_val(ed):
        tags.append(_val_str(ed))
    if not _is_none_val(enh):
        tags.append(_val_str(enh))
    if not _is_none_val(seal):
        tags.append(f"{_val_str(seal)} seal")
    tags.extend(flags)
    return f"{base} [{', '.join(tags)}]" if tags else base


def _hand_card_id(item: bm.HandCard | dict[str, Any]) -> str:
    if isinstance(item, dict):
        if item.get("facing_back"):
            return "facedown"
        return f"{_norm(item.get('rank', ''))}_{_norm(item.get('suit', ''))}"
    if item.facing_back:
        return "facedown"
    return f"{_norm(item.card.rank)}_{_norm(item.card.suit)}"


def _shop_item_id(item: Any) -> str:
    if isinstance(item, dict):
        if "name" in item:
            return _norm(item["name"])
        if "booster_type" in item or "size" in item:
            return f"booster_{_norm(item.get('booster_type', ''))}_{_norm(item.get('size', ''))}"
        if "rank" in item or "suit" in item:
            return f"{_norm(item.get('rank', ''))}_{_norm(item.get('suit', ''))}"
        return _norm(item.get("type", "unknown"))

    name = getattr(item, "name", None)
    if name:
        return _norm(name)
    booster_type = getattr(item, "booster_type", None)
    size = getattr(item, "size", None)
    if booster_type or size:
        return f"booster_{_norm(booster_type or '')}_{_norm(size or '')}"
    rank = getattr(item, "rank", None)
    suit = getattr(item, "suit", None)
    if rank or suit:
        return f"{_norm(rank or '')}_{_norm(suit or '')}"
    return _norm(type(item).__name__)


def _hand_card_mismatches(actual: bm.HandCard, expected: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    checks = [
        ("rank", _norm(actual.card.rank), _val_str(actual.card.rank)),
        ("suit", _norm(actual.card.suit), _val_str(actual.card.suit)),
        ("enhancement", _norm(actual.card.enhancement), _val_str(actual.card.enhancement)),
        ("edition", _norm(actual.card.edition), _val_str(actual.card.edition)),
        ("seal", _norm(actual.card.seal), _val_str(actual.card.seal)),
    ]
    for field, act_norm, act_disp in checks:
        if field in expected and act_norm != _norm(expected[field]):
            diffs.append(f"{field}: expected '{expected[field]}', got '{act_disp}'")

    bool_checks = [
        ("selected", bool(actual.selected)),
        ("debuffed", bool(actual.card.debuffed)),
        ("facing_back", bool(actual.facing_back)),
    ]
    for field, act_bool in bool_checks:
        if field in expected and act_bool != bool(expected[field]):
            diffs.append(f"{field}: expected {bool(expected[field])}, got {act_bool}")
    return diffs


def _shop_item_mismatches(actual: Any, expected: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    if "type" in expected:
        act_type = type(actual).__name__
        if _norm(act_type) != _norm(expected["type"]):
            diffs.append(f"type: expected '{expected['type']}', got '{act_type}'")
    if "name" in expected:
        act_name = getattr(actual, "name", None)
        if act_name is None or _norm(act_name) != _norm(expected["name"]):
            diffs.append(f"name: expected '{expected['name']}', got '{act_name}'")

    enum_fields = [
        ("edition", bm.Edition.NONE),
        ("booster_type", ""),
        ("size", ""),
        ("rank", bm.Rank.NONE),
        ("suit", bm.Suit.NONE),
        ("enhancement", bm.Enhancement.NONE),
        ("seal", bm.Seal.NONE),
    ]
    for field, default in enum_fields:
        if field in expected:
            act_val = getattr(actual, field, default)
            if _norm(act_val) != _norm(expected[field]):
                diffs.append(f"{field}: expected '{expected[field]}', got '{_val_str(act_val)}'")

    for field in ("eternal", "perishable", "rental", "debuffed"):
        if field in expected:
            act_bool = bool(getattr(actual, field, False))
            if act_bool != bool(expected[field]):
                diffs.append(f"{field}: expected {bool(expected[field])}, got {act_bool}")
    return diffs


def _load_ground_truth() -> dict[str, Any]:
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_frame(rel_path: str) -> Any:
    img_path = FIXTURES_DIR / rel_path
    if not img_path.exists():
        raise FileNotFoundError(
            f"Ground-truth image missing: '{img_path}'. Place the screenshot file there."
        )
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise ValueError(f"OpenCV failed to decode image '{img_path}'.")
    return frame


def _location_from_label(label: str | None, default: bm.Location) -> bm.Location:
    if not label:
        return default
    key = _norm(label)
    for loc in bm.Location:
        if _norm(loc.value) == key or _norm(loc.name) == key:
            return loc
    raise ValueError(f"Unknown Location '{label}' in ground_truth.json")


class TestRealImageCVFixtures(unittest.TestCase):
    """Validates CV stubs against user-provided screenshots and JSON ground truth."""

    def _assert_item_sequence(
        self,
        actual_items: list[Any],
        expected_items: list[dict[str, Any]],
        mismatch_fn: Any,
        fmt_fn: Any,
        id_fn: Any,
        ctx: str,
        *,
        vis_info: tuple[str, str, Any, float, str | None] | None = None,
    ) -> None:
        exp_ids = [id_fn(item) for item in expected_items]
        act_ids = [id_fn(item) for item in actual_items]

        matcher = difflib.SequenceMatcher(None, exp_ids, act_ids, autojunk=False)
        false_negatives: list[str] = []
        false_positives: list[str] = []
        substitutions: list[str] = []
        attr_mismatches: list[str] = []
        fp_indices: set[int] = set()
        warn_indices: set[int] = set()

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for exp_idx, act_idx in zip(range(i1, i2), range(j1, j2)):
                    diffs = mismatch_fn(actual_items[act_idx], expected_items[exp_idx])
                    if diffs:
                        warn_indices.add(act_idx)
                        label = fmt_fn(expected_items[exp_idx])
                        attr_mismatches.append(
                            f"  ~ [exp {exp_idx} == act {act_idx}] {label}: " + "; ".join(diffs)
                        )
            elif tag == "delete":
                for exp_idx in range(i1, i2):
                    false_negatives.append(f"  - [exp {exp_idx}] {fmt_fn(expected_items[exp_idx])}")
            elif tag == "insert":
                for act_idx in range(j1, j2):
                    fp_indices.add(act_idx)
                    false_positives.append(f"  + [act {act_idx}] {fmt_fn(actual_items[act_idx])}")
            elif tag == "replace":
                paired = min(i2 - i1, j2 - j1)
                for k in range(paired):
                    exp_idx = i1 + k
                    act_idx = j1 + k
                    warn_indices.add(act_idx)
                    substitutions.append(
                        f"  ! [exp {exp_idx} -> act {act_idx}] expected {fmt_fn(expected_items[exp_idx])}, "
                        f"got {fmt_fn(actual_items[act_idx])}"
                    )
                for exp_idx in range(i1 + paired, i2):
                    false_negatives.append(f"  - [exp {exp_idx}] {fmt_fn(expected_items[exp_idx])}")
                for act_idx in range(j1 + paired, j2):
                    fp_indices.add(act_idx)
                    false_positives.append(f"  + [act {act_idx}] {fmt_fn(actual_items[act_idx])}")

        if not (false_negatives or false_positives or substitutions or attr_mismatches):
            return

        lines = [f"\n{ctx}: expected {len(expected_items)}, detected {len(actual_items)}"]
        if false_negatives:
            lines.append("False Negatives (missed):")
            lines.extend(false_negatives)
        if false_positives:
            lines.append("False Positives (extra):")
            lines.extend(false_positives)
        if substitutions:
            lines.append("Substitutions (wrong identity at slot):")
            lines.extend(substitutions)
        if attr_mismatches:
            lines.append("Attribute Mismatches:")
            lines.extend(attr_mismatches)

        lines.append("Expected:")
        lines.extend(f"  [{i}] {fmt_fn(item)}" for i, item in enumerate(expected_items))
        lines.append("Detected:")
        lines.extend(f"  [{i}] {fmt_fn(item)}" for i, item in enumerate(actual_items))

        summary = "\n".join(lines)
        if vis_info is not None:
            section, rel_path, frame, conf_threshold, shelf_key = vis_info
            try:
                vis_path = visualize_cv_failure(
                    section,
                    rel_path,
                    frame,
                    conf_threshold,
                    summary,
                    shelf_key=shelf_key,
                    fp_indices=fp_indices,
                    warn_indices=warn_indices,
                )
                summary += f"\nVisualization saved to: {vis_path}"
            except Exception as exc:
                summary += f"\n(Visualization failed: {exc})"

        self.fail(summary)

    def test_ground_truth_schema_is_valid(self):
        """Always verifies ground_truth.json structure and images directory exist."""
        self.assertTrue(GROUND_TRUTH_PATH.exists(), f"Missing {GROUND_TRUTH_PATH}")
        self.assertTrue(IMAGES_DIR.is_dir(), f"Missing folder {IMAGES_DIR}")
        data = _load_ground_truth()
        for section in REQUIRED_SECTIONS:
            # Each section's image paths are relative to FIXTURES_DIR (e.g. "images/...").
            self.assertIn(section, data, f"ground_truth.json missing '{section}' list")
            self.assertIsInstance(data[section], list, f"'{section}' must be a list")

    def test_cv_detect_hand_cards_real_images(self):
        cases = _load_ground_truth().get("hand", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['hand']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(
                    location=_location_from_label(case.get("location"), bm.Location.BLIND),
                    round_won=bool(case.get("round_won", False)),
                )
                conf = float(case.get("conf_threshold", 0.50))
                detected = cv_stubs.cv_detect_hand_cards(
                    state, frame, conf_threshold=conf
                )
                self._assert_item_sequence(
                    detected,
                    case["cards"],
                    _hand_card_mismatches,
                    _fmt_hand_card,
                    _hand_card_id,
                    f"hand:{rel_path}",
                    vis_info=("hand", rel_path, frame, conf, None),
                )

    def test_cv_detect_shop_cards_real_images(self):
        cases = _load_ground_truth().get("shop", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['shop']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(location=bm.Location.SHOP)
                conf = float(case.get("conf_threshold", 0.80))
                detected = cv_stubs.cv_detect_shop_cards(
                    state, frame, conf_threshold=conf
                )
                shelf_pairs = [
                    ("shop_cards", list(detected)),
                    ("vouchers", detected.vouchers),
                    ("packs", detected.packs),
                    ("joker_bar", detected.joker_bar),
                    ("consumables", detected.consumables),
                ]
                for key, actual_list in shelf_pairs:
                    if key in case:
                        self._assert_item_sequence(
                            actual_list,
                            case[key],
                            _shop_item_mismatches,
                            _fmt_shop_item,
                            _shop_item_id,
                            f"shop:{rel_path}:{key}",
                            vis_info=("shop", rel_path, frame, conf, key),
                        )

    def test_cv_detect_pack_cards_real_images(self):
        cases = _load_ground_truth().get("pack", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['pack']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(
                    location=_location_from_label(case.get("location"), bm.Location.PACK)
                )
                conf = float(case.get("conf_threshold", 0.80))
                detected = cv_stubs.cv_detect_pack_cards(
                    state, frame, conf_threshold=conf
                )
                self._assert_item_sequence(
                    detected,
                    case["cards"],
                    _shop_item_mismatches,
                    _fmt_shop_item,
                    _shop_item_id,
                    f"pack:{rel_path}",
                    vis_info=("pack", rel_path, frame, conf, None),
                )

    def test_cv_detect_joker_bar_real_images(self):
        cases = _load_ground_truth().get("joker_bar", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['joker_bar']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(
                    location=_location_from_label(case.get("location"), bm.Location.BLIND),
                    round_won=bool(case.get("round_won", False)),
                )
                conf = float(case.get("conf_threshold", 0.80))
                detected = cv_stubs.cv_detect_joker_bar(
                    state, frame, conf_threshold=conf
                )
                self._assert_item_sequence(
                    detected,
                    case["jokers"],
                    _shop_item_mismatches,
                    _fmt_shop_item,
                    _shop_item_id,
                    f"joker_bar:{rel_path}",
                    vis_info=("joker_bar", rel_path, frame, conf, None),
                )


if __name__ == "__main__":
    unittest.main()
