"""Template matching for Balatro deck backs on 1080p VOD frames.

Uses left-lower slices of the 15 playable deck-back sprites in
``balatro_source/resources/textures/2x/Enhancers.png`` so matching stays invariant to
DrSpectred's bottom-right avatars (which often cover the right 45-64% and top edge of
the deck stack) and ``View Deck`` hover text. Intended to run on a small sample of
frames when video metadata lacks an unambiguous deck, never frame-by-frame in ``main.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from game.balatro_models import Deck

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENHANCERS_PATH = REPO_ROOT / "balatro_source" / "resources" / "textures" / "2x" / "Enhancers.png"

_TILE_W = 142
_TILE_H = 190
_FRAME_W = 1920
_FRAME_H = 1080
_SCALE_1080P = 1.152
_SPRITE_INSET = 2
_SLICE_X_FRAC = (0.0, 0.36)
_SLICE_Y_FRAC = (0.16, 0.96)

# Bottom-right deck stack ROI in 1920x1080 coordinates: frame[y1:y2, x1:x2]
_ROI_X1 = 1540
_ROI_X2 = 1685
_ROI_Y1 = 780
_ROI_Y2 = 1045

# Interior slice inside resized template (skips left 10px & bottom 8px white card border)
_LAB_BORDER_LEFT = 10
_LAB_BORDER_BOTTOM = 8

# Frame acceptance thresholds
_MIN_SQ_SCORE = 0.90
_MIN_CC_SCORE = 0.75
_MIN_COMB_SCORE = 0.74
_MIN_MARGIN = 0.08

# Exact playable deck sprite grid coordinates from balatro_source/game.lua:628-642
PLAYABLE_DECK_TILE_COORDS: dict[Deck, tuple[int, int]] = {
    Deck.RED_DECK: (0, 0),
    Deck.BLUE_DECK: (0, 2),
    Deck.YELLOW_DECK: (1, 2),
    Deck.GREEN_DECK: (2, 2),
    Deck.BLACK_DECK: (3, 2),
    Deck.MAGIC_DECK: (0, 3),
    Deck.NEBULA_DECK: (3, 0),
    Deck.GHOST_DECK: (6, 2),
    Deck.ABANDONED_DECK: (3, 3),
    Deck.CHECKERED_DECK: (1, 3),
    Deck.ZODIAC_DECK: (3, 4),
    Deck.PAINTED_DECK: (4, 3),
    Deck.ANAGLYPH_DECK: (2, 4),
    Deck.PLASMA_DECK: (4, 2),
    Deck.ERRATIC_DECK: (2, 3),
}


@dataclass(frozen=True, slots=True)
class DeckFrameMatch:
    """Result of template-matching the bottom-right deck stack in one video frame."""

    deck: Deck | None
    best_deck: Deck
    runner_up: Deck
    accepted: bool
    score: float
    sq_score: float
    cc_score: float
    lab_de: float
    margin: float
    top_left: tuple[int, int]


@dataclass(frozen=True, slots=True)
class DeckVideoMatch:
    """Consensus deck detection across a sparse sample of video frames."""

    deck: Deck
    confidence: float
    mean_margin: float
    matched_frames: int
    sampled_frames: int
    votes: dict[str, int]
    best_frame_idx: int


def _resolve_enhancers_path(enhancers_path: str | Path | None) -> Path:
    path = Path(enhancers_path) if enhancers_path is not None else DEFAULT_ENHANCERS_PATH
    return path.resolve()


@lru_cache(maxsize=4)
def _load_deck_templates(resolved_path: Path) -> dict[Deck, tuple[np.ndarray, np.ndarray]]:
    """Builds cached `(bgr_template, blurred_lab_template)` pairs for all 15 playable decks."""
    enh = cv2.imread(str(resolved_path), cv2.IMREAD_UNCHANGED)
    if enh is None or enh.ndim != 3 or enh.shape[2] < 4:
        raise FileNotFoundError(
            f"Could not load RGBA Enhancers sprite sheet from {resolved_path}"
        )

    templates: dict[Deck, tuple[np.ndarray, np.ndarray]] = {}
    for deck, (cx, cy) in PLAYABLE_DECK_TILE_COORDS.items():
        tile = enh[cy * _TILE_H : (cy + 1) * _TILE_H, cx * _TILE_W : (cx + 1) * _TILE_W]
        alpha = tile[:, :, 3]
        ys, xs = np.where(alpha > 200)
        if ys.size == 0 or xs.size == 0:
            raise ValueError(f"Empty alpha mask for deck tile {deck.value} at ({cx}, {cy})")

        ox0 = int(xs.min()) + _SPRITE_INSET
        ox1 = int(xs.max()) + 1 - _SPRITE_INSET
        oy0 = int(ys.min()) + _SPRITE_INSET
        oy1 = int(ys.max()) + 1 - _SPRITE_INSET
        ow = ox1 - ox0
        oh = oy1 - oy0

        cx0 = ox0 + int(round(ow * _SLICE_X_FRAC[0]))
        cx1 = ox0 + int(round(ow * _SLICE_X_FRAC[1]))
        cy0 = oy0 + int(round(oh * _SLICE_Y_FRAC[0]))
        cy1 = oy0 + int(round(oh * _SLICE_Y_FRAC[1]))

        crop_bgr = tile[cy0:cy1, cx0:cx1, :3]
        nw = max(8, int(round(crop_bgr.shape[1] * _SCALE_1080P)))
        nh = max(8, int(round(crop_bgr.shape[0] * _SCALE_1080P)))
        res_bgr = np.ascontiguousarray(cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA))
        blur_bgr = cv2.GaussianBlur(res_bgr, (5, 5), 0)
        res_lab = np.ascontiguousarray(cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2LAB).astype(np.float32))
        templates[deck] = (res_bgr, res_lab)
    return templates


def _score_patch(patch_bgr: np.ndarray, tpl_bgr: np.ndarray, tpl_lab: np.ndarray) -> tuple[float, float, float, float]:
    sq = 1.0 - float(cv2.matchTemplate(patch_bgr, tpl_bgr, cv2.TM_SQDIFF_NORMED)[0, 0])
    cc = float(cv2.matchTemplate(patch_bgr, tpl_bgr, cv2.TM_CCOEFF_NORMED)[0, 0])

    patch_blur = cv2.GaussianBlur(patch_bgr, (5, 5), 0)
    patch_lab = cv2.cvtColor(patch_blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    interior_patch = patch_lab[:-_LAB_BORDER_BOTTOM, _LAB_BORDER_LEFT:]
    interior_tpl = tpl_lab[:-_LAB_BORDER_BOTTOM, _LAB_BORDER_LEFT:]
    diff = interior_patch - interior_tpl
    de = np.sqrt((0.6 * diff[..., 0]) ** 2 + (1.4 * diff[..., 1]) ** 2 + (1.4 * diff[..., 2]) ** 2)

    flat = np.sort(de.ravel())
    keep_n = max(1, int(len(flat) * 0.85))
    trimmed_de = float(flat[:keep_n].mean())
    lab_sim = max(0.0, 1.0 - trimmed_de / 60.0)
    comb = 0.45 * lab_sim + 0.35 * sq + 0.20 * max(0.0, cc)
    return comb, sq, cc, trimmed_de


def match_deck_in_frame(
    frame_bgr: np.ndarray,
    *,
    enhancers_path: str | Path | None = None,
) -> DeckFrameMatch:
    """Matches the left-lower deck back slice in the bottom-right corner of a frame."""
    if frame_bgr is None or frame_bgr.ndim != 3:
        raise ValueError("Expected a 3-channel BGR frame")

    h, w = frame_bgr.shape[:2]
    if (w, h) != (_FRAME_W, _FRAME_H):
        frame_bgr = cv2.resize(frame_bgr, (_FRAME_W, _FRAME_H), interpolation=cv2.INTER_AREA)

    roi = frame_bgr[_ROI_Y1:_ROI_Y2, _ROI_X1:_ROI_X2]
    templates = _load_deck_templates(_resolve_enhancers_path(enhancers_path))

    scored: list[tuple[float, Deck, float, float, float, tuple[int, int]]] = []
    for deck, (tpl_bgr, tpl_lab) in templates.items():
        r_sq = cv2.matchTemplate(roi, tpl_bgr, cv2.TM_SQDIFF_NORMED)
        _, _, (px, py), _ = cv2.minMaxLoc(r_sq)
        th_tpl, tw_tpl = tpl_bgr.shape[:2]
        patch = roi[py : py + th_tpl, px : px + tw_tpl]
        comb, sq, cc, trimmed_de = _score_patch(patch, tpl_bgr, tpl_lab)
        scored.append((comb, deck, sq, cc, trimmed_de, (_ROI_X1 + int(px), _ROI_Y1 + int(py))))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_comb, best_deck, best_sq, best_cc, best_de, best_loc = scored[0]
    second_comb, second_deck = scored[1][0], scored[1][1]
    margin = best_comb - second_comb

    accepted = (
        best_sq >= _MIN_SQ_SCORE
        and best_cc >= _MIN_CC_SCORE
        and best_comb >= _MIN_COMB_SCORE
        and margin >= _MIN_MARGIN
    )
    return DeckFrameMatch(
        deck=best_deck if accepted else None,
        best_deck=best_deck,
        runner_up=second_deck,
        accepted=accepted,
        score=float(best_comb),
        sq_score=float(best_sq),
        cc_score=float(best_cc),
        lab_de=float(best_de),
        margin=float(margin),
        top_left=best_loc,
    )


def _compute_sample_indices(
    total_frames: int,
    num_samples: int,
    start_frac: float,
    end_frac: float,
) -> list[int]:
    n_samples = max(1, int(num_samples))
    if total_frames < 50:
        count = min(n_samples, total_frames)
        raw = np.linspace(0, max(0, total_frames - 1), count)
    else:
        lo = max(0, min(total_frames - 1, int(total_frames * start_frac)))
        hi = max(lo, min(total_frames - 1, int(total_frames * end_frac)))
        count = min(n_samples, hi - lo + 1)
        raw = np.linspace(lo, hi, count)

    seen: set[int] = set()
    indices: list[int] = []
    for val in raw:
        idx = int(round(float(val)))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def detect_deck_from_video(
    video_path: str | Path,
    *,
    num_samples: int = 12,
    start_frac: float = 0.12,
    end_frac: float = 0.88,
    early_stop_votes: int = 4,
    min_votes: int = 2,
    min_agreement: float = 0.75,
    enhancers_path: str | Path | None = None,
) -> DeckVideoMatch | None:
    """Samples sparse frames across `video_path` to identify the player's starting deck."""
    path = Path(video_path)
    if not path.is_file():
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    sampled_frames = 0
    votes: dict[Deck, int] = {}
    matches_by_deck: dict[Deck, list[tuple[int, DeckFrameMatch]]] = {}

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            for idx in range(max(1, int(num_samples))):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                sampled_frames += 1
                res = match_deck_in_frame(frame, enhancers_path=enhancers_path)
                if res.accepted and res.deck is not None:
                    votes[res.deck] = votes.get(res.deck, 0) + 1
                    matches_by_deck.setdefault(res.deck, []).append((idx, res))
                    if len(votes) == 1 and votes[res.deck] >= early_stop_votes:
                        break
        else:
            indices = _compute_sample_indices(total_frames, num_samples, start_frac, end_frac)
            for fidx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                sampled_frames += 1
                res = match_deck_in_frame(frame, enhancers_path=enhancers_path)
                if res.accepted and res.deck is not None:
                    votes[res.deck] = votes.get(res.deck, 0) + 1
                    matches_by_deck.setdefault(res.deck, []).append((fidx, res))
                    if len(votes) == 1 and votes[res.deck] >= early_stop_votes:
                        break
    finally:
        cap.release()

    if not votes or sampled_frames == 0:
        return None

    total_accepted = sum(votes.values())
    winner = max(
        votes,
        key=lambda d: (votes[d], sum(m.score for _, m in matches_by_deck[d])),
    )
    winner_votes = votes[winner]
    winner_entries = matches_by_deck[winner]
    best_fidx, best_match = max(winner_entries, key=lambda pair: pair[1].score)

    effective_min_votes = min(min_votes, sampled_frames)
    single_strong_short = (
        sampled_frames <= 3
        and total_accepted == 1
        and best_match.sq_score >= 0.95
        and best_match.cc_score >= 0.88
        and best_match.margin >= 0.14
    )
    if winner_votes < effective_min_votes and not single_strong_short:
        return None
    if (winner_votes / total_accepted) < min_agreement:
        return None

    mean_conf = float(np.mean([m.score for _, m in winner_entries]))
    mean_margin = float(np.mean([m.margin for _, m in winner_entries]))
    return DeckVideoMatch(
        deck=winner,
        confidence=mean_conf,
        mean_margin=mean_margin,
        matched_frames=winner_votes,
        sampled_frames=sampled_frames,
        votes={d.value: c for d, c in votes.items()},
        best_frame_idx=int(best_fidx),
    )


__all__ = [
    "DEFAULT_ENHANCERS_PATH",
    "DeckFrameMatch",
    "DeckVideoMatch",
    "PLAYABLE_DECK_TILE_COORDS",
    "detect_deck_from_video",
    "match_deck_in_frame",
]
