"""YouTube metadata fetching and Balatro deck/seed/stake extraction."""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

from game.balatro_models import Deck, Stake

# Exact random seed alphabet from balatro_source/functions/misc_functions.lua:random_string
# Note: '0' and 'O' are excluded by Balatro's RNG seed generator.
BALATRO_SEED_ALPHABET: str = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
_SEED_CHARS_CLASS = "1-9A-NP-Z"
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_VERSION_RE = re.compile(r"\((?:Balatro\s+)?Version\s+([^)\r\n]+)\)", re.IGNORECASE)

_DECK_MAP: dict[str, Deck] = {
    d.value.removesuffix(" Deck").lower(): d
    for d in Deck
    if d != Deck.CHALLENGE_DECK
}
_DECK_NAMES_ALT = "|".join(
    sorted((re.escape(k) for k in _DECK_MAP), key=len, reverse=True)
)
_EXPLICIT_DECK_RE = re.compile(
    rf"\b({_DECK_NAMES_ALT})\s+Deck\b", re.IGNORECASE
)
_DECK_KEY_VALUE_RE = re.compile(
    rf"\bDeck\s*[:\-]\s*({_DECK_NAMES_ALT})(?:\s+Deck)?\b", re.IGNORECASE
)

_STAKE_MAP: dict[str, Stake] = {
    s.value.removesuffix(" Stake").lower(): s for s in Stake
}
_STAKE_NAMES_ALT = "|".join(
    sorted((re.escape(k) for k in _STAKE_MAP), key=len, reverse=True)
)
_EXPLICIT_STAKE_RE = re.compile(
    rf"\b({_STAKE_NAMES_ALT})\s+Stake\b", re.IGNORECASE
)
_STAKE_KEY_VALUE_RE = re.compile(
    rf"\bStake\s*[:\-]\s*({_STAKE_NAMES_ALT})(?:\s+Stake)?\b", re.IGNORECASE
)

# Labelled seed header: handles "Seed: VUCDKYZ6", "Seed:\n\nBZTGPV12", "Erratic Deck:\n\nZ1DCPJSE"
_LABELED_SEED_RE = re.compile(
    rf"(?:\bSeed|(?:{_DECK_NAMES_ALT})\s+Deck)\s*[:\-]\s*[\r\n]*\s*([A-Z0-9]{{8}})\b",
    re.IGNORECASE,
)
# Standalone line containing solely an 8-char valid Balatro seed token (must include >=1 digit)
_STANDALONE_HEADER_SEED_RE = re.compile(
    rf"^\s*([{_SEED_CHARS_CLASS}]{{8}})\s*$", re.MULTILINE
)

_BOILERPLATE_MARKERS: tuple[str, ...] = (
    "in balatro, when you complete",
    "daily gold stake gameplay",
    "mod list/info:",
    "playlists:",
    "balatrou has a bunch",
    "you can get the balatrou",
    "for more information",
)

# Keep compact subset of raw yt-dlp JSON alongside extracted fields so saved metadata files stay small
_RAW_SUMMARY_KEYS: tuple[str, ...] = (
    "id",
    "webpage_url",
    "original_url",
    "title",
    "description",
    "upload_date",
    "timestamp",
    "duration",
    "channel",
    "uploader",
    "channel_id",
    "playlist_index",
    "tags",
    "categories",
    "chapters",
)


@dataclass
class ExtractedRunMetadata:
    """Structured run parameters and gating decision extracted from a VOD's metadata."""

    url: str
    video_id: str
    title: str = ""
    upload_date: str = ""
    duration_seconds: float | None = None
    channel: str = ""
    balatro_version: str = ""
    deck: Deck | None = None
    seed: str = ""
    stake: Stake = Stake.GOLD
    deck_source: str = ""
    seed_source: str = ""
    stake_source: str = "default_gold"
    ready: bool = False
    skip_reason: str | None = None
    fetch_error: str | None = None

    @property
    def deck_cli_value(self) -> str:
        """CLI `--deck` string accepted by `post_processing.parse_deck`."""
        if self.deck is None:
            return ""
        return self.deck.name.removesuffix("_DECK").lower()

    @property
    def stake_cli_value(self) -> str:
        """CLI `--stake` string accepted by `post_processing.parse_stake`."""
        return self.stake.name.lower()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["deck"] = self.deck.value if self.deck is not None else None
        data["deck_cli"] = self.deck_cli_value or None
        data["stake"] = self.stake.value
        data["stake_cli"] = self.stake_cli_value
        return data


def extract_video_id(url: str) -> str:
    """Extracts a filesystem-safe video ID from a YouTube URL (or fallback hash/slug)."""
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    path = parsed.path.strip("/")
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return re.sub(r"[^A-Za-z0-9_-]", "_", qs["v"][0])
    if "youtu.be" in host and path:
        return re.sub(r"[^A-Za-z0-9_-]", "_", path.split("/")[0])
    if "/shorts/" in parsed.path or "/live/" in parsed.path:
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            return re.sub(r"[^A-Za-z0-9_-]", "_", parts[-1])
    stem = Path(path).stem if path else ""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", stem or url)
    return safe[:64] or "video"


def _strip_urls(text: str) -> str:
    return _URL_RE.sub(" ", text or "")


def _split_description_header(description: str) -> tuple[str, str]:
    """Splits description into pre-boilerplate header and remaining body."""
    text = description or ""
    lower = text.lower()
    cut_idx = len(text)
    for marker in _BOILERPLATE_MARKERS:
        pos = lower.find(marker)
        if pos != -1 and pos < cut_idx:
            cut_idx = pos
    header = text[:cut_idx]
    body = text[cut_idx:]
    return header, body


def _find_decks_in_text(text: str, *, allow_key_value: bool = True) -> list[Deck]:
    found: list[Deck] = []
    for match in _EXPLICIT_DECK_RE.finditer(text):
        deck = _DECK_MAP.get(match.group(1).lower())
        if deck is not None and deck not in found:
            found.append(deck)
    if allow_key_value:
        for match in _DECK_KEY_VALUE_RE.finditer(text):
            deck = _DECK_MAP.get(match.group(1).lower())
            if deck is not None and deck not in found:
                found.append(deck)
    return found


def _find_stakes_in_text(text: str, *, allow_key_value: bool = True) -> list[Stake]:
    found: list[Stake] = []
    for match in _EXPLICIT_STAKE_RE.finditer(text):
        stake = _STAKE_MAP.get(match.group(1).lower())
        if stake is not None and stake not in found:
            found.append(stake)
    if allow_key_value:
        for match in _STAKE_KEY_VALUE_RE.finditer(text):
            stake = _STAKE_MAP.get(match.group(1).lower())
            if stake is not None and stake not in found:
                found.append(stake)
    return found


def _extract_deck(
    title: str, header_clean: str, full_clean: str
) -> tuple[Deck | None, str, str | None]:
    header_decks = _find_decks_in_text(header_clean, allow_key_value=True)
    title_decks = _find_decks_in_text(title, allow_key_value=True)

    if len(header_decks) > 1:
        return None, "header", "ambiguous_multiple_decks"
    if len(title_decks) > 1:
        return None, "title", "ambiguous_multiple_decks"
    if header_decks and title_decks and header_decks[0] != title_decks[0]:
        return None, "header_vs_title", "ambiguous_deck_conflict"

    if header_decks:
        return header_decks[0], "description_header", None
    if title_decks:
        return title_decks[0], "title", None

    body_decks = _find_decks_in_text(full_clean, allow_key_value=True)
    if len(body_decks) > 1:
        return None, "description_body", "ambiguous_multiple_decks"
    if len(body_decks) == 1:
        return body_decks[0], "description_body", None
    return None, "", None


def _extract_seed(header_clean: str, full_clean: str) -> tuple[str, str, str | None]:
    # 1. Explicit "Seed: XXXXXXXX" or "<Deck> Deck:\n\nXXXXXXXX" in header (then full clean text)
    labeled_header = [m.group(1).upper() for m in _LABELED_SEED_RE.finditer(header_clean)]
    unique_lh = list(dict.fromkeys(labeled_header))
    if len(unique_lh) > 1:
        return "", "description_header", "ambiguous_multiple_seeds"
    if len(unique_lh) == 1:
        return unique_lh[0], "description_header_labeled", None

    # 2. Standalone 8-char random Balatro seed line in header (requires >=1 digit to reject words)
    standalone: list[str] = []
    for m in _STANDALONE_HEADER_SEED_RE.finditer(header_clean):
        tok = m.group(1)
        if any(ch.isdigit() for ch in tok) and tok not in standalone:
            standalone.append(tok)
    if len(standalone) > 1:
        return "", "description_header", "ambiguous_multiple_seeds"
    if len(standalone) == 1:
        return standalone[0], "description_header_standalone", None

    # 3. Fallback: explicitly labeled seed anywhere in non-URL description
    labeled_any = list(dict.fromkeys(m.group(1).upper() for m in _LABELED_SEED_RE.finditer(full_clean)))
    if len(labeled_any) > 1:
        return "", "description_body", "ambiguous_multiple_seeds"
    if len(labeled_any) == 1:
        return labeled_any[0], "description_body_labeled", None
    return "", "", None


def _extract_stake(title: str, header_clean: str, body_clean: str) -> tuple[Stake, str]:
    title_stakes = _find_stakes_in_text(title, allow_key_value=True)
    if title_stakes:
        return title_stakes[0], "title"

    header_stakes = _find_stakes_in_text(header_clean, allow_key_value=True)
    if header_stakes:
        return header_stakes[0], "description_header"

    if re.search(r"\bC\+\+(?=\W|$)", title):
        return Stake.GOLD, "title_c_plus_plus"

    body_stakes = _find_stakes_in_text(body_clean, allow_key_value=True)
    if body_stakes:
        return body_stakes[0], "description_body"

    return Stake.GOLD, "default_gold"


def extract_run_metadata(raw_meta: dict[str, Any], url: str = "") -> ExtractedRunMetadata:
    """Extracts Balatro `(deck, seed, stake)` and gating state from a yt-dlp metadata dict."""
    resolved_url = url or str(raw_meta.get("webpage_url") or raw_meta.get("original_url") or "")
    vid = str(raw_meta.get("id") or extract_video_id(resolved_url))
    title = str(raw_meta.get("title") or "")
    description = str(raw_meta.get("description") or "")
    upload_date = str(raw_meta.get("upload_date") or "")
    duration_val = raw_meta.get("duration")
    duration = float(duration_val) if isinstance(duration_val, (int, float)) else None
    channel = str(raw_meta.get("channel") or raw_meta.get("uploader") or "")

    if raw_meta.get("fetch_error"):
        return ExtractedRunMetadata(
            url=resolved_url,
            video_id=vid,
            title=title,
            ready=False,
            skip_reason="metadata_fetch_error",
            fetch_error=str(raw_meta["fetch_error"]),
        )

    ver_match = _VERSION_RE.search(description)
    balatro_version = ver_match.group(1).strip() if ver_match else ""

    header_raw, body_raw = _split_description_header(description)
    header_clean = _strip_urls(header_raw)
    body_clean = _strip_urls(body_raw)
    full_clean = _strip_urls(description)

    deck, deck_source, deck_err = _extract_deck(title, header_clean, full_clean)
    seed, seed_source, seed_err = _extract_seed(header_clean, full_clean)
    stake, stake_source = _extract_stake(title, header_clean, body_clean)

    skip_reason: str | None = None
    if deck_err:
        skip_reason = deck_err
    elif seed_err:
        skip_reason = seed_err
    elif deck is None and not seed:
        skip_reason = "missing_deck_and_seed"
    elif deck is None:
        skip_reason = "missing_deck"
    elif not seed:
        skip_reason = "missing_seed"

    return ExtractedRunMetadata(
        url=resolved_url,
        video_id=vid,
        title=title,
        upload_date=upload_date,
        duration_seconds=duration,
        channel=channel,
        balatro_version=balatro_version,
        deck=deck,
        seed=seed,
        stake=stake,
        deck_source=deck_source,
        seed_source=seed_source,
        stake_source=stake_source,
        ready=(skip_reason is None),
        skip_reason=skip_reason,
    )


def _is_playlist_only_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    qs = parse_qs(parsed.query)
    return ("/playlist" in parsed.path) and ("list" in qs) and ("v" not in qs)


def _yt_dlp_extra_args(
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    sleep_requests: float | None = None,
) -> list[str]:
    args: list[str] = []
    if cookies_from_browser:
        args.extend(["--cookies-from-browser", str(cookies_from_browser)])
    elif cookies_file:
        args.extend(["--cookies", str(cookies_file)])
    if sleep_requests is not None and sleep_requests > 0:
        args.extend(["--sleep-requests", str(sleep_requests)])
    return args


def expand_input_urls(
    urls: Sequence[str],
    *,
    timeout: int = 60,
    retries: int = 3,
    retry_backoff_sec: float = 2.0,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    sleep_requests: float | None = None,
) -> list[str]:
    """Expands pure playlist URLs via `--flat-playlist`; keeps watch/video URLs as single entries."""
    extra = _yt_dlp_extra_args(
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        sleep_requests=sleep_requests,
    )
    max_attempts = 1 + max(0, int(retries))
    expanded: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        u = raw_url.strip()
        if not u or u.startswith("#"):
            continue
        if _is_playlist_only_url(u):
            loaded = False
            for attempt in range(max_attempts):
                try:
                    proc = subprocess.run(
                        ["yt-dlp", "--flat-playlist", "--dump-single-json", "--no-warnings", *extra, u],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        payload = json.loads(proc.stdout)
                        for entry in payload.get("entries") or ():
                            eid = entry.get("id")
                            eu = entry.get("url") or (
                                f"https://www.youtube.com/watch?v={eid}" if eid else None
                            )
                            if eu and eu not in seen:
                                seen.add(eu)
                                expanded.append(eu)
                        loaded = True
                        break
                except Exception:
                    pass
                if attempt + 1 < max_attempts and retry_backoff_sec > 0:
                    time.sleep(retry_backoff_sec * (2**attempt))
            if loaded:
                continue
        if u not in seen:
            seen.add(u)
            expanded.append(u)
    return expanded


def compact_raw_metadata(raw_meta: dict[str, Any]) -> dict[str, Any]:
    """Keeps lightweight metadata fields for readable per-video JSON archives."""
    return {k: raw_meta.get(k) for k in _RAW_SUMMARY_KEYS if k in raw_meta}


def fetch_video_metadata(
    url: str,
    *,
    timeout: int = 60,
    retries: int = 3,
    retry_backoff_sec: float = 2.0,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    sleep_requests: float | None = None,
) -> dict[str, Any]:
    """Fetches single-video metadata JSON via `yt-dlp --skip-download --no-playlist` with retries."""
    extra = _yt_dlp_extra_args(
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        sleep_requests=sleep_requests,
    )
    max_attempts = 1 + max(0, int(retries))
    last_err = "unknown error"
    for attempt in range(max_attempts):
        try:
            proc = subprocess.run(
                [
                    "yt-dlp",
                    "--dump-single-json",
                    "--no-warnings",
                    "--skip-download",
                    "--no-playlist",
                    *extra,
                    url,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            if proc.returncode == 0:
                raw = json.loads(proc.stdout)
                raw.setdefault("original_url", url)
                return raw
            last_err = (proc.stderr or proc.stdout or f"yt-dlp exit {proc.returncode}").strip()
        except Exception as exc:
            last_err = str(exc)
        if attempt + 1 < max_attempts and retry_backoff_sec > 0:
            time.sleep(retry_backoff_sec * (2**attempt))
    return {"id": extract_video_id(url), "original_url": url, "fetch_error": last_err[:500]}


def fetch_metadata_batch(
    urls: Sequence[str],
    *,
    max_workers: int = 6,
    save_dir: str | Path | None = None,
    timeout: int = 60,
    retries: int = 3,
    retry_backoff_sec: float = 2.0,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    sleep_requests: float | None = None,
    use_cache: bool = True,
) -> list[tuple[dict[str, Any], ExtractedRunMetadata]]:
    """Concurrently fetches and parses metadata for all URLs, reusing cached JSON snapshots when valid."""
    expanded = expand_input_urls(
        urls,
        timeout=timeout,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        sleep_requests=sleep_requests,
    )
    if not expanded:
        return []

    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    raw_by_idx: dict[int, dict[str, Any]] = {}
    to_fetch: list[tuple[int, str]] = []
    for idx, url in enumerate(expanded):
        cached: dict[str, Any] | None = None
        if use_cache and out_dir is not None:
            cache_path = out_dir / f"{extract_video_id(url)}.json"
            if cache_path.exists() and cache_path.stat().st_size > 0:
                try:
                    loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and not loaded.get("fetch_error") and loaded.get("title"):
                        loaded.setdefault("original_url", url)
                        cached = loaded
                except Exception:
                    cached = None
        if cached is not None:
            raw_by_idx[idx] = cached
        else:
            to_fetch.append((idx, url))

    if to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            fetched_vals = list(
                ex.map(
                    lambda item: fetch_video_metadata(
                        item[1],
                        timeout=timeout,
                        retries=retries,
                        retry_backoff_sec=retry_backoff_sec,
                        cookies_from_browser=cookies_from_browser,
                        cookies_file=cookies_file,
                        sleep_requests=sleep_requests,
                    ),
                    to_fetch,
                )
            )
        for (idx, _url), raw in zip(to_fetch, fetched_vals):
            raw_by_idx[idx] = raw

    raw_list = [raw_by_idx[i] for i in range(len(expanded))]

    paired: list[tuple[dict[str, Any], ExtractedRunMetadata]] = []
    index_rows: list[dict[str, Any]] = []
    for url, raw in zip(expanded, raw_list):
        extracted = extract_run_metadata(raw, url=url)
        compact = compact_raw_metadata(raw)
        if raw.get("fetch_error"):
            compact["fetch_error"] = raw["fetch_error"]
        compact["extracted"] = extracted.to_dict()
        if out_dir is not None:
            meta_path = out_dir / f"{extracted.video_id}.json"
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(compact, f, indent=2, ensure_ascii=False)
        paired.append((compact, extracted))
        index_rows.append(extracted.to_dict())

    if out_dir is not None:
        with (out_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump(index_rows, f, indent=2, ensure_ascii=False)
    return paired
