"""Unit tests for vod_pipeline.metadata and vod_pipeline.runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from unittest.mock import patch

from game.balatro_models import Deck, Stake
from vod_pipeline.deck_matcher import DeckVideoMatch
from vod_pipeline.metadata import (
    extract_run_metadata,
    extract_video_id,
    fetch_metadata_batch,
)
from vod_pipeline.runner import (
    PipelineConfig,
    download_video,
    process_downloaded_video,
    run_batch_pipeline,
)


class TestVodMetadataAndBatchPipeline(unittest.TestCase):
    def test_extract_run_metadata_on_all_19_saved_sample_inputs(self) -> None:
        meta_dir = Path("assets/sample_inputs/metadata")
        self.assertTrue(meta_dir.exists(), f"Expected saved metadata in {meta_dir}")
        index_path = meta_dir / "index.json"
        self.assertTrue(index_path.exists())

        expected_ready = {
            "xPGzf7YJlxM": (Deck.ERRATIC_DECK, "Z1DCPJSE", Stake.GOLD),
            "tIu3nyD178A": (Deck.ANAGLYPH_DECK, "5CT15E6W", Stake.GOLD),
            "_ZuzKjGVRPs": (Deck.MAGIC_DECK, "27T3PBQG", Stake.GOLD),
            "BlJRaf_XTYU": (Deck.MAGIC_DECK, "ISAD3FV6", Stake.GOLD),
            "1166hAVgn9M": (Deck.BLACK_DECK, "W3RIR565", Stake.GOLD),
            "WTS86AYxU6o": (Deck.PAINTED_DECK, "QDZMEKC3", Stake.GOLD),
            "QAi7tTvPXhc": (Deck.NEBULA_DECK, "42REVY7N", Stake.GOLD),
            "YpR4WuvVjbU": (Deck.PAINTED_DECK, "CM78H84Y", Stake.GOLD),
            "zqINP6NEwkQ": (Deck.BLUE_DECK, "PMLR4BVQ", Stake.GOLD),
            "sVuPmDmTyL4": (Deck.ZODIAC_DECK, "FNSD85BC", Stake.GOLD),
            "1HrCULIV_cg": (Deck.ZODIAC_DECK, "KX6LKY4S", Stake.GOLD),
            "Nj_k0gWhRH4": (Deck.ANAGLYPH_DECK, "EXESWZR2", Stake.GOLD),
            "GaIkZ2p07ZM": (Deck.GHOST_DECK, "M4FT2H9Y", Stake.GOLD),
            "BzMdXrJJGZI": (Deck.CHECKERED_DECK, "L68SU9WS", Stake.GOLD),
            "QRvxSAY-s1M": (Deck.PAINTED_DECK, "X37S2WYQ", Stake.GOLD),
            "gJVogwKQulY": (Deck.GHOST_DECK, "8B6YD9FJ", Stake.GOLD),
        }
        expected_skipped = {
            "_MDwyFmIo5s": ("missing_deck", "BZTGPV12", Stake.GOLD),
            "9vDE86eVa0Y": ("missing_deck", "UXS1G65E", Stake.GOLD),
            "KwcBB0pDDJo": ("missing_deck", "VUCDKYZ6", Stake.GOLD),
        }

        for vid, (exp_deck, exp_seed, exp_stake) in expected_ready.items():
            raw = json.loads((meta_dir / f"{vid}.json").read_text(encoding="utf-8"))
            parsed = extract_run_metadata(raw)
            self.assertTrue(parsed.ready, f"{vid} should be ready: {parsed}")
            self.assertIsNone(parsed.skip_reason)
            self.assertEqual(parsed.deck, exp_deck, vid)
            self.assertEqual(parsed.seed, exp_seed, vid)
            self.assertEqual(parsed.stake, exp_stake, vid)

        for vid, (exp_reason, exp_seed, exp_stake) in expected_skipped.items():
            raw = json.loads((meta_dir / f"{vid}.json").read_text(encoding="utf-8"))
            parsed = extract_run_metadata(raw)
            self.assertFalse(parsed.ready, f"{vid} should be skipped: {parsed}")
            self.assertEqual(parsed.skip_reason, exp_reason, vid)
            self.assertIsNone(parsed.deck, vid)
            self.assertEqual(parsed.seed, exp_seed, vid)
            self.assertEqual(parsed.stake, exp_stake, vid)

    def test_metadata_extraction_edge_cases_and_ambiguity_guards(self) -> None:
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=abc123_X&t=10s"),
            "abc123_X",
        )
        self.assertEqual(extract_video_id("https://youtu.be/xyz987-Q?t=5"), "xyz987-Q")

        # Title White Stake overrides Gold Stake description boilerplate
        m1 = extract_run_metadata(
            {
                "id": "stake_override",
                "title": "White Stake Chill Run",
                "description": "Plasma Deck:\n\nAB12CD34\nDaily Gold Stake gameplay.",
            }
        )
        self.assertTrue(m1.ready)
        self.assertEqual(m1.deck, Deck.PLASMA_DECK)
        self.assertEqual(m1.seed, "AB12CD34")
        self.assertEqual(m1.stake, Stake.WHITE)

        # Missing stake defaults to Gold Stake; bare color words ("Red Card", "Blue Seal") ignored
        m2 = extract_run_metadata(
            {
                "id": "default_gold",
                "title": "Red Card + Blue Seal Combo!",
                "description": "Abandoned Deck: \n\n7K9MP2QT\nJust fun Balatro.",
            }
        )
        self.assertTrue(m2.ready)
        self.assertEqual(m2.deck, Deck.ABANDONED_DECK)
        self.assertEqual(m2.seed, "7K9MP2QT")
        self.assertEqual(m2.stake, Stake.GOLD)
        self.assertEqual(m2.stake_source, "default_gold")

        # Conflicting decks between title and description header -> skip safely
        m3 = extract_run_metadata(
            {
                "id": "conflict",
                "title": "Checkered Deck madness",
                "description": "Yellow Deck:\n\n2P3Q4R5S",
            }
        )
        self.assertFalse(m3.ready)
        self.assertEqual(m3.skip_reason, "ambiguous_deck_conflict")

        # Multiple seeds -> skip safely
        m4 = extract_run_metadata(
            {
                "id": "multiseed",
                "title": "Two Green Deck runs",
                "description": "Seed: 1A2B3C4D\nSeed: 5E6F7G8H",
            }
        )
        self.assertFalse(m4.ready)
        self.assertEqual(m4.skip_reason, "ambiguous_multiple_seeds")

        # URLs containing 8-char alphanumeric strings must never match as seeds
        m5 = extract_run_metadata(
            {
                "id": "url_only",
                "title": "Plasma Deck Only",
                "description": "Check https://bit.ly/4qXxtdN2 and https://example.com/1234ABCD",
            }
        )
        self.assertFalse(m5.ready)
        self.assertEqual(m5.skip_reason, "missing_seed")

    def test_batch_pipeline_overlaps_bounds_backlog_skips_and_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "batch_runs"
            urls = [f"https://www.youtube.com/watch?v=vid{i}" for i in range(7)]

            # vid1 missing deck -> must be skipped BEFORE downloading
            # vid4 missing seed -> must be skipped BEFORE downloading
            # vid5 fails during processing -> video MUST still be deleted
            def fake_fetch(input_urls, _cfg):
                out = []
                for idx, u in enumerate(input_urls):
                    vid = f"vid{idx}"
                    if idx == 1:
                        raw = {"id": vid, "title": "Red Card only", "description": "Seed: 12AB34CD"}
                    elif idx == 4:
                        raw = {"id": vid, "title": "Anaglyph Deck", "description": "No seed here"}
                    else:
                        raw = {
                            "id": vid,
                            "title": "Purple Stake Magic Deck",
                            "description": f"Magic Deck:\n\n2B3C4D5{idx}",
                        }
                    out.append((raw, extract_run_metadata(raw, url=u)))
                return out

            state_lock = threading.Lock()
            downloaded_vids: list[str] = []
            processed_vids: list[str] = []
            active_downloads = [0]
            max_active_downloads = [0]
            max_files_on_disk = [0]

            def _count_disk_videos(ddir: Path) -> int:
                return len(list(ddir.glob("*.mp4"))) if ddir.exists() else 0

            def fake_download(_url, video_id, downloads_dir, _cfg, _stop_event):
                with state_lock:
                    downloaded_vids.append(video_id)
                    active_downloads[0] += 1
                    max_active_downloads[0] = max(max_active_downloads[0], active_downloads[0])
                time.sleep(0.06)
                p = downloads_dir / f"{video_id}.mp4"
                p.write_bytes(b"fake_mp4")
                with state_lock:
                    active_downloads[0] -= 1
                    max_files_on_disk[0] = max(max_files_on_disk[0], _count_disk_videos(downloads_dir))
                return p

            def fake_process(video_path, meta, _cfg):
                with state_lock:
                    max_files_on_disk[0] = max(
                        max_files_on_disk[0], _count_disk_videos(video_path.parent)
                    )
                    processed_vids.append(meta.video_id)
                time.sleep(0.05)
                if meta.video_id == "vid5":
                    raise RuntimeError("simulated processing failure")
                traj_out = work_dir / "trajectories" / f"{meta.video_id}_trajectory.json"
                traj_out.write_text("{}", encoding="utf-8")
                return {
                    "actions_json": str(work_dir / "actions" / f"{meta.video_id}_actions.json"),
                    "trajectory_json": str(traj_out),
                    "num_steps": 12,
                    "final_ante": 8,
                    "final_round": 24,
                    "elapsed_sec": 0.05,
                }

            cfg = PipelineConfig(
                work_dir=work_dir,
                max_backlog=2,
                download_workers=2,
                keep_videos=False,
                allow_deck_template_fallback=False,
            )
            report = run_batch_pipeline(
                urls,
                cfg,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download,
                process_fn=fake_process,
            )

            self.assertEqual(report["counts"]["total"], 7)
            self.assertEqual(report["counts"]["eligible"], 5)
            self.assertEqual(report["counts"]["processed"], 4)
            self.assertEqual(report["counts"]["skipped"], 2)
            self.assertEqual(report["counts"]["failed"], 1)

            self.assertNotIn("vid1", downloaded_vids)
            self.assertNotIn("vid4", downloaded_vids)
            self.assertEqual(set(downloaded_vids), {"vid0", "vid2", "vid3", "vid5", "vid6"})
            self.assertGreaterEqual(max_active_downloads[0], 2)
            self.assertLessEqual(max_files_on_disk[0], 2)
            # All processed & failed videos deleted after completion
            self.assertEqual(list((work_dir / "downloads").glob("*.mp4")), [])
            self.assertTrue(Path(report["report_path"]).exists())
            self.assertTrue((work_dir / "reports" / "manifest.jsonl").exists())

            # Re-running with resume=True skips the 4 completed videos without re-downloading
            downloaded_vids.clear()
            report_resume = run_batch_pipeline(
                urls,
                cfg,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download,
                process_fn=fake_process,
            )
            self.assertEqual(report_resume["counts"]["eligible"], 1)  # only previously failed vid5
            self.assertEqual(downloaded_vids, ["vid5"])

    def test_batch_pipeline_deck_template_fallback_recovers_and_skips_unmatched(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "batch_runs"
            urls = [f"https://www.youtube.com/watch?v={vid}" for vid in ("miss_ok", "conflict_ok", "miss_fail", "noseed_skip")]

            def fake_fetch(input_urls, _cfg):
                raws = {
                    "miss_ok": {"id": "miss_ok", "title": "Square over Telescope", "description": "Seed: BZTGPV12"},
                    "conflict_ok": {"id": "conflict_ok", "title": "Checkered Deck", "description": "Yellow Deck:\n\n2P3Q4R5S"},
                    "miss_fail": {"id": "miss_fail", "title": "Mystery Run", "description": "Seed: VUCDKYZ6"},
                    "noseed_skip": {"id": "noseed_skip", "title": "Mystery Run", "description": "No seed and no deck"},
                }
                out = []
                for u in input_urls:
                    vid = extract_video_id(u)
                    raw = raws[vid]
                    out.append((raw, extract_run_metadata(raw, url=u)))
                return out

            downloaded: list[str] = []
            processed: list[tuple[str, Deck | None, str]] = []

            def fake_download(_url, video_id, downloads_dir, _cfg, _stop_event):
                downloaded.append(video_id)
                p = downloads_dir / f"{video_id}.mp4"
                p.write_bytes(b"fake_mp4")
                return p

            def fake_deck_matcher(video_path: Path, _cfg):
                stem = video_path.stem
                if stem == "miss_ok":
                    return DeckVideoMatch(
                        deck=Deck.NEBULA_DECK,
                        confidence=0.91,
                        mean_margin=0.14,
                        matched_frames=4,
                        sampled_frames=6,
                        votes={"Nebula Deck": 4},
                        best_frame_idx=300,
                    )
                if stem == "conflict_ok":
                    return DeckVideoMatch(
                        deck=Deck.CHECKERED_DECK,
                        confidence=0.87,
                        mean_margin=0.36,
                        matched_frames=3,
                        sampled_frames=6,
                        votes={"Checkered Deck": 3},
                        best_frame_idx=400,
                    )
                return None

            def fake_process(_video_path, meta, _cfg):
                processed.append((meta.video_id, meta.deck, meta.deck_source))
                traj_out = work_dir / "trajectories" / f"{meta.video_id}_trajectory.json"
                traj_out.write_text("{}", encoding="utf-8")
                return {
                    "actions_json": str(work_dir / "actions" / f"{meta.video_id}_actions.json"),
                    "trajectory_json": str(traj_out),
                    "num_steps": 5,
                    "final_ante": 8,
                    "final_round": 24,
                    "elapsed_sec": 0.01,
                }

            cfg = PipelineConfig(
                work_dir=work_dir,
                max_backlog=2,
                download_workers=2,
                keep_videos=False,
                allow_deck_template_fallback=True,
            )
            report = run_batch_pipeline(
                urls,
                cfg,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download,
                process_fn=fake_process,
                deck_matcher_fn=fake_deck_matcher,
            )

            self.assertEqual(report["counts"]["total"], 4)
            self.assertEqual(report["counts"]["eligible"], 3)
            self.assertEqual(report["counts"]["processed"], 2)
            self.assertEqual(report["counts"]["skipped"], 2)
            self.assertEqual(report["counts"]["failed"], 0)

            self.assertEqual(set(downloaded), {"miss_ok", "conflict_ok", "miss_fail"})
            self.assertEqual(
                {vid: deck for vid, deck, _ in processed},
                {"miss_ok": Deck.NEBULA_DECK, "conflict_ok": Deck.CHECKERED_DECK},
            )
            for _, _, src in processed:
                self.assertTrue(src.startswith("deck_template_match("), src)
            skip_reasons = {row["video_id"]: row["skip_reason"] for row in report["skipped"]}
            self.assertEqual(
                skip_reasons,
                {"noseed_skip": "missing_deck_and_seed", "miss_fail": "deck_template_not_found"},
            )
            self.assertEqual(list((work_dir / "downloads").glob("*.mp4")), [])

    def test_dry_run_19_saved_metadata_all_eligible_with_deck_template_fallback(self) -> None:
        meta_dir = Path("assets/sample_inputs/metadata")
        paths = sorted(p for p in meta_dir.glob("*.json") if p.name != "index.json")
        self.assertEqual(len(paths), 19)

        def load_saved(urls, _cfg):
            out = []
            for p in paths:
                raw = json.loads(p.read_text(encoding="utf-8"))
                out.append((raw, extract_run_metadata(raw)))
            return out

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PipelineConfig(
                work_dir=Path(tmpdir) / "batch_runs",
                dry_run=True,
                allow_deck_template_fallback=True,
                min_duration_sec=0.0,
            )
            report = run_batch_pipeline(
                ["https://www.youtube.com/watch?v=dummy"],
                cfg,
                fetch_metadata_fn=load_saved,
            )

        self.assertEqual(report["counts"]["total"], 19)
        self.assertEqual(report["counts"]["eligible"], 19)
        self.assertEqual(report["counts"]["skipped"], 0)
        pending = {
            row["video_id"]: row["deck_source"]
            for row in report["eligible_preview"]
            if row["deck_source"] == "pending_deck_template_match"
        }
        self.assertEqual(
            set(pending.keys()),
            {"_MDwyFmIo5s", "9vDE86eVa0Y", "KwcBB0pDDJo"},
        )

    def test_process_downloaded_video_reuses_actions_json_and_collects_cleanliness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "batch_runs"
            actions_dir = work_dir / "actions"
            traj_dir = work_dir / "trajectories"
            dl_dir = work_dir / "downloads"
            actions_dir.mkdir(parents=True, exist_ok=True)
            traj_dir.mkdir(parents=True, exist_ok=True)
            dl_dir.mkdir(parents=True, exist_ok=True)

            vid = "dQw4w9WgXcQ"
            video_path = dl_dir / f"{vid}.mp4"
            video_path.write_bytes(b"mp4")
            actions_path = actions_dir / f"{vid}_actions.json"
            actions_path.write_text(
                json.dumps({"key": {"Play Hand": 0}, "frames": {"100": {"action": 0}}}),
                encoding="utf-8",
            )
            traj_path = traj_dir / f"{vid}_trajectory.json"
            traj_path.write_text(
                json.dumps(
                    {
                        "initial_state": {"ante": 1},
                        "steps": [
                            {"ante": 1, "action": {"metadata": {"corrupted": False, "corruption_reasons": []}}},
                            {"ante": 7, "action": {"metadata": {"corrupted": True, "corruption_reasons": ["location_mismatch"]}}},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            meta = extract_run_metadata(
                {
                    "id": vid,
                    "title": "Blue Deck Gold Stake",
                    "description": "Seed: ABCD1234",
                },
                url=f"https://www.youtube.com/watch?v={vid}",
            )

            seen_flags: list[tuple[bool, bool]] = []

            def fake_run_end_to_end(args):
                seen_flags.append((bool(args.actions_json), bool(args.full_video)))
                return {
                    "actions_json": args.actions_json or args.actions_out,
                    "trajectory_json": str(traj_path),
                    "trajectory": None,
                    "state": None,
                }

            import main as vod_main

            cfg_resume = PipelineConfig(work_dir=work_dir, resume=True)
            with patch.object(vod_main, "run_end_to_end", side_effect=fake_run_end_to_end):
                res_resume = process_downloaded_video(video_path, meta, cfg_resume)
                self.assertTrue(res_resume["actions_reused"])
                self.assertEqual(res_resume["clean_steps"], 1)
                self.assertEqual(res_resume["total_steps"], 2)
                self.assertEqual(res_resume["ante78_clean"], 0)
                self.assertEqual(res_resume["ante78_total"], 1)

                cfg_fresh = PipelineConfig(work_dir=work_dir, resume=False)
                res_fresh = process_downloaded_video(video_path, meta, cfg_fresh)
                self.assertFalse(res_fresh["actions_reused"])

            self.assertEqual(seen_flags, [(True, False), (False, True)])

    def test_download_video_retries_with_backoff_and_metadata_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dl_dir = root / "downloads"
            meta_dir = root / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)

            # 1) Retry + backoff recovery in download_video
            attempt_count = [0]

            class _FakePopen:
                def __init__(self, cmd, **_kwargs):
                    attempt_count[0] += 1
                    self._attempt = attempt_count[0]
                    if self._attempt >= 3:
                        (dl_dir / "vidRetry123.mp4").write_bytes(b"ok_video")
                    self.returncode = 0 if self._attempt >= 3 else 1

                def poll(self):
                    return self.returncode

                def communicate(self):
                    if self.returncode == 0:
                        return str(dl_dir / "vidRetry123.mp4") + "\n", ""
                    return "", "HTTP Error 429: Too Many Requests"

            sleeps: list[float] = []
            with (
                patch("vod_pipeline.runner.subprocess.Popen", side_effect=lambda *a, **kw: _FakePopen(a[0], **kw)),
                patch("vod_pipeline.runner._sleep_with_stop", side_effect=lambda s, _ev: sleeps.append(s)),
            ):
                out_path = download_video(
                    "https://www.youtube.com/watch?v=vidRetry123",
                    "vidRetry123",
                    dl_dir,
                    retries=2,
                    retry_backoff_sec=1.5,
                )
            self.assertTrue(out_path.exists())
            self.assertEqual(attempt_count[0], 3)
            self.assertEqual(sleeps, [1.5, 3.0])

            # 2) Cached metadata reuse in fetch_metadata_batch without calling fetch_video_metadata
            cached_payload = {
                "id": "cachedVid01",
                "title": "Yellow Deck Gold Stake",
                "description": "Seed: QWER5678",
            }
            (meta_dir / "cachedVid01.json").write_text(json.dumps(cached_payload), encoding="utf-8")
            with patch(
                "vod_pipeline.metadata.fetch_video_metadata",
                side_effect=AssertionError("Should reuse valid cached metadata"),
            ):
                paired = fetch_metadata_batch(
                    ["https://www.youtube.com/watch?v=cachedVid01"],
                    save_dir=meta_dir,
                    use_cache=True,
                )
            self.assertEqual(len(paired), 1)
            self.assertTrue(paired[0][1].ready)
            self.assertEqual(paired[0][1].deck, Deck.YELLOW_DECK)
            self.assertEqual(paired[0][1].seed, "QWER5678")

    def test_batch_pipeline_parallel_stage1_serial_stage2_and_cleanliness_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "batch_runs"
            vids = ["par0", "par1", "par2", "par3"]
            urls = [f"https://www.youtube.com/watch?v={v}" for v in vids]

            def fake_fetch(input_urls, _cfg):
                return [
                    (
                        {"id": v, "title": "Red Deck Gold Stake", "description": "Seed: AAAA1111"},
                        extract_run_metadata(
                            {"id": v, "title": "Red Deck Gold Stake", "description": "Seed: AAAA1111"},
                            url=f"https://www.youtube.com/watch?v={v}",
                        ),
                    )
                    for v in vids
                ]

            lock = threading.Lock()
            active_s1 = [0]
            max_active_s1 = [0]
            active_s2 = [0]
            max_active_s2 = [0]
            max_files_on_disk = [0]

            def fake_download(_url, video_id, downloads_dir, _cfg, _stop_event):
                p = downloads_dir / f"{video_id}.mp4"
                p.write_bytes(b"mp4")
                with lock:
                    on_disk = len(list(downloads_dir.glob("*.mp4")))
                    max_files_on_disk[0] = max(max_files_on_disk[0], on_disk)
                return p

            def fake_stage1(_vpath, _meta, _cfg, actions_out: Path) -> float:
                with lock:
                    active_s1[0] += 1
                    max_active_s1[0] = max(max_active_s1[0], active_s1[0])
                time.sleep(0.08)
                actions_out.write_text(json.dumps({"key": {}, "frames": {}}), encoding="utf-8")
                with lock:
                    active_s1[0] -= 1
                return 0.08

            def fake_stage2(_vpath, meta, _cfg, _actions_out: Path) -> dict:
                with lock:
                    active_s2[0] += 1
                    max_active_s2[0] = max(max_active_s2[0], active_s2[0])
                time.sleep(0.02)
                with lock:
                    active_s2[0] -= 1
                traj_out = work_dir / "trajectories" / f"{meta.video_id}_trajectory.json"
                traj_out.write_text("{}", encoding="utf-8")
                return {
                    "actions_json": str(work_dir / "actions" / f"{meta.video_id}_actions.json"),
                    "trajectory_json": str(traj_out),
                    "num_steps": 10,
                    "clean_steps": 9,
                    "total_steps": 10,
                    "clean_pct": 90.0,
                    "ante13_clean": 5,
                    "ante13_total": 5,
                    "ante46_clean": 3,
                    "ante46_total": 3,
                    "ante78_clean": 1,
                    "ante78_total": 2,
                    "ante78_pct": 50.0,
                    "cascade_step": None,
                    "corruption_reasons": {"large_ocr_override": 1},
                    "stage2_sec": 0.02,
                    "elapsed_sec": 0.1,
                }

            cfg = PipelineConfig(
                work_dir=work_dir,
                max_backlog=2,
                download_workers=2,
                stage1_workers=2,
                keep_videos=False,
            )
            report = run_batch_pipeline(
                urls,
                cfg,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download,
                stage1_fn=fake_stage1,
                stage2_fn=fake_stage2,
            )

            self.assertEqual(report["counts"]["processed"], 4)
            self.assertEqual(report["counts"]["failed"], 0)
            self.assertGreaterEqual(max_active_s1[0], 2)
            self.assertEqual(max_active_s2[0], 1)
            self.assertLessEqual(max_files_on_disk[0], 2)
            self.assertEqual(list((work_dir / "downloads").glob("*.mp4")), [])

            csum = report["cleanliness_summary"]
            self.assertEqual(csum["clean_steps"], 36)
            self.assertEqual(csum["total_steps"], 40)
            self.assertEqual(csum["clean_pct"], 90.0)
            self.assertEqual(csum["ante78_clean"], 4)
            self.assertEqual(csum["ante78_total"], 8)
            self.assertEqual(csum["ante78_pct"], 50.0)
            self.assertEqual(csum["corruption_reasons"], {"large_ocr_override": 4})

    def test_min_duration_filter_skips_short_videos_prior_to_download(self) -> None:
        raw_items = [
            {
                "id": "short_ready",
                "webpage_url": "https://www.youtube.com/watch?v=short_ready",
                "title": "Red Deck quick loss",
                "description": "Red Deck:\n\nAAAA1111",
                "duration": 1799,
            },
            {
                "id": "short_fallback",
                "webpage_url": "https://www.youtube.com/watch?v=short_fallback",
                "title": "Mystery loss",
                "description": "Seed: BBBB2222",
                "duration": 600,
            },
            {
                "id": "exact_30m",
                "webpage_url": "https://www.youtube.com/watch?v=exact_30m",
                "title": "Blue Deck full run",
                "description": "Blue Deck:\n\nCCCC3333",
                "duration": 1800,
            },
        ]

        def fake_fetch(urls, cfg):
            return [(raw, extract_run_metadata(raw)) for raw in raw_items]

        downloaded_ids: list[str] = []

        def fake_download(url, vid, ddir, cfg, stop_ev):
            downloaded_ids.append(vid)
            p = ddir / f"{vid}.mp4"
            p.write_bytes(b"mp4")
            return p

        def fake_process(vpath, meta, cfg, **kwargs):
            traj_out = cfg.work_dir / "trajectories" / f"{meta.video_id}_trajectory.json"
            traj_out.write_text("{}", encoding="utf-8")
            return {"trajectory_json": str(traj_out), "num_steps": 5}

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "batch_runs"
            cfg = PipelineConfig(
                work_dir=work_dir,
                allow_deck_template_fallback=True,
                allow_ocr_seed_fallback=True,
                keep_videos=True,
            )
            report = run_batch_pipeline(
                [r["webpage_url"] for r in raw_items],
                cfg,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download,
                process_fn=fake_process,
            )
            self.assertEqual(report["counts"]["processed"], 1)
            self.assertEqual(report["counts"]["skipped"], 2)
            self.assertEqual(downloaded_ids, ["exact_30m"])
            skipped_by_id = {s["video_id"]: s for s in report["skipped"]}
            self.assertEqual(skipped_by_id["short_ready"]["skip_reason"], "too_short_under_30m")
            self.assertEqual(skipped_by_id["short_fallback"]["skip_reason"], "too_short_under_30m")

        # Setting min_duration_sec=0 disables the filter
        downloaded_all: list[str] = []

        def fake_download_all(url, vid, ddir, cfg, stop_ev):
            downloaded_all.append(vid)
            p = ddir / f"{vid}.mp4"
            p.write_bytes(b"mp4")
            return p

        with tempfile.TemporaryDirectory() as tmp2:
            cfg_off = PipelineConfig(
                work_dir=Path(tmp2) / "batch_runs",
                min_duration_sec=0.0,
                allow_deck_template_fallback=False,
            )
            report_off = run_batch_pipeline(
                [r["webpage_url"] for r in raw_items],
                cfg_off,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download_all,
                process_fn=fake_process,
            )
            self.assertEqual(sorted(downloaded_all), ["exact_30m", "short_ready"])
            self.assertEqual(report_off["counts"]["processed"], 2)
            self.assertEqual(report_off["counts"]["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
