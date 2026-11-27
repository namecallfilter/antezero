"""
Integration test script: Validates Python GameState engine parity against Lua telemetry logs,
simulating a video pipeline with perfect CV/OCR observations and zero ground-truth leakage
across closed-loop state transitions.

Features:
- Validates active gameplay runs from Balatro's run_telemetry.jsonl.
- Archives runs to testing/telemetry_data/ to accumulate a comprehensive regression corpus.
- Runs batch parity checks across all accumulated edge cases (--all).
- Provides a hook (--clear-archive) to clear legacy logs when the JSONL structure fundamentally changes.

Usage:
    python testing/test_engine_parity.py [--telemetry path/to/run_telemetry.jsonl] [--verbose]
    python testing/test_engine_parity.py --archive [--telemetry path/to/run_telemetry.jsonl]
    python testing/test_engine_parity.py --all [--verbose]
    python testing/test_engine_parity.py --list
    python testing/test_engine_parity.py --clear-archive [--force]
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Ensure 'game' and repo root are in pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ARCHIVE_DIR = REPO_ROOT / "testing" / "telemetry_data"

CLEAR_ARCHIVE_NOTE = """
================================================================================
NOTE / WARNING: CLEARING TELEMETRY ARCHIVE
--------------------------------------------------------------------------------
This hook clears all stored JSONL test trajectories in testing/telemetry_data/.
It should ONLY be run whenever the structure of run_telemetry.jsonl fundamentally
changes (such as telemetry schema migrations, card representation changes, or new
action formats) such that legacy logs become invalid or incompatible with
new engine systems.
================================================================================
"""

try:
    from testing.test_engine import (
        run_synthetic_test,
        test_trajectory_parity as run_parity_check,
    )
except ImportError:
    from test_engine import (
        run_synthetic_test,
        test_trajectory_parity as run_parity_check,
    )


def find_telemetry_file(explicit_path: Path | None = None) -> Path | None:
    """Finds the most recently modified telemetry file across standard Balatro and repo paths."""
    if explicit_path and explicit_path.exists() and explicit_path.stat().st_size > 0:
        return explicit_path

    candidates = [
        REPO_ROOT / "run_telemetry.jsonl",
        REPO_ROOT / "output" / "run_telemetry.jsonl",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "Balatro" / "run_telemetry.jsonl")
        candidates.append(Path(appdata) / "Balatro" / "Mods" / "run_telemetry.jsonl")
    candidates.append(Path.home() / "AppData" / "Roaming" / "Balatro" / "run_telemetry.jsonl")
    candidates.append(Path.home() / "AppData" / "Roaming" / "Balatro" / "Mods" / "run_telemetry.jsonl")

    non_empty = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if non_empty:
        return max(non_empty, key=lambda p: p.stat().st_mtime)

    if explicit_path and explicit_path.exists():
        return explicit_path
    for p in candidates:
        if p.exists():
            return p

    return None


def get_telemetry_metadata(file_path: Path) -> dict:
    """Reads telemetry header and line counts to extract run seed and step count."""
    meta = {
        "seed": "UNKNOWN",
        "step_count": 0,
        "first_step": 1,
        "last_step": 1,
        "ante": 1,
        "round": 1,
    }
    if not file_path.exists() or file_path.stat().st_size == 0:
        return meta

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return meta

        meta["step_count"] = len(lines)
        first_entry = json.loads(lines[0])
        first_state = first_entry.get("state", {})
        meta["seed"] = first_state.get("seed", "UNKNOWN")
        meta["first_step"] = first_entry.get("step", 1)
        meta["ante"] = first_state.get("ante", 1)
        meta["round"] = first_state.get("round", 1)

        last_entry = json.loads(lines[-1])
        meta["last_step"] = last_entry.get("step", len(lines))
    except Exception as e:
        meta["error"] = str(e)

    return meta


def archive_telemetry(
    source_path: Path | None = None,
    dest_dir: Path = DEFAULT_ARCHIVE_DIR,
    label: str | None = None,
) -> Path | None:
    """
    Saves an active telemetry run to the telemetry archive directory for regression testing.
    Automatically generates a readable filename: run_<seed>_<steps>steps.jsonl.
    """
    target_path = find_telemetry_file(source_path)
    if target_path is None or not target_path.exists():
        print("Cannot archive: No telemetry file found.")
        return None

    if target_path.stat().st_size == 0:
        print(f"Cannot archive: Telemetry file {target_path} is empty (0 bytes).")
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    meta = get_telemetry_metadata(target_path)
    seed = meta.get("seed", "UNKNOWN")
    step_count = meta.get("step_count", 0)

    if label:
        filename = f"run_{seed}_{label}.jsonl"
    else:
        filename = f"run_{seed}_{step_count}steps.jsonl"

    dest_path = dest_dir / filename

    # Avoid duplicating identical files
    if dest_path.exists():
        if dest_path.stat().st_size == target_path.stat().st_size:
            print(f"[Archive] Run is already archived at: {dest_path} ({step_count} steps, seed {seed})")
            return dest_path
        # If size differs, append a timestamp
        ts = int(time.time())
        dest_path = dest_dir / f"run_{seed}_{step_count}steps_{ts}.jsonl"

    shutil.copy2(target_path, dest_path)
    print(f"[Archive] Successfully archived telemetry run to: {dest_path}")
    print(f"          Seed: {seed} | Steps: {step_count} | Source: {target_path}")
    return dest_path


def list_archived_runs(archive_dir: Path = DEFAULT_ARCHIVE_DIR) -> list[Path]:
    """Returns a list of non-empty archived .jsonl runs sorted by name."""
    if not archive_dir.exists():
        return []
    files = sorted(archive_dir.glob("*.jsonl"))
    return [f for f in files if f.stat().st_size > 0]


def clear_telemetry_archive(
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    force: bool = False,
) -> int:
    """
    Hook to clear all stored JSONL test trajectories in testing/telemetry_data/.
    Prominently displays warning note that this should only be run when telemetry schema changes.
    """
    print(CLEAR_ARCHIVE_NOTE)

    if not archive_dir.exists():
        print(f"Archive directory {archive_dir} does not exist.")
        return 0

    jsonl_files = list(archive_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No archived .jsonl runs found in {archive_dir}.")
        return 0

    print(f"Found {len(jsonl_files)} archived run(s) in {archive_dir}:")
    for f in jsonl_files:
        print(f"  - {f.name} ({f.stat().st_size} bytes)")

    if not force:
        try:
            confirm = input("\nAre you sure you want to delete all archived telemetry runs? (y/N): ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Clear archive cancelled. No files were removed.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 0

    count = 0
    for f in jsonl_files:
        try:
            f.unlink()
            count += 1
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    print(f"\n[Clear] Successfully removed {count} archived telemetry file(s) from {archive_dir}.")
    return count


def run_parity_suite_all(archive_dir: Path = DEFAULT_ARCHIVE_DIR, verbose: bool = False) -> int:
    """Runs closed-loop parity checks across every archived telemetry run in the corpus."""
    runs = list_archived_runs(archive_dir)
    if not runs:
        print(f"No archived telemetry runs found in {archive_dir}.")
        print("Archive a run first using: python testing/test_engine_parity.py --archive")
        return 0

    print(f"================================================================================")
    print(f"RUNNING TELEMETRY PARITY TEST SUITE ({len(runs)} ARCHIVED RUN(S))")
    print(f"================================================================================")

    results = []
    total_discrepancies = 0
    total_steps = 0

    for idx, run_file in enumerate(runs, 1):
        meta = get_telemetry_metadata(run_file)
        steps = meta.get("step_count", 0)
        seed = meta.get("seed", "UNKNOWN")
        total_steps += steps

        print(f"\n[{idx}/{len(runs)}] Testing: {run_file.name} (Seed: {seed}, Steps: {steps})...")
        errors = run_parity_check(run_file, verbose=verbose)
        total_discrepancies += errors
        results.append({
            "file": run_file.name,
            "seed": seed,
            "steps": steps,
            "errors": errors,
            "status": "PASSED" if errors == 0 else f"FAILED ({errors} discrepancies)",
        })

    print(f"\n================================================================================")
    print(f"PARITY SUITE SUMMARY:")
    print(f"--------------------------------------------------------------------------------")
    for res in results:
        status_str = "[OK] PASSED" if res["errors"] == 0 else f"[FAIL] {res['errors']} ERRORS"
        print(f"  {status_str:15} | {res['file']:<35} | {res['steps']:>4} steps | Seed: {res['seed']}")
    print(f"--------------------------------------------------------------------------------")
    print(f"Total Runs: {len(runs)} | Total Steps: {total_steps} | Total Discrepancies: {total_discrepancies}")
    if total_discrepancies == 0:
        print("ALL ARCHIVED RUNS ACHIEVED PERFECT PARITY!")
    else:
        print(f"PARITY FAILURES DETECTED: {total_discrepancies} total discrepancies.")
    print(f"================================================================================\n")

    return total_discrepancies


def main(
    telemetry_path: Path | None = None,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    do_archive: bool = False,
    do_all: bool = False,
    do_clear: bool = False,
    do_list: bool = False,
    force: bool = False,
    verbose: bool = False,
    label: str | None = None,
) -> int:
    # 1. Clear archive hook
    if do_clear:
        clear_telemetry_archive(archive_dir=archive_dir, force=force)
        return 0

    # 2. List archive
    if do_list:
        runs = list_archived_runs(archive_dir)
        print(f"\nArchived Telemetry Runs in {archive_dir} ({len(runs)} found):")
        if not runs:
            print("  (No runs archived yet)")
        for idx, r in enumerate(runs, 1):
            meta = get_telemetry_metadata(r)
            print(f"  {idx}. {r.name} | Steps: {meta.get('step_count', '?')} | Seed: {meta.get('seed', '?')}")
        print()
        return 0

    # 3. Archive command
    if do_archive:
        archived_path = archive_telemetry(source_path=telemetry_path, dest_dir=archive_dir, label=label)
        if archived_path and not do_all:
            # Also run parity on newly archived run to verify it
            print(f"\nValidating newly archived run for parity...")
            return run_parity_check(archived_path, verbose=verbose)
        if not archived_path:
            return 1

    # 4. Batch run all archived test cases
    if do_all:
        return run_parity_suite_all(archive_dir=archive_dir, verbose=verbose)

    # 5. Default single-file test
    target_path = find_telemetry_file(telemetry_path)
    if target_path is None or not target_path.exists():
        print("No telemetry file found at default locations or %APPDATA%\\Balatro\\run_telemetry.jsonl.")
        archived = list_archived_runs(archive_dir)
        if archived:
            print(f"Found {len(archived)} archived run(s) in {archive_dir}. Running parity on archived runs...")
            return run_parity_suite_all(archive_dir=archive_dir, verbose=verbose)
        print("Running synthetic integration test instead...")
        run_synthetic_test()
        return 0

    if target_path.stat().st_size == 0:
        print(f"Found telemetry file at: {target_path}")
        print("Notice: The telemetry log is currently 0 bytes (no gameplay actions recorded yet).")
        print("\nHow to record telemetry:")
        print("  1. Launch Balatro (with Steamodded + GameState Dataset CV mod installed).")
        print("  2. Start a run (play hands, discard, or interact with shop/blinds).")
        print("  3. The mod automatically records every action to run_telemetry.jsonl.")
        print("  4. Rerun this script to verify parity against your live run!\n")
        archived = list_archived_runs(archive_dir)
        if archived:
            print(f"Found {len(archived)} archived run(s). Testing archive instead...")
            return run_parity_suite_all(archive_dir=archive_dir, verbose=verbose)
        print("Running synthetic integration test in the meantime...")
        run_synthetic_test()
        return 0

    errors = run_parity_check(target_path, verbose=verbose)
    print(f"\nParity Test Completed on {target_path}. Discrepancies: {errors}")
    if errors == 0:
        print("PERFECT PARITY ACHIEVED!")

    archived_runs = list_archived_runs(archive_dir)
    if archived_runs:
        print(f"\n[Tip] {len(archived_runs)} archived regression run(s) saved in {archive_dir}.")
        print(f"      Run `python testing/test_engine_parity.py --all` to test the full regression corpus.")
    else:
        print(f"\n[Tip] Save this verified run to your regression corpus with:")
        print(f"      python testing/test_engine_parity.py --archive")

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Balatro GameState Engine Parity Validator & Telemetry Archive Manager."
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=None,
        help="Path to specific run_telemetry.jsonl file to validate.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Directory for archived telemetry logs (default: {DEFAULT_ARCHIVE_DIR}).",
    )
    parser.add_argument(
        "--archive",
        "--save",
        dest="do_archive",
        action="store_true",
        help="Archive the active or specified telemetry run to the regression dataset directory.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Custom label for the archived telemetry file (e.g., 'glass_cards_test').",
    )
    parser.add_argument(
        "--all",
        dest="do_all",
        action="store_true",
        help="Run parity checks across all archived telemetry runs in the corpus.",
    )
    parser.add_argument(
        "--list",
        dest="do_list",
        action="store_true",
        help="List all archived telemetry runs currently in the regression corpus.",
    )
    parser.add_argument(
        "--clear-archive",
        dest="do_clear",
        action="store_true",
        help="Hook to clear all archived telemetry runs (run ONLY when telemetry structure fundamentally changes).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass interactive confirmation prompt when clearing archive.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step-by-step state comparison logs.",
    )
    args = parser.parse_args()

    sys.exit(
        main(
            telemetry_path=args.telemetry,
            archive_dir=args.archive_dir,
            do_archive=args.do_archive,
            do_all=args.do_all,
            do_clear=args.do_clear,
            do_list=args.do_list,
            force=args.force,
            verbose=args.verbose,
            label=args.label,
        )
    )
