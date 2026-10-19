"""Install check and process management for the real game.

The game is launched through BalatroBot's own launcher code
(``balatrobot.config.Config`` + ``balatrobot.platforms``), i.e. exactly what
``uvx balatrobot serve`` does, plus the companion-mod environment variables.

Parity defaults (:data:`PARITY`): BalatroBot patches ``love.update`` to a
fixed ``dt`` (1/60, or 4.99/60 in ``--headless``); the oracle harness runs
dt = 1/60 at GAMESPEED = 1, so replays use non-headless + ``gamespeed=1``
and get their speed from a high frame-rate cap (2000) with rendering only on
API calls (``render_on_api``). ``--fast`` (GAMESPEED 10) is not
seed-faithful (see coder/balatrobot#234).

The companion mod is installed from ``agent/bridge/mod/antezero_bridge`` of
this checkout (``AZB_MOD_SRC`` overrides it), so the bridge is meant to be
used from the repository (as the uv workspace installs it). Tested with
BalatroBot 1.5.x, Steamodded 1.0.0~BETA / 26.x and lovely 0.10; ``--check``
reports what is installed.

CLI: ``uv run python -m antezero_bridge.serve [--check|--install|--serve]`` (or ``az-game``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from balatrobot.config import BOOL_FIELDS, ENV_MAP, Config
from balatrobot.platforms import get_launcher

from .client import DEFAULT_PORT, BridgeClient

GAME_DIR = Path(os.environ.get("BALATRO_GAME_DIR", r"C:\Program Files (x86)\Steam\steamapps\common\Balatro"))
SAVE_DIR = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "Balatro"
MODS_DIR = SAVE_DIR / "Mods"
MOD_NAME = "antezero_bridge"
MOD_SRC = Path(os.environ.get("AZB_MOD_SRC", Path(__file__).resolve().parents[2] / "mod" / MOD_NAME))
LOGS_DIR = Path(os.environ.get("AZB_LOGS", Path(tempfile.gettempdir()) / "antezero_bridge" / "logs"))


def lovely_dll() -> Path:
    """lovely >= 0.10 ships winmm.dll; older releases version.dll."""
    for name in ("winmm.dll", "version.dll"):
        p = GAME_DIR / name
        if p.exists():
            return p
    return GAME_DIR / "winmm.dll"


@dataclass
class LaunchConfig:
    port: int = DEFAULT_PORT
    gamespeed: int = 1
    fps_cap: int = 2000
    render_on_api: bool = True
    headless: bool = False
    fast: bool = False
    audio: bool = False
    no_shaders: bool = True
    # Companion mod switches (see agent/bridge/mod/antezero_bridge/src/core.lua).
    allow_saves: bool = False
    unlock_all: bool = True
    allow_eval: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)
    logs_path: Path = LOGS_DIR
    health_timeout: float = 90.0

    def balatrobot_config(self) -> Config:
        return Config(
            port=self.port,
            fast=self.fast,
            headless=self.headless,
            render_on_api=self.render_on_api and not self.headless,
            audio=self.audio,
            no_shaders=self.no_shaders,
            fps_cap=self.fps_cap,
            gamespeed=self.gamespeed,
            lovely_path=str(lovely_dll()),
            love_path=str(GAME_DIR / "Balatro.exe"),
            logs_path=str(self.logs_path),
        )

    def env(self) -> dict[str, str]:
        e = {
            "AZB_ALLOW_SAVES": "1" if self.allow_saves else "0",
            "AZB_UNLOCK_ALL": "1" if self.unlock_all else "0",
            "AZB_ALLOW_EVAL": "1" if self.allow_eval else "0",
        }
        e.update(self.extra_env)
        return e


PARITY = LaunchConfig()
"""Settings for sim-parity replays (dt 1/60, GAMESPEED 1, fps cap 2000)."""

FAST = LaunchConfig(fast=True, headless=True, render_on_api=False)
"""BalatroBot ``--fast --headless`` (GAMESPEED 10, dt 4.99/60, no fps cap):
~4-6x faster than PARITY but *not* timing-faithful to the oracle."""


def child_env(cfg: LaunchConfig, launcher=None) -> dict[str, str]:
    """The game's environment: BalatroBot's (which starts from ours) with
    every BalatroBot switch set explicitly (BalatroBot only writes the true
    ones, so e.g. an inherited ``BALATROBOT_FAST=1`` would otherwise win),
    then the companion switches."""
    bb = cfg.balatrobot_config()
    launcher = launcher or get_launcher("windows" if sys.platform == "win32" else None)
    env = launcher.build_env(bb)
    for f in BOOL_FIELDS:
        env[ENV_MAP[f]] = "1" if getattr(bb, f) else "0"
    env.update(cfg.env())
    return env


def _tree_hash(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(f.relative_to(p).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def install_companion_mod(src: Path = MOD_SRC, mods_dir: Path = MODS_DIR) -> Path:
    """Copy the companion mod into the Mods folder (if changed). The copy is
    staged next to it first, so a failure never leaves a partial mod."""
    dst = mods_dir / MOD_NAME
    if not (src / "antezero_bridge.json").exists():
        raise FileNotFoundError(f"companion mod source not found at {src}")
    if dst.exists() and _tree_hash(src) == _tree_hash(dst):
        return dst
    new, old = dst.with_name(MOD_NAME + ".new"), dst.with_name(MOD_NAME + ".old")
    for p in (new, old):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(src, new)
    if _tree_hash(src) != _tree_hash(new):
        shutil.rmtree(new)
        raise OSError(f"copying the companion mod to {new} failed")
    if dst.exists():
        dst.rename(old)
    try:
        new.rename(dst)
    except OSError:
        if old.exists():
            old.rename(dst)
        raise
    if old.exists():
        shutil.rmtree(old)
    return dst


def check_stack() -> dict[str, object]:
    """What is installed where (no side effects)."""
    def mod_version(name: str, manifest: str) -> str | None:
        m = MODS_DIR / name / manifest
        if not m.exists():
            return None
        import json

        try:
            return json.loads(m.read_text(encoding="utf-8")).get("version", "?")
        except ValueError:
            return "?"

    smods_ver = None
    v = MODS_DIR / "smods" / "version.lua"
    if v.exists():
        smods_ver = v.read_text(encoding="utf-8").strip().removeprefix("return ").strip('"')
    dst = MODS_DIR / MOD_NAME
    return {
        "game_exe": (GAME_DIR / "Balatro.exe").exists(),
        "lovely_dll": str(lovely_dll()) if lovely_dll().exists() else None,
        "smods": smods_ver,
        "balatrobot": mod_version("balatrobot", "balatrobot.json"),
        "companion": mod_version(MOD_NAME, "antezero_bridge.json"),
        "companion_in_sync": dst.exists() and MOD_SRC.exists() and _tree_hash(MOD_SRC) == _tree_hash(dst),
        "other_mods": sorted(p.name for p in MODS_DIR.iterdir() if p.is_dir() and p.name not in ("smods", "balatrobot", MOD_NAME, "lovely")) if MODS_DIR.exists() else [],
    }


def save_fingerprint() -> dict[str, str]:
    """sha256 of the player's settings/profile/save files (to prove we left them alone)."""
    out = {}
    for f in [SAVE_DIR / "settings.jkr", *sorted(SAVE_DIR.glob("[123]/*.jkr"))]:
        if f.exists():
            out[str(f.relative_to(SAVE_DIR))] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


class GameProcess:
    """A running Balatro + BalatroBot + companion instance."""

    def __init__(self, cfg: LaunchConfig, proc: subprocess.Popen, log_path: Path):
        self.cfg = cfg
        self.proc = proc
        self.log_path = log_path
        self.client = BridgeClient(port=cfg.port)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        self.client.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def __enter__(self) -> GameProcess:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def launch(cfg: LaunchConfig = PARITY, install: bool = True) -> GameProcess:
    """Start the game and wait until BalatroBot and the companion respond
    (the process is stopped again if that fails)."""
    probe = BridgeClient(port=cfg.port)
    try:
        if probe.health():
            raise RuntimeError(f"something already answers on port {cfg.port}; stop it or attach to it")
    finally:
        probe.close()
    if install:
        install_companion_mod()
    bb = cfg.balatrobot_config()
    launcher = get_launcher("windows" if sys.platform == "win32" else None)
    launcher.validate_paths(bb)
    env = child_env(cfg, launcher)
    session = Path(cfg.logs_path) / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    session.mkdir(parents=True, exist_ok=True)
    log_path = session / f"{cfg.port}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(launcher.build_cmd(bb), env=env, stdout=log, stderr=subprocess.STDOUT)
    gp = GameProcess(cfg, proc, log_path)
    try:
        deadline = time.monotonic() + cfg.health_timeout
        last: Exception | None = None
        while (left := deadline - time.monotonic()) > 0:
            if proc.poll() is not None:
                raise RuntimeError(f"Balatro exited with {proc.returncode}; see {log_path}")
            if gp.client.health(timeout=min(2.0, left)):
                try:
                    gp.client.az_info(timeout=max(1.0, left))
                    return gp
                except Exception as e:  # companion missing / not yet registered
                    last = e
            time.sleep(0.5)
        raise RuntimeError(f"game did not become healthy in {cfg.health_timeout}s ({last}); see {log_path}")
    except BaseException:
        gp.stop()
        raise


def attach(port: int = DEFAULT_PORT) -> BridgeClient:
    """A client of a game that is already running (with the companion)."""
    c = BridgeClient(port=port)
    if not c.health():
        c.close()
        raise RuntimeError(f"no BalatroBot server on port {port}")
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report installed components")
    ap.add_argument("--install", action="store_true", help="copy the companion mod into Mods/")
    ap.add_argument("--serve", action="store_true", help="launch with parity settings and wait for Ctrl+C")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--fast", action="store_true", help="use the FAST (not timing-faithful) preset")
    ap.add_argument("--allow-eval", action="store_true", help="enable the az_eval debug method")
    a = ap.parse_args(argv)
    if a.install:
        print("installed", install_companion_mod())
    if a.check or not (a.install or a.serve):
        for k, v in check_stack().items():
            print(f"{k:18} {v}")
        print("save files        ", save_fingerprint())
    if a.serve:
        base = FAST if a.fast else PARITY
        cfg = dataclasses.replace(base, port=a.port, allow_eval=a.allow_eval)
        gp = launch(cfg)
        print(f"running (pid {gp.proc.pid}, log {gp.log_path}); Ctrl+C to stop", flush=True)
        try:
            while gp.alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            gp.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
