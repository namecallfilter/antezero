"""Replay a sim history file in the real game and record the state after
every step as JSONL.

History format (``sim/oracle/src/diff.rs`` ``Failure::codes``)::

    start <seed> <back_key> <stake>
    <action code>        e.g. "play 30", "buy 1", "pick 1 0"
    <inject code>        e.g. "inject_joker 17 0 0"

Output: one JSON object per line, in step order (step 0 = the start)::

    {"step": 3, "line": "play 30", "kind": "action", "route": "harness",
     "ok": true, "error": null, "secs": 0.41, "state": {...}}

``state`` is embedded verbatim as produced in-game by the companion mod: the
same JSON as the oracle harness's ``H.state`` (sorted keys, %.17g numbers),
so ``json::parse`` + ``canon::from_lua`` in ``sim/oracle`` read it unchanged
(``antezero_bridge.canon.from_lua`` is the Python equivalent).

With ``--oracle-cli path/to/ocli.exe`` the same commands are also sent to the
headless vanilla oracle and the canonical states are diffed after every step
(prints the first divergence, like ``sim/oracle/src/bin/replay.rs``).

Examples::

    uv run python -m antezero_bridge.replay .scratch/fail_T0006000_b_blue_2.txt -o out.jsonl
    uv run python -m antezero_bridge.replay hist.txt --attach --oracle-cli target/release/ocli.exe
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from . import canon
from .actions import Action, lua_value
from .game_env import ActionRejected, GameEnv
from .injects import Inject
from .launch import FAST, PARITY, attach, launch, save_fingerprint


def read_history(path: Path) -> tuple[tuple[str, str, int], list[str]]:
    """The ``(seed, back, stake)`` of a history file and its step lines."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    head = lines[0].split() if lines else []
    if len(head) != 4 or head[0] != "start" or not head[3].isdigit() or not 1 <= int(head[3]) <= 8:
        raise SystemExit(f"{path}: the first line must be 'start <seed> <back> <stake 1..8>', got {(lines or [''])[0]!r}")
    return (head[1], head[2], int(head[3])), lines[1:]


class Oracle:
    """Pipe to the oracle CLI (``sim/oracle/src/bin/ocli.rs``): one Lua
    expression per stdin line, one result line on stdout (``ERROR: ...`` on
    failure)."""

    def __init__(self, exe: str, timeout: float = 120.0):
        self.timeout = timeout
        self.p = subprocess.Popen([str(Path(exe).resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.p.stdout
        for ln in self.p.stdout:
            self._lines.put(ln)
        self._lines.put("")  # end of output

    def call(self, cmd: str) -> str:
        assert self.p.stdin
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        try:
            out = self._lines.get(timeout=self.timeout)
        except queue.Empty:
            raise RuntimeError(f"the oracle did not answer within {self.timeout}s: {cmd[:80]}") from None
        if not out:
            raise RuntimeError(f"the oracle exited (status {self.p.poll()})")
        out = out.rstrip("\n")
        if out.startswith("ERROR:"):
            raise RuntimeError(out)
        return out

    def close(self) -> None:
        try:
            if self.p.stdin:
                self.p.stdin.close()
            self.p.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait()


def record(out: TextIO | None, rec: dict[str, Any], raw_state: str | None) -> None:
    if out is None:
        return
    body = json.dumps(rec, separators=(",", ":"))
    if raw_state is not None:
        body = body[:-1] + ',"state":' + raw_state + "}"
    out.write(body + "\n")
    out.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("history", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="JSONL output (default: <history>.game.jsonl)")
    ap.add_argument("--route", choices=["harness", "balatrobot"], default="harness")
    ap.add_argument("--inject-route", choices=["harness", "balatrobot"], default="harness")
    ap.add_argument("--attach", action="store_true", help="use an already running game instead of launching one")
    ap.add_argument("--port", type=int, default=12346)
    ap.add_argument("--fast", action="store_true", help="launch with the FAST preset (not timing-faithful)")
    ap.add_argument("--keep-going", action="store_true", help="continue after a rejected step / divergence")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--settle-timeout", type=float, default=120.0)
    ap.add_argument("--oracle-cli", help="path to ocli.exe: diff against the headless vanilla oracle")
    ap.add_argument("--oracle-out", type=Path, help="also write the oracle states as JSONL")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    (seed, back, stake), lines = read_history(a.history)
    if a.max_steps:
        lines = lines[: a.max_steps]
    out_path = a.out or a.history.with_suffix(".game.jsonl")
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))

    fp_before = save_fingerprint()
    diverged = None
    errors = 0
    t_all = time.monotonic()
    # Every resource is released (in reverse order) however the replay ends.
    with contextlib.ExitStack() as stack:
        if a.attach:
            try:
                client = attach(a.port)
            except RuntimeError as e:
                raise SystemExit(str(e)) from None
            stack.callback(client.close)
        else:
            cfg = dataclasses.replace(FAST if a.fast else PARITY, port=a.port)
            log(f"launching game (gamespeed {cfg.gamespeed}, headless {cfg.headless}) ...")
            proc = launch(cfg)
            stack.callback(proc.stop)
            client = proc.client
        info = client.az_info()
        diag = info.get("diag", {})
        log(f"companion {info.get('version')} smods {info.get('smods_version')} balatrobot {info.get('balatrobot_version')} "
            f"gamespeed {info.get('gamespeed')} dt {diag.get('real_dt')}")
        if info.get("gamespeed") != 1 or abs((diag.get("real_dt") or 0) - 1 / 60) > 1e-9:
            log("WARNING: not running at GAMESPEED 1 / dt 1/60 - timing differs from the oracle")

        env = GameEnv(client, route=a.route, inject_route=a.inject_route, settle_timeout=a.settle_timeout)
        oracle = Oracle(a.oracle_cli) if a.oracle_cli else None
        if oracle:
            stack.callback(oracle.close)
        ofile = stack.enter_context(open(a.oracle_out, "w", encoding="utf-8")) if (oracle and a.oracle_out) else None
        out = stack.enter_context(open(out_path, "w", encoding="utf-8"))
        t = time.monotonic()
        env.reset(seed, back, stake)
        record(out, {"step": 0, "line": f"start {seed} {back} {stake}", "kind": "start", "route": "harness",
                     "ok": True, "error": None, "secs": round(time.monotonic() - t, 3)}, env.state_json())
        ostate = oracle.call(f"H.start({lua_value(seed)},{lua_value(back)},{stake})") if oracle else None
        if ofile and ostate:
            ofile.write(ostate + "\n")

        def compare(step: int, line: str) -> bool:
            nonlocal diverged
            if not oracle or ostate is None:
                return True
            d = canon.diff(canon.from_lua(ostate), env.canonical())
            if d and diverged is None:
                diverged = step
                log(f"DIVERGED at step {step}: {line}")
                for k, x, y in d[:60]:
                    log(f"  {k}: oracle={x} game={y}")
                if len(d) > 60:
                    log(f"  ... {len(d) - 60} more")
            return not d

        ok = compare(0, "<start>")
        for step, line in enumerate(lines, start=1):
            if not ok and not a.keep_going:
                break
            act = Action.parse(line)
            inj = None if act else Inject.parse(line)
            if act is None and inj is None:
                raise SystemExit(f"step {step}: bad line {line!r}")
            t = time.monotonic()
            err = None
            try:
                if act:
                    env.step(act)
                else:
                    env.inject(inj)  # type: ignore[arg-type]
            except ActionRejected as e:
                err = str(e)
                errors += 1
            route = env.last_route or "harness"
            secs = round(time.monotonic() - t, 3)
            rec = {"step": step, "line": line, "kind": "action" if act else "inject", "route": route,
                   "ok": err is None, "error": err, "secs": secs, "frames": env.last_frames if err is None else None}
            record(out, rec, env.state_json() if err is None else None)
            st = env.state()
            log(f"{step:4d} {line:24s} -> {st.get('state'):14s} ante {st.get('ante')} ${st.get('dollars')} "
                f"[{route} {secs:.2f}s {env.last_frames}f]" + (f" ERROR {err}" if err else ""))
            if err and not a.keep_going:
                break
            if oracle:
                cmd = act.harness_lua() if act else inj.harness_lua()  # type: ignore[union-attr]
                ostate = oracle.call(cmd)
                if ofile:
                    ofile.write(ostate + "\n")
                ok = compare(step, line)
            if st.get("state") == "GAME_OVER":
                log("game over")
                break
    fp_after = save_fingerprint()
    log(f"wrote {out_path} in {time.monotonic() - t_all:.1f}s; rejected steps: {errors}"
        + (f"; first divergence at step {diverged}" if diverged is not None else ("; no divergence" if oracle else "")))
    if fp_after != fp_before:
        log(f"WARNING: player save files changed: {fp_before} -> {fp_after}")
    return 1 if (errors or diverged is not None) else 0


if __name__ == "__main__":
    sys.exit(main())
