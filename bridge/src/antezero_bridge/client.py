"""Thin typed JSON-RPC client for BalatroBot + the antezero companion mod.

BalatroBot serves JSON-RPC 2.0 over HTTP (one client at a time, one request
per connection). Its own ``balatrobot.cli.client.BalatroClient`` has a fixed
timeout and no knowledge of the companion methods, so this client wraps the
same protocol with per-call timeouts and typed helpers; errors are raised as
:class:`RPCError`, a subclass of BalatroBot's ``APIError``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from balatrobot.cli.client import APIError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 12346


class RPCError(APIError):
    """A JSON-RPC error response (``name`` is BalatroBot's error name)."""


class ProtocolError(RuntimeError):
    """The server's reply is not a JSON-RPC response to the request."""


class SettleTimeout(RuntimeError):
    """The game did not reach a stable decision point in time."""

    def __init__(self, diag: dict[str, Any]):
        self.diag = diag
        super().__init__(f"game did not settle: {json.dumps(diag, sort_keys=True)}")


@dataclass(slots=True)
class StateDump:
    """One oracle-shaped state dump.

    ``raw`` is the exact JSON text produced in-game (harness encoder: sorted
    keys, ``%.17g`` numbers) and is what should be written to disk / fed to
    ``canon::from_lua``; ``data`` is its parsed form.
    """

    raw: str
    settled: bool
    reason: str
    frames: int
    diag: dict[str, Any] | None = None
    _data: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def data(self) -> dict[str, Any]:
        if self._data is None:
            self._data = json.loads(self.raw)
        return self._data


class BridgeClient:
    """Calls BalatroBot (native methods) and the companion ``az_*`` methods."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._ids = itertools.count(1)
        # Extra seconds to wait for a settling call: in a watched bot run the
        # player's menus hold the game's settle waits for as long as they stay open.
        self.patience = 0.0
        self._http = httpx.Client(timeout=timeout)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def close(self) -> None:
        self._http.close()

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        rid = next(self._ids)
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": rid}
        r = self._http.post(self.url, json=payload, timeout=timeout or self.timeout)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError as e:
            raise ProtocolError(f"{method}: reply is not JSON") from e
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0" or data.get("id") != rid or ("result" in data) == ("error" in data):
            raise ProtocolError(f"{method}: not a JSON-RPC response to request {rid}: {str(data)[:200]}")
        if "error" in data:
            e = data["error"] if isinstance(data["error"], dict) else {}
            raise RPCError(name=(e.get("data") or {}).get("name", "ERROR"), message=e.get("message", ""), code=e.get("code", 0))
        return data["result"]

    def _result(self, method: str, params: dict[str, Any], timeout: float | None, *keys: str) -> dict[str, Any]:
        """``call`` for methods whose result is an object with ``keys``."""
        res = self.call(method, params, timeout=timeout)
        if not isinstance(res, dict) or any(k not in res for k in keys):
            raise ProtocolError(f"{method}: unexpected result {str(res)[:200]}")
        return res

    # -- BalatroBot -------------------------------------------------------
    def health(self, timeout: float = 2.0) -> bool:
        try:
            return self.call("health", timeout=timeout).get("status") == "ok"
        except (httpx.HTTPError, RPCError, ValueError):
            return False

    def gamestate(self) -> dict[str, Any]:
        return self.call("gamestate")

    # -- companion mod ----------------------------------------------------
    def az_info(self, timeout: float | None = None) -> dict[str, Any]:
        return self._result("az_info", {}, timeout, "version")

    def _dump(self, res: dict[str, Any], require_settled: bool) -> StateDump:
        d = StateDump(
            raw=res["state"],
            settled=bool(res.get("settled")),
            reason=str(res.get("reason", "")),
            frames=int(res.get("frames") or 0),
            diag=res.get("diag"),
        )
        if require_settled and not d.settled and d.reason == "timeout":
            raise SettleTimeout(d.diag or {})
        return d

    def _t(self, settle_timeout: float | None, phases: int = 1) -> tuple[dict[str, Any], float | None]:
        """Params and HTTP timeout for a call that settles up to ``phases``
        times in-game, each within ``settle_timeout``."""
        if settle_timeout is None:
            return {}, None
        return {"timeout": float(settle_timeout)}, phases * settle_timeout + 30.0 + self.patience

    def az_state(self, settle: bool = True, settle_timeout: float | None = None) -> StateDump:
        p, t = self._t(settle_timeout)
        p["settle"] = settle
        return self._dump(self._result("az_state", p, t, "state"), settle)

    def az_start(self, seed: str, back: str = "b_red", stake: int = 1, settle_timeout: float | None = None) -> StateDump:
        p, t = self._t(settle_timeout)
        p.update(seed=seed, back=back, stake=int(stake))
        return self._dump(self._result("az_start", p, t, "state"), True)

    def az_act(self, action: dict[str, Any], settle_timeout: float | None = None, **bot: Any) -> StateDump:
        """``bot``: ``run`` (the bot run the action is for; refused once that
        run is over) and ``note`` (shown with the action)."""
        # settle before, after selecting the cards, and after the action
        p, t = self._t(settle_timeout, phases=3)
        p.update(bot, action=action)
        return self._dump(self._result("az_act", p, t, "state"), True)

    def az_bot(self, **params: Any) -> dict[str, Any]:
        """The bot run to play (``run``: id, seed, back, stake; while
        ``active``) and whether a menu pauses it. ``run`` + ``status`` or
        ``stop``: the text to show for that run."""
        return self._result("az_bot", params, 10.0, "active")

    def az_inject(self, inject: dict[str, Any], settle_timeout: float | None = None) -> StateDump:
        p, t = self._t(settle_timeout, phases=2)
        p["inject"] = inject
        return self._dump(self._result("az_inject", p, t, "state"), True)

    def az_legal(self, settle_timeout: float | None = None) -> tuple[list[str], str | None]:
        p, t = self._t(settle_timeout)
        res = self._result("az_legal", p, t, "codes_json")
        return json.loads(res["codes_json"]), res.get("phase")

    def az_eval(self, code: str) -> Any:
        """Evaluate Lua in-game (requires the game to run with AZB_ALLOW_EVAL=1)."""
        res = self.call("az_eval", {"code": code})
        try:
            return json.loads(res["result_json"])
        except (TypeError, ValueError):
            return res.get("result_json")
