"""Offline tests: action/inject codes, harness translation, canon formatting,
and the parts of the real-game stack that work without the game."""

import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest

from antezero_bridge import _codes
from antezero_bridge.actions import ARITY, Action, mask_indices, move_permutation
from antezero_bridge.canon import diff, fmt, from_lua
from antezero_bridge.client import BridgeClient, ProtocolError, RPCError
from antezero_bridge.compare import main as compare_main
from antezero_bridge.env import Env, back_key, stake_num
from antezero_bridge.injects import Inject

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "code",
    ["select", "skip", "reroll_boss", "play 30", "discard 158", "use 0 0", "use 1 6", "sell_joker 2",
     "sell_consumable 1", "move_joker 0 4", "move_hand 7 5", "sort 1", "sort 0", "buy 1", "buy_use 0 3",
     "redeem 0", "open 1", "reroll", "end_shop", "cash_out", "pick 1 0", "skip_pack"],
)
def test_roundtrip(code):
    a = Action.parse(code)
    assert a is not None and a.code() == code


def test_parse_rules():
    for bad in ["play", "use 0", "inject_joker 1 0 0", "sort 7", "play 30 extra", "sell_joker 256", "play -1", "play 4294967296", "play 1_0"]:
        assert Action.parse(bad) is None, bad
    assert Action.parse("  play   30 ").code() == "play 30"
    assert Action.parse("play +3").code() == "play 3"
    with pytest.raises(ValueError):
        Action.play(-1)
    with pytest.raises(ValueError):
        Action.sell_joker(256)
    assert set(ARITY) == {
        "select", "skip", "reroll_boss", "play", "discard", "use", "sell_joker", "sell_consumable", "move_joker",
        "move_hand", "sort", "buy", "buy_use", "redeem", "open", "reroll", "end_shop", "cash_out", "pick", "skip_pack",
    }


ACTION_CORPUS = [
    "select", "select 1", "skip", "play 30", "play", "play 30 1", "play 4294967295", "play 4294967296", "play -1",
    "play +5", "play 0x1f", "play 3.0", "discard 7", "use 0 0", "use 255 7", "use 256 7", "use 0", "sell_joker 0",
    "sell_joker 300", "move_joker 1 2", "move_joker 1 999", "move_hand 0 1", "sort 0", "sort 1", "sort 2", "buy 3",
    "buy_use 1 8", "redeem 0", "open 1", "reroll", "end_shop", "cash_out", "pick 0 0", "pick 0", "skip_pack", "jump",
    "", "   ", "PLAY 3", "play\t3",
]
INJECT_CORPUS = [
    "inject_joker 17 3 1", "inject_joker 149 0 0", "inject_joker 150 0 0", "inject_joker 1 5 0", "inject_joker 1 0 300",
    "inject_joker 1 0 0 9", "inject_joker", "inject_joker -1 0 0", "inject_cons 0 21 0", "inject_cons 1 12 0",
    "inject_cons 2 17 4", "inject_cons 3 0 0", "inject_cons 1.5 2 0", "inject_card 0 2 0 0 0", "inject_card 3 14 8 4 4",
    "inject_card 4 2 0 0 0", "inject_card 0 15 0 0 0", "inject_card 0 1 0 0 0", "inject_card 0 2 9 0 0",
    "inject_card 0 2 0 0 5", "inject_dollars 19", "inject_dollars 17.5", "inject_dollars -3", "inject_dollars .5",
    "inject_dollars nan", "inject_dollars inf", "inject_dollars 1e3", "inject_dollars 1_0", "inject_voucher 31",
    "inject_voucher 32", "inject_tag 1",
]


def test_codes_match_rust_parsers():
    """The Python codecs accept exactly what the simulator's parsers accept."""
    az = pytest.importorskip("antezero_sim")
    for code in ACTION_CORPUS:
        g = az.Game("CODES", "b_red", 1)
        try:
            g.step(code)
            rust = True
        except ValueError:
            rust = False
        except RuntimeError:  # parsed, but not legal now
            rust = True
        assert (Action.parse(code) is not None) == rust, code
    for code in INJECT_CORPUS:
        g = az.Game("CODES", "b_red", 1)
        g.step("select")
        try:
            g.inject(code)
            rust = True
        except ValueError:
            rust = False
        assert (Inject.parse(code) is not None) == rust, code


def test_generated_tables_match_rust():
    """Every index of the generated tables names the card the simulator
    injects for it."""
    az = pytest.importorskip("antezero_sim")
    for i, key in enumerate(_codes.JOKERS):
        g = az.Game("TABLES", "b_red", 1)
        g.inject(f"inject_joker {i} 0 0")
        assert g.canonical()["jokers.0.key"] == key == Inject.parse(f"inject_joker {i} 0 0").harness()["key"]
    for t, table in enumerate((_codes.TAROTS, _codes.PLANETS, _codes.SPECTRALS)):
        for i, key in enumerate(table):
            g = az.Game("TABLES", "b_red", 1)
            g.inject(f"inject_cons {t} {i} 0")
            assert g.canonical()["consumables.0.key"] == key == Inject.parse(f"inject_cons {t} {i} 0").harness()["key"]
    for i, key in enumerate(_codes.VOUCHERS):
        g = az.Game("TABLES", "b_red", 1)
        g.inject(f"inject_voucher {i}")
        assert key in g.canonical()["used_vouchers"].split(",")
    assert [b for b in _codes.BACKS if b != "b_challenge"] == az.backs()


def test_generated_tables_are_current():
    spec = importlib.util.spec_from_file_location("gen_codes", ROOT / "bridge" / "scripts" / "gen_codes.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.OUT.read_text(encoding="utf-8") == gen.render(gen.SRC.read_text(encoding="utf-8"))


def _ocli() -> Path:
    exe = ROOT / "target" / "release" / ("ocli.exe" if os.name == "nt" else "ocli")
    if not exe.exists():
        pytest.skip("needs the oracle CLI (cargo build --release -p oracle) and the game's lua51.dll")
    return exe


def test_canon_port_matches_rust_on_oracle_dumps():
    """``canon.from_lua`` of the oracle's state dumps equals the simulator's
    canonical view along a game (the Rust canonicalizers agree on these)."""
    az = pytest.importorskip("antezero_sim")
    from antezero_bridge.actions import lua_value
    from antezero_bridge.replay import Oracle

    oracle = Oracle(str(_ocli()), timeout=120)
    try:
        steps = 0
        for n, (back, stake) in enumerate([("b_blue", 2), ("b_red", 1), ("b_ghost", 5), ("b_plasma", 3), ("b_zodiac", 7)]):
            seed = f"CANONPY{n}"
            try:
                dump = oracle.call(f"H.start({lua_value(seed)},{lua_value(back)},{stake})")
            except RuntimeError as e:
                pytest.skip(f"oracle unavailable: {e}")
            g = az.Game(seed, back, stake)
            for k in range(80):
                assert diff(from_lua(dump), dict(g.canonical())) == [], f"{seed} after {k} commands"
                if g.is_terminal():
                    break
                if k % 7 == 3:
                    code = "inject_joker 17 3 0" if k % 2 else "inject_card 1 12 1 2 3"
                    g.inject(code)
                    dump = oracle.call(Inject.parse(code).harness_lua())
                    continue
                codes = g.legal_codes()
                plays = [c for c in codes if c.startswith("play ")]
                code = plays[-1] if plays else codes[k % len(codes)]
                g.step(code)
                dump = oracle.call(Action.parse(code).harness_lua())
                steps += 1
        assert steps > 40, steps
    finally:
        oracle.close()


def test_harness_tables_match_diff_rs():
    # sim/oracle/src/diff.rs action_to_lua (1-based indices, cards = mask bits + 1)
    assert Action.parse("play 30").harness_lua() == "H.act({type='play', cards={2,3,4,5}})"
    assert Action.parse("use 0 0").harness_lua() == "H.act({type='use', index=1, cards={}})"
    assert Action.parse("buy_use 2 1").harness_lua() == "H.act({type='buy_and_use', index=3, cards={1}})"
    assert Action.parse("move_joker 0 4").harness_lua() == "H.act({type='move_joker', from=1, to=5})"
    assert Action.parse("sort 1").harness_lua() == "H.act({type='sort_hand', by='suit'})"
    assert Action.parse("select").harness_lua() == "H.act({type='select_blind'})"


def test_balatrobot_mapping():
    assert Action.parse("play 30").balatrobot() == ("play", {"cards": [1, 2, 3, 4]})
    assert Action.parse("use 1 0").balatrobot() == ("use", {"consumable": 1})
    assert Action.parse("pick 0 5").balatrobot() == ("pack", {"card": 0, "targets": [0, 2]})
    assert Action.parse("move_joker 0 2").balatrobot(n_jokers=3) == ("rearrange", {"jokers": [1, 2, 0]})
    assert Action.parse("move_hand 3 0").balatrobot(n_hand=4) == ("rearrange", {"hand": [3, 0, 1, 2]})
    for code in ("reroll_boss", "sort 0", "buy_use 0 0"):
        assert Action.parse(code).balatrobot() is None


def test_move_permutation_is_remove_insert():
    for n in range(2, 6):
        for f in range(n):
            for t in range(n):
                old = list("abcdef"[:n])
                new = old.copy()
                new.insert(t, new.pop(f))
                perm = move_permutation(n, f, t)
                assert [old[i] for i in perm] == new


def test_masks():
    assert mask_indices(0b10110) == [1, 2, 4]


def test_injects():
    assert len(_codes.JOKERS) == 150 and len(_codes.VOUCHERS) == 32
    j = Inject.parse("inject_joker 17 3 1")
    assert j.harness() == {"kind": "joker", "key": "j_four_fingers", "edition": "polychrome", "eternal": True}
    assert j.harness_lua() == "H.inject({kind='joker', key='j_four_fingers', edition='polychrome', eternal=true})"
    c = Inject.parse("inject_cons 1 2 0")
    assert c.harness() == {"kind": "consumable", "key": "c_earth"}
    card = Inject.parse("inject_card 1 12 1 0 0")
    assert card.harness() == {"kind": "card", "key": "H_Q", "center": "m_bonus"}
    card = Inject.parse("inject_card 0 2 0 0 4")
    assert card.harness() == {"kind": "card", "key": "S_2", "center": "c_base", "seal": "Purple"}
    assert Inject.parse("inject_dollars 19").harness() == {"kind": "dollars", "amount": 19}
    assert Inject.parse("inject_voucher 0").harness() == {"kind": "voucher", "key": "v_overstock_norm"}
    for bad in ["play 3", "inject_joker", "inject_joker -1 0 0", "inject_joker garbage 17 0 0", "inject_cons 1.5 2 0", "inject_dollars nan"]:
        assert Inject.parse(bad) is None, bad


def test_decks_and_stakes():
    assert back_key("red") == back_key("RED") == back_key("b_red") == "b_red"
    assert stake_num("GOLD") == stake_num("8") == stake_num(8) == 8
    for bad in ("b_challenge", "b_nope", "CHALLENGE"):
        with pytest.raises(ValueError):
            back_key(bad)
    for bad in (0, 9, "9", "TIN"):
        with pytest.raises(ValueError):
            stake_num(bad)


def test_sim_env_implements_env():
    pytest.importorskip("antezero_sim")
    from antezero_bridge.sim_env import SimEnv

    env = SimEnv()
    assert isinstance(env, Env)
    env.reset("SIMENV", "RED", "WHITE")
    assert Action.parse("select") in env.legal_actions()
    env.step("select")
    assert env.canonical()["state"] == "SELECTING_HAND" and not env.is_terminal()


def test_fmt_matches_rust_display():
    assert fmt(5.0) == "5"
    assert fmt(-0.0) == "-0"
    assert fmt(0.1) == "0.1"
    assert fmt(1e21) == "1000000000000000000000"
    assert fmt(1.5e-7) == "0.00000015"
    assert fmt(0.6813263181887) == "0.6813263181887"
    assert fmt(float("nan")) == "nan"
    assert fmt(float("inf")) == "inf"


def test_from_lua_minimal():
    s = {"state": "BLIND_SELECT", "dollars": 4, "blind_tags": [], "used_vouchers": [], "used_jokers": ["c_base", "j_joker"],
         "hand": [], "deck": [{"key": "c_base", "sort_id": 3, "card": "H_A", "ability": {"set": "Default", "perma_bonus": 0}}],
         "rng": {"seed": "X", "hashed_seed": 0.5, "shuffle": 0.25}}
    c = from_lua(json.dumps(s))
    assert c["dollars"] == "4" and c["blind_tags.Small"] == "nil" and c["used_jokers"] == "j_joker"
    assert c["deck.0.card"] == "H_A" and c["deck.0.id"] == "3" and c["rng.shuffle"] == "0.25" and "rng.seed" not in c
    assert diff(c, dict(c, dollars="5")) == [("dollars", "4", "5")]


def test_compare_fails_on_any_divergence(tmp_path, capsys):
    s = {"state": "BLIND_SELECT", "dollars": 4}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    def write(p, lines):
        p.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")

    write(a, [s, s])
    write(b, [s, s])
    assert compare_main([str(a), str(b)]) == 0
    write(b, [s])  # a missing tail
    assert compare_main([str(a), str(b)]) == 1
    write(b, [s, dict(s, dollars=5)])
    assert compare_main([str(a), str(b)]) == 1
    rec = lambda line, st, err=None: {"step": 1, "line": line, "ok": err is None, "error": err, "state": st}  # noqa: E731
    write(a, [rec("play 3", s)])
    write(b, [rec("play 5", s)])  # different history
    assert compare_main([str(a), str(b)]) == 1
    write(b, [rec("play 3", None, "rejected")])  # rejected step
    assert compare_main([str(a), str(b)]) == 1
    capsys.readouterr()


def test_launch_env_overrides_inherited_switches(monkeypatch):
    from antezero_bridge.launch import FAST, PARITY, child_env

    monkeypatch.setenv("BALATROBOT_FAST", "1")
    monkeypatch.setenv("BALATROBOT_HEADLESS", "1")
    e = child_env(PARITY)
    assert e["BALATROBOT_FAST"] == "0" and e["BALATROBOT_HEADLESS"] == "0" and e["BALATROBOT_GAMESPEED"] == "1"
    assert e["AZB_ALLOW_SAVES"] == "0"
    e = child_env(FAST)
    assert e["BALATROBOT_FAST"] == "1" and e["BALATROBOT_RENDER_ON_API"] == "0"


def test_companion_install_is_staged(tmp_path):
    from antezero_bridge.launch import install_companion_mod

    src, mods = tmp_path / "src", tmp_path / "Mods"
    (src / "src").mkdir(parents=True)
    (src / "AnteZero.json").write_text("{}")
    (src / "src" / "a.lua").write_text("return 1")
    (mods / "antezero_bridge").mkdir(parents=True)  # the mod under its old name: both would load
    dst = install_companion_mod(src, mods)
    assert (dst / "src" / "a.lua").read_text() == "return 1"
    (src / "src" / "a.lua").write_text("return 2")
    install_companion_mod(src, mods)
    assert (dst / "src" / "a.lua").read_text() == "return 2"
    assert sorted(p.name for p in mods.iterdir()) == ["AnteZero"]


def test_balatrobot_loads_only_inside_the_tools_launch(tmp_path, monkeypatch):
    launch = importlib.import_module("antezero_bridge.launch")  # the package's `launch` is the function

    flag = tmp_path / "balatrobot" / ".lovelyignore"
    flag.parent.mkdir()
    flag.touch()
    monkeypatch.setattr(launch, "BB_DISABLED", flag)
    with launch.balatrobot_enabled():
        assert not flag.exists()
    assert flag.exists()


def _client(handler) -> BridgeClient:
    c = BridgeClient()
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_bot_calls_carry_the_run_and_outwait_a_menu():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen[body["method"]] = (body["params"], req.extensions["timeout"]["read"])
        result = {"state": "{}", "settled": True} if body["method"] == "az_act" else {"active": True}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    c = _client(handler)
    c.az_act({"type": "reroll"}, settle_timeout=10.0, run=3, note="90% sure")
    assert seen["az_act"] == ({"timeout": 10.0, "run": 3, "note": "90% sure", "action": {"type": "reroll"}}, 60.0)
    c.patience = 1000.0  # a watched bot run: the player's menus hold the settle waits
    c.az_act({"type": "reroll"}, settle_timeout=10.0)
    assert seen["az_act"][1] == 1060.0
    assert c.az_bot(run=3, stop="lost the run") == {"active": True} and seen["az_bot"][0] == {"run": 3, "stop": "lost the run"}


def test_client_checks_the_jsonrpc_envelope():
    def ok(req):
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": json.loads(req.content)["id"], "result": {"status": "ok"}})

    assert _client(ok).call("health") == {"status": "ok"}

    def wrong_id(req):
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 999, "result": {}})

    with pytest.raises(ProtocolError):
        _client(wrong_id).call("health")

    def error(req):
        rid = json.loads(req.content)["id"]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": rid, "error": {"code": 1, "message": "no", "data": {"name": "BAD_REQUEST"}}})

    with pytest.raises(RPCError) as e:
        _client(error).call("play")
    assert e.value.name == "BAD_REQUEST"

    def no_state(req):
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": json.loads(req.content)["id"], "result": {}})

    with pytest.raises(ProtocolError):
        _client(no_state).az_state()
