import zlib
from collections import Counter
from types import SimpleNamespace

import antezero_sim as az
import numpy as np
import torch

from antezero_agent import round_search
from antezero_agent.config import ModelConfig, preset
from antezero_agent.model import Net, Obs, obs_from_numpy
from antezero_agent.round_search import FREE, RoundSearch, _sign_p
from antezero_agent.runtime import make_env, seeded

SEARCH = dict(gate=4, final=4, keep=2, samples=4, threads=2, eg_gate=4, eg_final=4, eg_keep=2)


def tiny():
    torch.manual_seed(0)
    return Net(ModelConfig(d_model=32, n_layers=1, n_heads=2)).eval()


def round_states(n=12):
    """Round states from a few games of the tiny policy."""
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=8, seed=11)
    m, out = tiny(), []
    for _ in range(200):
        for i in range(env.num_envs):
            g = env.game(i)
            if g.phase == "round" and not g.is_terminal() and len(out) < n:
                out.append(g)
        a = m.act(obs_from_numpy(env))
        env.step(a["op"].to(torch.int32).numpy(), a["cards"].numpy().astype(np.uint32), a["dest"].to(torch.int32).numpy())
    assert len(out) == n
    return out


def test_round_search_ignores_the_hidden_arrangement():
    """Permuting the hidden state (deck order, future randomness) changes
    neither the worlds nor the search's decision: worlds come from a
    canonical order, never from the true one."""
    rs = RoundSearch(tiny(), preset("smoke"), **SEARCH)
    risky = 0
    for k, g in enumerate(round_states()):
        h = g.clone()
        h.resample_hidden(f"PERM{k}")  # same visible state, other hidden arrangement
        assert h.canonical() != g.canonical()
        w, v = g.clone(), h.clone()
        w.resample_hidden("T")
        v.resample_hidden("T")
        assert w.canonical() == v.canonical(), "a world depends on the hidden arrangement"
        decisions = []
        for x in (g, h):
            before = x.canonical()
            with seeded(k):  # the same policy samples
                (code, info), = rs.run([_decide(rs, x, f"{k}_0")])
            assert x.canonical() == before, "the search mutated the game it searched"
            decisions.append((code, info))
        assert decisions[0] == decisions[1]
        risky += decisions[0][1]["risky"]
    assert risky >= 3  # the screen and final stages ran


def _decide(rs, g, tag, endgame=False):
    code, d, _a = yield ("act", g)
    return (yield from (rs.decide_endgame if endgame else rs.decide)(tag, g, d, code, Counter()))


def test_round_search_plays_legal_games():
    cfg = preset("smoke")
    rs = RoundSearch(tiny(), cfg, **SEARCH)
    env = make_env(cfg, num_envs=3, seed=5)
    recs = rs.run([rs.play(env.game(i), i) for i in range(3)])
    assert sorted(r["slot"] for r in recs) == [0, 1, 2]
    assert all(r["illegal"] == 0 and not r["truncated"] and r["gated"] > 0 for r in recs)
    g = round_states(1)[0]
    codes = []
    for _ in range(2):
        with seeded(1):
            codes.append(rs.choose(g, "0_0"))
    assert codes[0] == codes[1]
    g.clone().step(codes[0])  # legal (raises otherwise)


def test_eval_round_search_counts_every_game_once():
    from antezero_agent.eval import evaluate_round_search

    res = evaluate_round_search(tiny(), preset("smoke"), episodes=3, batch=2, **SEARCH)
    assert res["summary"]["episodes"] == 3 and res["summary"]["invalid_actions"] == 0
    assert res["search"]["gated_per_game"] > 0 and res["search"]["illegal_per_game"] == 0


def scripted(g):
    """Best scored play, else progress, opening the first booster it can."""
    if g.phase == "round":
        d = g.encode(8)
        n = int(d["n_ops"])
        k = int(np.argmax(np.where(d["ops"][:n, 0] == az.OP_PLAY_TOP, d["ops_f"][:n, 1], -1)))
        return f"play {int(d['ops'][k, 4]) & 0xFFFFFFFF}"
    first = {"shop": "open", "pack": "pick"}.get(g.phase)
    return next((c for c in g.legal_codes() if first and c.startswith(first)), {"shop": "end_shop", "pack": "skip_pack", "blind_select": "select", "round_eval": "cash_out"}[g.phase])


def shop_states(seeds=2):
    """Shop and pack states of antes 1-2 (some packs with targeted picks)
    from scripted games given a head start."""
    out = []
    for i in range(seeds):
        g = az.Game(f"EG{i}", "b_red", 1)
        for code in ("inject_dollars 40", "inject_joker 0 0 0", "inject_joker 5 0 0", "inject_joker 1 0 0"):
            g.inject(code)
        while not g.is_terminal() and g.ante <= 2:
            if g.phase in ("shop", "pack"):
                out.append(g.clone())
            g.step(scripted(g))
    return out


def _ops(rs, g):
    _code, d, _a = yield ("act", g)
    return (yield ("ops", g, d))


def test_endgame_search_triggers_only_in_end_ante_shops_and_packs(monkeypatch):
    cfg = preset("smoke")
    on, off = (RoundSearch(tiny(), cfg, endgame=e, min_ante=99, **dict(SEARCH, eg_gate=2)) for e in (True, False))  # endgame search only
    states = shop_states() + round_states(4)
    assert not any(on.endgame_at(g, "end_shop") for g in states)  # ante 8 only
    monkeypatch.setattr(round_search, "END_ANTE", 2)
    monkeypatch.setattr(on, "eg_min_ante", 2)  # its default (8) does not follow the patched END_ANTE
    monkeypatch.setattr(round_search, "GAME_MAX", 8)  # short rollouts (a world still playing counts as lost)
    for g in states:
        want = g.phase in ("shop", "pack") and g.ante == 2
        assert on.endgame_at(g, "end_shop") == want and not on.endgame_at(g, "move_joker 0 1") and not off.endgame_at(g, "end_shop")
    starts = [g for g in states if g.phase == "shop" and g.ante == 2][:1]
    seen = []
    for rs in (on, off):
        watch = lambda g, n, d, a, greedy, code, info, rs=rs: seen.append((rs is on, g.phase, g.ante, greedy.startswith(FREE), info is not None))
        recs = rs.run([rs.play(g.clone(), i, watch=watch) for i, g in enumerate(starts)])
        assert all(r["illegal"] == 0 for r in recs)
        assert sum(r["eg_gated"] for r in recs) == sum(x[0] and x[1] in ("shop", "pack") and x[2] == 2 and not x[3] for x in seen if x[0] == (rs is on))
    assert all(x[4] == (x[0] and x[1] in ("shop", "pack") and x[2] == 2 and not x[3]) for x in seen if x[1] != "round")
    assert any(x[4] for x in seen if x[1] != "round")


def test_endgame_search_from_an_earlier_ante():
    """``eg_min_ante=7`` (``--endgame-ante 7``): the shops and packs of antes 7
    and 8 before the win; the default stays ante 8 only."""
    on, default = (RoundSearch(tiny(), preset("smoke"), endgame=True, **SEARCH, **kw) for kw in ({"eg_min_ante": 7}, {}))

    def at(rs, phase, ante, won=False, code="end_shop", legal=("select_blind",)):
        return rs.endgame_at(SimpleNamespace(phase=phase, ante=ante, run_won=won, legal_codes=lambda: list(legal)), code)

    assert at(on, "shop", 7) and at(on, "pack", 7) and at(on, "shop", 8) and at(on, "pack", 8)
    assert not any((at(on, "shop", 6), at(on, "round", 7), at(on, "blind_select", 7), at(on, "shop", 9), at(on, "shop", 7, won=True), at(on, "shop", 7, code="move_joker 0 1")))
    assert at(default, "shop", 8) and not at(default, "shop", 7) and not at(default, "pack", 7)
    assert at(on, "blind_select", 7, legal=("select_blind", "reroll_boss")) and not at(on, "blind_select", 6, legal=("select_blind", "reroll_boss"))


def test_endgame_search_weighs_a_boss_reroll(monkeypatch):
    """With a boss reroll legal (Director's Cut and the money), the blind select is an endgame decision too, and the
    reroll is one of the choices it compares."""
    rs = RoundSearch(tiny(), preset("smoke"), endgame=True, min_ante=99, **dict(SEARCH, eg_gate=2))
    monkeypatch.setattr(round_search, "END_ANTE", 2)
    monkeypatch.setattr(rs, "eg_min_ante", 2)
    monkeypatch.setattr(round_search, "GAME_MAX", 8)
    g = next(g for g in shop_states() if g.phase == "shop" and g.ante == 2).clone()
    g.step("end_shop")
    assert g.phase == "blind_select" and not rs.endgame_at(g, "select_blind")
    g.inject("inject_voucher 28")  # Director's Cut
    g.inject("inject_dollars 50")
    assert rs.endgame_at(g, "select_blind") and not rs.endgame_at(g, "move_joker 0 1")
    seen = []
    rec = rs.run([rs.play(g.clone(), 0, watch=lambda g, n, d, a, greedy, code, info: seen.append((g.phase, info, code)))])[0]
    assert rec["illegal"] == 0 and rec["eg_gated"] >= 1 and seen[0][0] == "blind_select" and seen[0][1] is not None
    assert "reroll_boss" in rs.run([_ops(rs, g)])[0]


def test_endgame_search_ignores_the_hidden_arrangement(monkeypatch):
    """The endgame search at shop and pack states: worlds and decisions do
    not depend on the hidden arrangement, and repeat exactly given the tags
    (no policy sampling)."""
    monkeypatch.setattr(round_search, "GAME_MAX", 30)
    rs = RoundSearch(tiny(), preset("smoke"), endgame=True, **SEARCH)
    risky = 0
    states = shop_states()
    for k, g in enumerate([g for g in states if g.phase == "shop"][:2] + [g for g in states if g.phase == "pack"][:2]):
        h = g.clone()
        h.resample_hidden(f"PERM{k}")
        assert h.canonical() != g.canonical()
        w, v = g.clone(), h.clone()
        w.resample_hidden("T")
        v.resample_hidden("T")
        assert w.canonical() == v.canonical(), "a world depends on the hidden arrangement"
        decisions = []
        for x in (g, h, g):
            before = x.canonical()
            ((code, info),) = rs.run([_decide(rs, x, f"{k}_0", endgame=True)])
            assert x.canonical() == before, "the search mutated the game it searched"
            decisions.append((code, info))
        assert decisions[0] == decisions[1] == decisions[2]
        g.clone().step(decisions[0][0])  # legal (raises otherwise)
        risky += decisions[0][1]["risky"]
    assert risky >= 3 and rs.rows["eg_screen"] > 0


def test_endgame_decisions_ignore_the_hidden_arrangement(monkeypatch):
    """The tiny policy loses every world, so above no finalist forms and the
    decision is greedy whatever the worlds. Here a world's outcome is a hash
    of its full state (RNG streams, deck order), so decisions depend on the
    worlds through the final stage, and must still ignore the hidden
    arrangement."""

    def rule(self, ws, label, to_end=False):
        return [(zlib.crc32(repr(w.canonical()).encode()) % 2 == 0, 0.0) for w in ws]
        yield  # a generator, like rollout

    monkeypatch.setattr(RoundSearch, "rollout", rule)
    rs = RoundSearch(tiny(), preset("smoke"), endgame=True, **dict(SEARCH, eg_final=12))
    finals = 0
    for k, g in enumerate(shop_states()):
        h = g.clone()
        h.resample_hidden(f"PERM{k}")
        a, b = (rs.run([_decide(rs, x, f"{k}_0", endgame=True)])[0] for x in (g, h))
        assert a == b, "an endgame decision depends on the hidden arrangement"
        finals += "final_worlds" in a[1]
    assert finals >= 3


def test_endgame_candidates_are_the_legal_nonfree_ops():
    """One legal code per non-free op: every untargeted legal action, and a
    pointer-picked card set for targeted pack picks / uses."""
    rs = RoundSearch(tiny(), preset("smoke"), endgame=True, **SEARCH)
    targeted = 0
    for g in shop_states():
        legal = set(g.legal_codes())  # no rearranges: they are not enumerated
        (codes,) = rs.run([_ops(rs, g)])
        assert codes and len(set(codes)) == len(codes) and set(codes) <= legal
        assert {c for c in legal if c.split()[0] not in ("use", "pick", "buy_use") or c.endswith(" 0")} <= set(codes)
        targeted += any(c.split()[0] in ("use", "pick") and not c.endswith(" 0") for c in codes)
    assert targeted


def test_eval_endgame_search_counts_every_game_once(monkeypatch):
    from antezero_agent.eval import evaluate_round_search

    monkeypatch.setattr(round_search, "END_ANTE", 1)
    res = evaluate_round_search(tiny(), preset("smoke"), episodes=3, batch=2, endgame=True, eg_min_ante=1, **SEARCH)
    assert res["summary"]["episodes"] == 3 and res["summary"]["invalid_actions"] == 0
    assert res["search"]["eg_gated_per_game"] > 0 and res["search"]["illegal_per_game"] == 0


def test_endgame_search_overrides_greedy_only_on_significant_wins(monkeypatch):
    """Rollouts replaced by a rule (a world is won iff its money differs
    from greedy's after the action): every other finalist wins every world
    greedy loses, which is significant on 12 final worlds, not on 4."""
    g = [s for s in shop_states() if s.phase == "shop"][0]
    gcode = RoundSearch(tiny(), preset("smoke"), **SEARCH).choose(g, "0_0")  # greedy: no search in shops
    after = g.clone()
    after.step(gcode)

    def rule(self, ws, label, to_end=False):
        return [(w.dollars != after.dollars, 0.0) for w in ws]
        yield  # a generator, like rollout

    monkeypatch.setattr(RoundSearch, "rollout", rule)
    for final, override in ((4, False), (12, True)):
        rs = RoundSearch(tiny(), preset("smoke"), endgame=True, **dict(SEARCH, eg_final=final))
        ((code, info),) = rs.run([_decide(rs, g, "0_0", endgame=True)])
        assert info["risky"] and info["final_greedy"] == 0 and info["final_best"] == final
        assert (code != gcode) == override and rs.rows["ops"] > 0


def test_sign_test():
    w, l = [True] * 6, [False] * 6
    assert _sign_p(w, l) == 1 / 64 and _sign_p(l, w) == 1.0 and _sign_p(w, w) == 1.0
    assert _sign_p(w + [False], l + [True]) == 8 / 128  # 6 of the 7 worlds only one of them wins


def _each_op(o):
    """Every op of every row alone (as ``RoundSearch._complete`` asks): the
    card pointer and the destination run for every op kind."""
    b, j = (torch.arange(o.ops.shape[1])[None] < o.n_ops[:, None]).nonzero().T
    return Obs(o.tok_i[b], o.tok_f[b], o.glob_i[b], o.glob_f[b], o.ops[b, j][:, None], o.ops_f[b, j][:, None], torch.ones_like(b), o.n_tokens[b], o.n_hand[b])


def test_an_ensemble_of_one_model_plays_like_it(tmp_path):
    """``a.pt+b.pt+c.pt`` loads a greedy :class:`Ens`: with identical members
    it picks the single model's actions exactly, for its chosen op and for
    every op alone (discards, targeted uses and picks, hand and joker moves
    with their destination); samples (the round search's proposals) are
    member 0's."""
    from antezero_agent.ensemble import Ens
    from antezero_agent.runtime import load_model, save_checkpoint

    cfg, net = preset("smoke"), tiny()
    paths = [tmp_path / f"{x}.pt" for x in "abc"]
    for p in paths:
        save_checkpoint(p, net, torch.optim.Adam(net.parameters()), cfg, 1, 1)
    ens, _, meta = load_model("+".join(map(str, paths)))
    assert isinstance(ens, Ens) and len(ens.members) == 3 and meta["update"] == 1

    def same(o):
        a, b = net.act(o, deterministic=True), ens.act(o, deterministic=True)
        assert all(torch.equal(a[k], b[k]) for k in ("op", "cards", "card_seq", "dest"))
        return a

    for g in shop_states():
        same(_each_op(obs_from_numpy(g.encode(cfg.env.play_candidates, cfg.env.deck_tokens, cfg.env.build_previews))))
    env = make_env(cfg, num_envs=8)
    for _ in range(60):
        o = obs_from_numpy(env)
        same(_each_op(o))
        a = same(o)
        with seeded(7, "cpu"):
            s0 = net.act(o)
        with seeded(7, "cpu"):
            s1 = ens.act(o)
        assert all(torch.equal(s0[k], s1[k]) for k in ("op", "cards", "dest"))
        env.step(a["op"].to(torch.int32).numpy(), a["cards"].numpy().astype(np.uint32), a["dest"].to(torch.int32).numpy())
