import csv

import numpy as np
import pytest
import torch

import antezero_sim as az
from antezero_agent.config import Config, ModelConfig, parse_overrides, preset
from antezero_agent.model import Net, Obs, obs_from_numpy
from antezero_agent.runtime import load_model, make_env, read_checkpoint
from antezero_agent.search import shaped_reward
from antezero_agent.stats import wilson
from antezero_agent.train import LOG_FIELDS, Rollout, gae, train

OP = {"select": 0, "cash_out": 14, "end_shop": 13, "skip_pack": 16}


def tiny(d=32, layers=1):
    torch.manual_seed(0)
    return Net(ModelConfig(d_model=d, n_layers=layers, n_heads=2)).eval()


def step(env, out):
    return env.step(
        out["op"].to(torch.int32).numpy(),
        out["cards"].numpy().astype(np.uint32),
        out["dest"].to(torch.int32).numpy(),
    )


def heuristic_choice(env):
    """Best scored play when offered, otherwise progress (select blind,
    cash out, leave shop / pack): reaches later antes without a model."""
    ops = np.asarray(env.ops)
    n_ops = np.asarray(env.n_ops)
    N = env.num_envs
    op = np.zeros(N, np.int32)
    cards = np.zeros(N, np.uint32)
    for i in range(N):
        kinds = ops[i, : n_ops[i], 0]
        top = np.flatnonzero(kinds == az.OP_PLAY_TOP)
        if len(top):
            op[i] = top[0]
            cards[i] = np.int32(ops[i, top[0], 5]).view(np.uint32)
            continue
        for k in (OP["select"], OP["cash_out"], OP["end_shop"], OP["skip_pack"]):
            hit = np.flatnonzero(kinds == k)
            if len(hit):
                op[i] = hit[0]
                break
    return op, cards


def heuristic(env):
    op, cards = heuristic_choice(env)
    return env.step(op, cards, np.zeros(env.num_envs, np.int32))


def test_sampled_actions_are_always_legal():
    cfg = preset("smoke")
    cfg.env.num_envs = 32
    env = make_env(cfg)
    m = tiny()
    for _ in range(300):
        out = m.act(obs_from_numpy(env))
        step(env, out)
    assert env.invalid_actions() == 0
    assert len(env.pop_episodes()) > 0


def test_teacher_forced_logp_matches_sampling():
    cfg = preset("smoke")
    cfg.env.num_envs = 64
    env = make_env(cfg)
    m = tiny()
    for _ in range(40):
        o = obs_from_numpy(env)
        out = m.act(o)
        with torch.no_grad():
            logp, ent, val = m.evaluate(o, out["op"], out["card_seq"], out["dest"])
        assert torch.allclose(logp, out["logp"], atol=1e-4), (logp - out["logp"]).abs().max()
        assert torch.allclose(val, out["value"], atol=1e-5)
        assert (ent >= -1e-5).all()
        step(env, out)


def test_shaping_telescopes_to_the_true_return():
    """Shaped rewards add no objective: over a whole episode their
    discounted sum is the discounted true return (Phi(start) = 0,
    Phi(terminal) = 0), whose only reward is the final one."""
    for gamma in (1.0, 0.99):
        cfg = preset("smoke")
        env = make_env(cfg, num_envs=32, gamma=gamma, r_loss=0.5, play_candidates=4)
        acc, t = np.zeros(32), np.zeros(32)
        checked = 0
        for _ in range(1500):
            r, done, trunc = heuristic(env)
            acc += gamma**t * r
            t += 1
            for ep in env.pop_episodes():
                i = ep[0]
                if not ep[9]:
                    assert abs(acc[i] - gamma ** (ep[7] - 1) * ep[8]) < 1e-3, (gamma, acc[i], ep)
                    checked += 1
                acc[i], t[i] = 0.0, 0
        assert checked >= 20


def test_search_reward_matches_vecenv():
    """``search.shaped_reward`` is the reward ``VecEnv`` gives, terminal
    transitions included."""
    cfg = preset("smoke")
    cfg.env.play_candidates = 4
    cfg.env.r_loss = 0.5
    env = make_env(cfg, num_envs=1, seed=3)
    g = env.game(0)
    ends = 0
    for _ in range(800):
        op, cards = heuristic_choice(env)
        before = g.clone()
        g.step_factored(int(op[0]), int(cards[0]), 0, 4)
        r, done, _trunc = env.step(op, cards, np.zeros(1, np.int32))
        assert abs(float(r[0]) - shaped_reward(before, g, cfg)) < 1e-5
        if done[0]:
            ends += 1
            g = env.game(0)
    assert ends >= 2


def test_gae_stops_at_episode_ends():
    r = Rollout(3, 1, torch.device("cpu"))
    r.reward[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    r.value[:, 0] = 0.5
    r.done[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    adv, ret = gae(r, torch.tensor([10.0]), gamma=0.9, lam=0.8)
    # t=2: 3 + 0.9*10 - 0.5; t=1 ends an episode: 2 - 0.5 (no leak from
    # t=2); t=0: 1 + 0.9*0.5 - 0.5 + 0.9*0.8*1.5
    assert torch.allclose(adv[:, 0], torch.tensor([2.03, 1.5, 11.5]))
    assert torch.allclose(ret[:, 0], torch.tensor([2.53, 2.0, 12.0]))


def test_truncation_bootstraps_from_final_observation():
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=8, step_limit=7)
    m = tiny()
    seen = False
    for _ in range(20):
        _r, done, trunc = step(env, m.act(obs_from_numpy(env)))
        assert not (trunc & ~done).any()
        if trunc.any():
            rows = np.flatnonzero(trunc)
            fo = obs_from_numpy(env, "cpu", final=True, rows=rows)
            assert (fo.n_tokens > 0).all() and (fo.n_ops > 0).all()
            # the final state of a 7-step episode is not a fresh run
            assert not torch.equal(fo.glob_f, obs_from_numpy(env).glob_f[torch.as_tensor(rows)])
            seen = True
            eps = env.pop_episodes()
            assert all(e[9] for e in eps if e[0] in set(rows.tolist()))
    assert seen


def test_snapshot_restarts():
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=64, play_candidates=4, snapshot_prob=1.0, snapshot_min_ante=2)
    snap = 0
    for _ in range(2500):
        heuristic(env)
        snap += sum(e[10] for e in env.pop_episodes())
    assert env.reservoir_size > 0
    assert snap > 0
    assert env.invalid_actions() == 0


def test_deterministic_policy_is_reproducible():
    m = tiny()
    g = az.Game("REPRO", "b_red", 1)
    o = obs_from_numpy(g.encode())
    a = m.act(o, deterministic=True)
    b = m.act(o, deterministic=True)
    assert a["op"].item() == b["op"].item() and a["cards"].item() == b["cards"].item()


def test_tiny_overfit_every_head():
    """The network can memorize arbitrary legal actions (op, card set,
    destination) on a small batch: gradients reach every head."""
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=48, play_candidates=4)
    m = tiny(64, 2)
    obs, acts = [], []
    for _ in range(6):
        o = obs_from_numpy(env)
        out = m.act(o)
        # The env overwrites the buffers `o` borrows from on its next step.
        obs.append(Obs(*(getattr(o, f).clone() for f in Obs.__dataclass_fields__)))
        acts.append(out)
        step(env, out)
    from antezero_agent.search import _stack

    o = _stack(obs)
    op = torch.cat([a["op"] for a in acts])
    seq = torch.cat([a["card_seq"] for a in acts])
    dest = torch.cat([a["dest"] for a in acts])
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(300):
        logp, _ent, _v = m.evaluate(o, op, seq, dest)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(logp.detach().mean()) > -0.15, float(logp.detach().mean())


def test_smoke_train_resume_and_run_protection(tmp_path):
    cfg = preset("smoke")
    cfg.run_dir = str(tmp_path)
    row = train(cfg, max_updates=1)
    assert row["update"] == 1 and row["invalid_actions"] == 0
    ck = tmp_path / "smoke" / "last.pt"
    assert read_checkpoint(ck)["update"] == 1
    # A fresh run does not overwrite an existing one ...
    with pytest.raises(FileExistsError):
        train(cfg, max_updates=1)
    # ... but it can be resumed, appending to the same log.
    row = train(cfg, resume=str(ck), max_updates=1)
    assert row["update"] == 2
    with open(tmp_path / "smoke" / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == LOG_FIELDS and [r["update"] for r in rows] == ["1", "2"]
    model, cfg2, meta = load_model(ck)
    assert meta["update"] == 2 and not model.training and cfg2.name == "smoke"


def test_compiled_encoder_keeps_checkpoint_keys():
    m = tiny()
    keys = set(m.state_dict())
    m.encoder.compile()
    assert set(m.state_dict()) == keys


def test_config_is_strict():
    with pytest.raises(ValueError):
        Config.from_dict({"ppo": {"lr_typo": 1}})
    with pytest.raises(ValueError):
        Config.from_dict({"nope": 1})
    cfg = preset("smoke")
    assert Config.from_dict(__import__("json").loads(cfg.to_json())) == cfg
    for bad in ["env.endless=treu", "ppo.epochs=1.9", "ppo.nope=1", "nope.lr=1", "ppo.lr"]:
        with pytest.raises(ValueError):
            parse_overrides(preset("smoke"), [bad])
    c = preset("smoke")
    parse_overrides(c, ["env.endless=true", "ppo.total_steps=2e3", "env.stakes=1,2", "env.backs=b_red,b_blue"])
    assert c.env.endless and c.ppo.total_steps == 2000 and c.env.stakes == [1, 2] and c.env.backs == ["b_red", "b_blue"]
    for bad in ["ppo.minibatches=3", "model.dropout=0.1", "env.play_candidates=99"]:
        c = preset("smoke")
        parse_overrides(c, [bad])
        with pytest.raises(ValueError):
            c.validate()


def test_search_returns_legal_action():
    from antezero_agent.search import choose

    cfg = preset("smoke")
    cfg.env.play_candidates = 4
    m = tiny()
    g = az.Game("SEARCH", "b_red", 1)
    g.step("select")
    code = choose(m, g, cfg=cfg, n_actions=4, n_samples=2)
    assert code == choose(m, g, cfg=cfg, n_actions=4, n_samples=2), "search is not deterministic given its seed"
    g.step(code)  # raises if illegal (sort/move codes are legal but not enumerated)


def test_resample_hidden_keeps_visible_state():
    g = az.Game("HIDDEN", "b_red", 1)
    g.step("select")
    before = g.encode(4)
    c = g.clone()
    c.resample_hidden("OTHER")
    after = c.encode(4)
    for k in ("tok_i", "tok_f", "glob_i", "glob_f", "ops", "ops_f"):
        assert (before[k] == after[k]).all(), k


def test_challenge_deck_is_rejected():
    with pytest.raises(ValueError):
        az.VecEnv(2, backs=["b_challenge"])
    with pytest.raises(ValueError):
        az.Game("X", "b_challenge")
    assert "b_challenge" not in az.backs()


def test_wilson():
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.9 and 0.97 < hi < 0.99
    assert wilson(0, 0) == (0.0, 1.0)
