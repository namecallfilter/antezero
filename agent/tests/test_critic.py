import csv
import math

import pytest
import torch

from antezero_agent import train as train_mod
from antezero_agent.config import ModelConfig, preset
from antezero_agent.critic import N_BINS, V_MAX, V_MIN, Critic, critic_state
from antezero_agent.model import obs_from_numpy
from antezero_agent.runtime import make_env, read_checkpoint
from antezero_agent.train import CRITIC_FIELDS, LOG_FIELDS, Rollout, mc_returns, train


def smoke_critic(tmp_path):
    """A smoke config with a decoupled critic read from a fresh tiny one (step limit 7: episodes end in-window)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = preset("smoke")
    cfg.run_dir, cfg.device, cfg.env.step_limit = str(tmp_path), "cpu", 7
    torch.manual_seed(1)
    path = tmp_path / "critic0.pt"
    torch.save({"model": Critic(cfg.model).state_dict()}, path)
    cfg.ppo.critic = str(path)
    return cfg


def test_critic_value():
    torch.manual_seed(0)
    c = Critic(ModelConfig(d_model=32, n_layers=1, n_heads=2))
    o = obs_from_numpy(make_env(preset("smoke"), num_envs=4, play_candidates=4))
    v = c.value(o)
    assert v.shape == (4,) and ((v > V_MIN) & (v < V_MAX)).all() and c.v_head[-1].out_features == N_BINS
    # Explicit token slots give the same values as the default (the longest list).
    assert torch.allclose(c.value(o, int(o.n_tokens.max())), v)


def test_critic_off_adds_nothing(tmp_path):
    cfg = preset("smoke")
    cfg.run_dir, cfg.device = str(tmp_path), "cpu"
    train(cfg, max_updates=1)
    assert "critic" not in read_checkpoint(tmp_path / "smoke" / "last.pt")
    with open(tmp_path / "smoke" / "log.csv", newline="") as f:
        assert next(csv.reader(f)) == LOG_FIELDS


def test_critic_is_frozen_checkpointed_and_resumed(tmp_path):
    cfg = smoke_critic(tmp_path)
    warm = critic_state(cfg.ppo.critic)
    row = train(cfg, max_updates=1)
    assert all(math.isfinite(row[k]) for k in CRITIC_FIELDS)
    ck_path = tmp_path / "smoke" / "last.pt"
    ck = read_checkpoint(ck_path)
    assert all(torch.equal(ck["critic"][k], warm[k]) for k in warm)  # never trained
    assert critic_state(ck_path).keys() == warm.keys()  # a training checkpoint can start another run
    # A resumed run reads its own critic (the start path is not read again).
    cfg.ppo.critic = str(tmp_path / "missing.pt")
    row = train(cfg, resume=str(ck_path), max_updates=1)
    assert row["update"] == 2
    with open(tmp_path / "smoke" / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == LOG_FIELDS + CRITIC_FIELDS and len(rows) == 2


def test_critic_values_drive_the_advantages(tmp_path, monkeypatch):
    """GAE runs over the critic's values (rollout states, the last observation, truncations), not the head's."""
    real_gae = train_mod.gae

    def run(c, name):
        seen = {}

        def spy(r, last_value, gamma, lam, value=None):
            if lam != 1.0:  # the advantages (mc_returns' pass has lambda 1)
                seen.update(last=last_value.clone(), value=value.clone(), head=r.value.clone(), reward=r.reward.clone(), done=r.done.clone())
            return real_gae(r, last_value, gamma, lam, value)

        monkeypatch.setattr(train_mod, "gae", spy)
        monkeypatch.setattr(Critic, "value", lambda self, o, T=None: torch.full((o.batch,), c))
        train(smoke_critic(tmp_path / name), max_updates=1)
        return seen

    a, b = run(0.125, "a"), run(0.5, "b")
    assert torch.equal(a["value"], torch.full_like(a["head"], 0.125)) and torch.equal(a["last"], torch.full_like(a["last"], 0.125))
    assert not torch.equal(a["head"], a["value"])  # the policy's own values stay in the rollout (its value loss)
    # The same games (the critic never acts): rewards differ only at truncations (step limit 7), by gamma * the change.
    diff, done = b["reward"] - a["reward"], a["done"] > 0
    assert torch.equal(a["done"], b["done"]) and not diff[~done].any()
    assert torch.isclose(diff[done], torch.tensor(preset("smoke").ppo.gamma * 0.375)).any()


def test_mc_returns_are_value_free():
    r = Rollout(4, 2, torch.device("cpu"))
    r.reward = torch.tensor([[1.0, 2.0], [0.5, 0.0], [0.25, 1.0], [3.0, 4.0]])
    r.done = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    r.value.normal_()
    ends, ret = mc_returns(r, 0.5)
    # Env 0 ends at step 1 (steps 0-1), env 1 at step 2 (steps 0-2); the steps after an end have no end in-window.
    assert ends.tolist() == [[True, True], [True, True], [False, True], [False, False]]
    assert torch.allclose(ret, torch.tensor([1.25, 2.25, 0.5, 0.5, 1.0]))  # row-major: (t0 e0, t0 e1, t1 e0, t1 e1, t2 e1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ppo.fast_rollout needs CUDA")
def test_critic_with_fast_rollout_is_the_plain_loop(tmp_path):
    """fast_rollout 1 (CUDA graphs; truncations bootstrap through the critic) trains the same policy, bitwise, also
    with the observations in host memory (ppo.rollout_obs_on_cpu: the critic's pass gathers them)."""
    out = []
    for fast, host in ((0, False), (1, False), (0, True), (1, True)):
        cfg = smoke_critic(tmp_path / f"f{fast}{host:d}")
        cfg.device, cfg.ppo.fast_rollout, cfg.ppo.rollout_obs_on_cpu = "cuda", fast, host
        train(cfg, max_updates=2)
        out.append(read_checkpoint(tmp_path / f"f{fast}{host:d}" / "smoke" / "last.pt"))
    for o in out[1:]:
        for k in ("model", "critic"):
            assert all(torch.equal(out[0][k][n], o[k][n]) for n in out[0][k]), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ppo.fast_rollout needs CUDA")
def test_critic_inside_the_collector_trains_like_its_own_pass(tmp_path, monkeypatch):
    """fast_rollout 2 has the critic value each state as it is acted on (no pass after the rollout): the same games,
    values equal up to float16 rounding of other batches, so the update barely differs from one made with the pass."""
    out = []
    for inside in (True, False):
        cfg = smoke_critic(tmp_path / f"c{inside:d}")
        cfg.device, cfg.ppo.fast_rollout = "cuda", 2
        if not inside:  # the collector without the critic: train() then runs the pass
            real = train_mod.Collector
            monkeypatch.setattr(train_mod, "Collector", lambda *a: real(*a[:6]))
        train(cfg, max_updates=1)
        out.append(read_checkpoint(tmp_path / f"c{inside:d}" / "smoke" / "last.pt")["model"])
        rows = list(csv.DictReader(open(tmp_path / f"c{inside:d}" / "smoke" / "log.csv")))
        assert math.isfinite(float(rows[-1]["critic_ev"]))
    assert max(float((out[0][k] - out[1][k]).abs().max()) for k in out[0]) < 1e-4
