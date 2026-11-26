import csv
import json
from pathlib import Path

import numpy as np
import torch

import antezero_sim as az
from antezero_agent.config import Config, ModelConfig, preset
from antezero_agent.interval_log import IntervalLogger
from antezero_agent.model import Net, obs_from_numpy
from antezero_agent.models import (
    ACTION_HEADS,
    ActionHead,
    FactoredActionHead,
    HandBuilderActionHead,
    Run95Net,
    build_model,
)
from antezero_agent.runtime import make_env
from antezero_agent.train import train


def test_start_pool_loads_and_resets():
    cfg = preset("run95")
    assert cfg.env.start_pool == "pools/human_gold_v2_ante6"
    assert cfg.env.start_frac == 0.2
    assert cfg.env.snapshot_prob == 0.0

    env = az.VecEnv(
        16,
        backs=["b_red"],
        stakes=[8],
        seed=1,
        start_pool="pools/human_gold_v2_ante6",
        start_frac=1.0,
    )
    assert env.start_pool_size == 1341
    # At start_frac=1.0, initial games come from the start pool (some at ante >= 1).
    # Step until at least one episode finishes and verify from_snapshot=True.
    ops = np.asarray(env.ops)
    n_ops = np.asarray(env.n_ops)
    saw_snapshot_ep = False
    for _ in range(400):
        # Pick first legal op with 0 required/max cards or play_top if available, else op 0.
        act_op = np.zeros(16, dtype=np.int32)
        act_cards = np.zeros(16, dtype=np.uint32)
        for i in range(16):
            for j in range(n_ops[i]):
                if ops[i, j, 3] == 0:
                    act_op[i] = j
                    break
                elif ops[i, j, 4] == ops[i, j, 5]:
                    act_op[i] = j
                    act_cards[i] = np.int32(ops[i, j, 5]).view(np.uint32)
                    break
        env.step(act_op, act_cards, np.zeros(16, dtype=np.int32))
        eps = env.pop_episodes()
        if any(e[10] for e in eps):
            saw_snapshot_ep = True
            break
    assert saw_snapshot_ep


def test_action_heads_registry_and_logp():
    assert ACTION_HEADS == {"factored": FactoredActionHead, "hand_builder": HandBuilderActionHead}
    assert issubclass(FactoredActionHead, ActionHead)
    assert issubclass(HandBuilderActionHead, ActionHead)

    cfg = preset("smoke")
    env = make_env(cfg, num_envs=8)
    o = obs_from_numpy(env)

    for arch in ("antezero", "run95"):
        for head_name in ("factored", "hand_builder"):
            mcfg = ModelConfig(d_model=32, n_layers=1, n_heads=2, arch=arch, action_head=head_name)
            m = build_model(mcfg).eval()
            assert isinstance(m.action_head, ACTION_HEADS[head_name])
            out = m.act(o)
            with torch.no_grad():
                logp, ent, val = m.evaluate(o, out["op"], out["card_seq"], out["dest"])
            assert torch.allclose(logp, out["logp"], atol=1e-4)
            assert torch.allclose(val, out["value"], atol=1e-5)
            assert (ent >= -1e-5).all()
            if head_name == "factored":
                sd = m.state_dict()
                assert "op_kind.weight" in sd
                assert "op_feat.weight" in sd
                assert "dst_q.0.weight" in sd


def test_interval_logging_csvs(tmp_path):
    cfg = preset("smoke")
    cfg.name = "interval_test"
    cfg.run_dir = str(tmp_path)
    cfg.log_interval_updates = 1
    cfg.save_every = 0
    cfg.eval_every = 0
    cfg.ppo.rollout_len = 16
    cfg.ppo.minibatches = 2
    cfg.ppo.total_steps = 8 * 16 * 2
    train(cfg, max_updates=2)

    run_dir = tmp_path / "interval_test"
    for fname in ("hands.csv", "jokers.csv", "consumables.csv", "combos.csv"):
        assert (run_dir / fname).exists(), f"missing {fname}"
    with open(run_dir / "hands.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24  # 12 hands * 2 updates

    # IntervalLogger.flush directly
    env = make_env(cfg, num_envs=4)
    il = IntervalLogger(tmp_path, cfg.env.backs, cfg.env.stakes)
    il.add_episodes([(0, "SEED", "b_red", 1, 2, True, 6, 50, 1.0, False, False)])
    d = il.flush(1000, 10000, env)
    assert "completion/stake1_ge1" in d
    assert d["completion/stake1_ge1"] == 1.0
    assert "hands/high_card_per_game" in d
