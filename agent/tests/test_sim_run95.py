"""The simulator features the run95 training uses: the obs_v2 globals, the
rearrange limit, the blinds reward, the interval stats, and the native port of
RLBalatro's environment."""

import numpy as np

import antezero_sim as az

ZERO_CARDS = np.zeros(1, dtype=np.uint32)
ZERO_DEST = np.zeros(1, dtype=np.int32)


def test_obs_v2_appends_the_last_hand_globals():
    g = az.Game("OBSV2PY")
    g.step("select")
    g.step("play 1")
    o1 = g.encode(obs_v2=False)
    o2 = g.encode(obs_v2=True)
    assert o1["glob_f"].shape == (az.GLOB_FLOAT,)
    assert o2["glob_f"].shape == (az.GLOB_FLOAT_V2,)
    assert np.array_equal(o2["glob_f"][: az.GLOB_FLOAT], o1["glob_f"])
    assert np.all(o2["glob_f"][az.GLOB_FLOAT :] > 0)

    env = az.VecEnv(2, seed=1, obs_v2=True, max_rearranges=5)
    assert env.glob_f.shape == (2, az.GLOB_FLOAT_V2)
    assert env.final_glob_f.shape == (2, az.GLOB_FLOAT_V2)
    assert env.game(0).max_rearranges == 5


def op_of_kind(env, kinds):
    return next(i for i in range(int(env.n_ops[0])) if env.ops[0, i, 0] in kinds)


def test_the_blinds_reward_pays_a_skip_with_the_next_blind_beaten():
    env = az.VecEnv(1, seed=1, reward_mode="blinds")
    skip = op_of_kind(env, (1,))  # OpKind::SkipBlind
    rew, done, _ = env.step(np.array([skip], dtype=np.int32), ZERO_CARDS, ZERO_DEST)
    assert rew[0] == 0.0 and not done[0]
    select = op_of_kind(env, (0,))  # OpKind::SelectBlind
    rew, done, _ = env.step(np.array([select], dtype=np.int32), ZERO_CARDS, ZERO_DEST)
    assert rew[0] == 0.0 and not done[0]
    total = 0.0
    for _ in range(30):
        op = next((i for i in range(int(env.n_ops[0])) if env.ops[0, i, 0] in (21, 3)), 0)  # PlayTop, Play
        cards = int(env.ops[0, op, 5]) or 1
        rew, done, _ = env.step(np.array([op], dtype=np.int32), np.array([cards], dtype=np.uint32), ZERO_DEST)
        total += float(rew[0])
        if total > 0:
            break
    assert total > 0.0


def test_a_bad_reward_mode_or_start_share_is_refused():
    import pytest

    with pytest.raises(ValueError, match="reward_mode"):
        az.VecEnv(1, reward_mode="nope")
    with pytest.raises(ValueError, match="start_frac"):
        az.VecEnv(1, start_frac=1.5)
    with pytest.raises(ValueError, match="must be <= 1"):
        az.VecEnv(1, snapshot_prob=0.6, start_frac=0.6)


def test_interval_stats_count_the_finished_episodes():
    env = az.VecEnv(4, seed=3, step_limit=200)
    assert env.start_pool_size == 0
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(400):
        op = (rng.random(4) * np.maximum(np.asarray(env.n_ops), 1)).astype(np.int32)
        env.step(op, np.zeros(4, dtype=np.uint32), np.zeros(4, dtype=np.int32))
        finished += len(env.pop_episodes())
        if finished:
            break
    assert finished
    stats = env.pop_interval_stats()
    assert isinstance(stats, dict) and stats
    assert env.pop_interval_stats() != stats  # drained


def test_the_balatro_core_env_runs_a_seeded_game():
    groups, obs_size = az.balatro_obs_layout()
    assert groups and obs_size > 0
    assert len(az.balatro_hands()) == 12
    assert az.balatro_joker_keys() and az.balatro_voucher_keys()

    def run(seed):
        env = az.BalatroCoreEnv(deck="b_red", stake=1, max_steps=300)
        obs = np.zeros(obs_size, dtype=np.float32)
        sel = np.zeros(44, dtype=np.float32)
        mask = np.zeros(152, dtype=np.uint8)
        specs = np.zeros((32, 3), dtype=np.int32)
        hctx = np.zeros(32, dtype=np.int32)
        env.reset_into(seed, None, obs, sel, mask, specs, hctx)
        rng = np.random.default_rng(seed)
        trace = []
        for _ in range(300):
            assert mask.any() and np.isfinite(obs).all()
            a = int(rng.choice(np.flatnonzero(mask)))
            info = env.step_into(a, [], obs, sel, mask, specs, hctx)  # a play without picks is illegal
            trace.append((a, bool(info["illegal"]), float(info["reward"])))
            if info["terminated"] or info["truncated"]:
                break
        return trace, obs.copy()

    t1, o1 = run(7)
    t2, o2 = run(7)
    assert t1 == t2 and np.array_equal(o1, o2)
    assert not all(illegal for _, illegal, _ in t1)
