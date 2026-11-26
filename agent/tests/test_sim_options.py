"""The opt-in simulator features: the obs_v2 globals, the rearrange limit, the
blinds reward, the start pool, the interval stats, and the native port of
RLBalatro's environment."""

import json

import numpy as np
import pytest

import antezero_sim as az

ZERO_CARDS = np.zeros(1, dtype=np.uint32)
ZERO_DEST = np.zeros(1, dtype=np.int32)
SKIP_BLIND, SELECT_BLIND, PLAY, MOVE_JOKER, PLAY_TOP = 1, 0, 3, 17, 21  # OpKind


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


def op_of_kind(env, kind):
    return next(i for i in range(int(env.n_ops[0])) if env.ops[0, i, 0] == kind)


def step_kind(env, kind):
    op = op_of_kind(env, kind)
    cards = np.array([env.ops[0, op, 5]], dtype=np.uint32)
    rew, done, _ = env.step(np.array([op], dtype=np.int32), cards, ZERO_DEST)
    return float(rew[0]), bool(done[0])


def play_until_paid(env):
    """Top plays until one pays or ends the episode: its reward and `done`."""
    for _ in range(12):
        rew, done = step_kind(env, PLAY_TOP)
        if rew or done:
            return rew, done
    raise AssertionError("no hand paid")


def test_the_blinds_reward_pays_a_skip_with_the_next_blind_beaten():
    env = az.VecEnv(1, seed=0, reward_mode="blinds", play_candidates=1)
    assert step_kind(env, SKIP_BLIND) == (0.0, False)
    assert step_kind(env, SELECT_BLIND) == (0.0, False)
    assert play_until_paid(env) == (2.0, False)
    assert env.game(0).phase == "round_eval"


def test_the_blinds_reward_pays_a_lost_blind_by_the_share_scored():
    env = az.VecEnv(1, seed=1, reward_mode="blinds")
    step_kind(env, SKIP_BLIND)
    step_kind(env, SELECT_BLIND)
    total = 0.0
    for _ in range(30):
        op = next((i for i in range(int(env.n_ops[0])) if env.ops[0, i, 0] == PLAY), 0)
        rew, done, _ = env.step(np.array([op], dtype=np.int32), np.array([1], dtype=np.uint32), ZERO_DEST)
        total += float(rew[0])
        if done[0]:
            break
    (episode,) = env.pop_episodes()
    assert not episode[5]  # won
    assert 0.0 < total < 1.0


def two_jokers_in_a_round():
    g = az.Game("ORDERPROBE")
    g.inject("inject_joker 60 0 0")
    g.inject("inject_joker 0 0 0")
    g.step("select")
    return g


@pytest.mark.parametrize(("max_rearranges", "labelled"), [(0, False), (2, True)])
def test_a_play_gets_a_joker_order_label_only_while_joker_moves_are_offered(max_rearranges, labelled):
    env = az.VecEnv(1, order_label_rate=1.0, max_rearranges=max_rearranges)
    env.set_game(0, two_jokers_in_a_round())
    kinds = np.asarray(env.ops[0, : env.n_ops[0], 0])
    assert bool(np.any(kinds == MOVE_JOKER)) == labelled
    env.step(np.array([op_of_kind(env, PLAY)], dtype=np.int32), np.array([1], dtype=np.uint32), ZERO_DEST)
    assert (env.aux_kind[0] >= 0) == labelled


def test_interval_stats_count_a_truncated_episode_as_a_run_but_not_a_loss():
    env = az.VecEnv(1, step_limit=1)
    env.set_game(0, two_jokers_in_a_round())
    _, done, truncated = env.step(np.zeros(1, dtype=np.int32), ZERO_CARDS, ZERO_DEST)
    assert done[0] and truncated[0]
    stats = env.pop_interval_stats()
    assert set(stats) == {"hands", "jokers", "consumables", "best"}
    assert len(stats["hands"]) == 12
    assert stats["jokers"]["j_joker"][:3] == (1, 0, 0)  # runs, wins, losses
    assert env.pop_interval_stats()["jokers"] == {}


def write_pool(tmp_path, recording, ks):
    (tmp_path / "recordings").mkdir()
    (tmp_path / "recordings" / "one.bal").write_text(recording)
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps({"rec": "one.bal", "k": k}) for k in ks))
    return str(tmp_path)


def test_episodes_start_from_the_start_pool(tmp_path):
    pool = write_pool(tmp_path, "seed TEST deck b_yellow stake 1\nselect\n", [1])
    env = az.VecEnv(2, start_pool=pool, start_frac=1.0)
    assert env.start_pool_size == 1
    assert all(env.game(i).phase == "round" for i in range(2))
    assert env.game(0).seed != env.game(1).seed


def test_a_start_share_without_usable_start_states_is_refused(tmp_path):
    with pytest.raises(ValueError, match="needs a start_pool"):
        az.VecEnv(1, start_frac=0.5)
    pool = write_pool(tmp_path, "seed TEST deck b_nope stake 1\nselect\n", [1])
    with pytest.raises(ValueError, match="no usable start state"):
        az.VecEnv(1, start_pool=pool, start_frac=0.5)


def test_a_bad_reward_mode_or_start_share_is_refused():
    with pytest.raises(ValueError, match="reward_mode"):
        az.VecEnv(1, reward_mode="nope")
    with pytest.raises(ValueError, match="start_frac"):
        az.VecEnv(1, start_frac=1.5)
    with pytest.raises(ValueError, match="must be <= 1"):
        az.VecEnv(1, snapshot_prob=0.6, start_frac=0.6)


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
