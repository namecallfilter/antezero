"""The native port of RLBalatro's environment: BalatroCoreEnv, BalatroCoreVecEnv and their start points."""

import numpy as np
import pytest

import antezero_sim as az

PLAY, DISCARD, SELECT_BLIND, PLAY_HAND = 0, 1, 2, 141  # action ids
PAIR = az.balatro_hands().index("Pair")
GROUPS, OBS_SIZE = az.balatro_obs_layout()


def core_env_arrays():
    """The arrays a BalatroCoreEnv writes into: observation, selection, mask, selection specs, hand context."""
    return (
        np.zeros(OBS_SIZE, dtype=np.float32),
        np.zeros(44, dtype=np.float32),
        np.zeros(152, dtype=np.uint8),
        np.zeros((32, 3), dtype=np.int32),
        np.zeros(32, dtype=np.int32),
    )


def vec_env_arrays(n):
    return tuple(np.zeros((n, *a.shape), dtype=a.dtype) for a in core_env_arrays())


def hand_cards(obs):
    """The (rank, suit) of each hand card the observation shows."""
    off, rows, cols = next((off, rows, cols) for name, off, rows, cols in GROUPS if name == "hand")
    return [(int(row[2]), int(row[3])) for row in obs[off : off + rows * cols].reshape(rows, cols) if row[0]]


def test_a_recorded_core_env_game_replays_to_the_same_state():
    env = az.BalatroCoreEnv(record=True)
    live = core_env_arrays()
    env.reset_into(None, "RECORD", *live)
    for action, cards in [(SELECT_BLIND, []), (DISCARD, [0, 5])]:
        assert not env.step_into(action, cards, *live)["illegal"]
    script = env.script()
    assert script == "seed RECORD deck b_red stake 1\nselect\ndiscard 0 5\n"

    pool = az.StartPool()
    assert list(pool.add(script, [2])) == [0] and pool.error == ""
    replayed = core_env_arrays()
    env.set_start(pool, 0, False)
    env.reset_into(None, "OTHER", *replayed)
    assert all(np.array_equal(a, b) for a, b in zip(live, replayed))

    env.set_start(pool, 0, True)
    env.reset_into(None, "OTHER", *replayed)
    assert not env.step_into(DISCARD, [5], *replayed)["illegal"]
    reseeded = env.script()
    assert reseeded == script + "dbg_reseed OTHER\ndiscard 5\n"
    assert list(pool.add(reseeded, [4])) == [1] and pool.error == ""


def test_the_balatro_core_env_runs_a_seeded_game():
    assert GROUPS and OBS_SIZE > 0
    assert len(az.balatro_hands()) == 12
    assert az.balatro_joker_keys() and az.balatro_voucher_keys()

    def run(seed):
        env = az.BalatroCoreEnv(deck="b_red", stake=1, max_steps=300)
        obs, sel, mask, specs, hctx = core_env_arrays()
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


def test_a_declared_pair_is_played_with_the_cards_picked():
    env = az.BalatroCoreEnv()
    obs, _, mask, specs, _ = arrays = core_env_arrays()

    def first_hand(seed):
        env.reset_into(seed, None, *arrays)
        env.step_into(SELECT_BLIND, [], *arrays)
        cards = hand_cards(obs)
        return cards, [(i, j) for i in range(len(cards)) for j in range(i) if cards[i][0] == cards[j][0]]

    cards, pairs = next(hand for hand in map(first_hand, range(20)) if len(hand[1]) == 1)
    (i, j) = pairs[0]
    assert mask[PLAY_HAND + PAIR]
    pair_verb = 21 + PAIR  # after PLAY, DISCARD, ten uses, eight pack picks and the joker reorder
    assert tuple(specs[pair_verb][:2]) == (1, 5) and specs[pair_verb][2] == (1 << len(cards)) - 1

    half = env.step_into(PLAY_HAND + PAIR, [i], *arrays)
    assert half["illegal"] and half["hand_played"] == -1
    assert hand_cards(obs) == cards

    info = env.step_into(PLAY_HAND + PAIR, [i, j], *arrays)
    assert not info["illegal"]
    assert (info["hs_declared"], info["hand_played"]) == (PAIR, PAIR)
    assert (info["played_cards"], info["played_scoring"]) == (2, 2)
    assert info["hand_score"] > 0
    left = hand_cards(obs)
    assert cards[i] not in left and cards[j] not in left
    assert all(card in left for k, card in enumerate(cards) if k not in (i, j))


@pytest.mark.parametrize("output", ["strided", "short"])
def test_a_step_refused_for_an_output_array_is_not_taken(output):
    env = az.BalatroCoreEnv(record=True)
    arrays = core_env_arrays()
    env.reset_into(None, "REFUSED", *arrays)
    before = [a.copy() for a in arrays]
    obs = np.zeros(2 * OBS_SIZE, dtype=np.float32)[::2] if output == "strided" else np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        env.step_into(SELECT_BLIND, [], obs, *arrays[1:])
    with pytest.raises(ValueError):
        env.reset_into(None, "OTHER", obs, *arrays[1:])
    assert env.script() == "seed REFUSED deck b_red stake 1\n"
    assert all(np.array_equal(a, b) for a, b in zip(arrays, before))

    info = env.step_into(SELECT_BLIND, [], *arrays)
    assert not info["illegal"] and info["steps"] == 1
    assert env.script() == "seed REFUSED deck b_red stake 1\nselect\n"


@pytest.mark.parametrize("seed", ["S" * 33, "", "A B", "A#1"])
def test_a_seed_a_script_header_cannot_hold_is_refused(seed):
    env = az.BalatroCoreEnv()
    with pytest.raises(ValueError, match="a seed is 1 to 32 bytes"):
        env.reset_into(None, seed, *core_env_arrays())


def test_a_batch_call_with_an_array_of_the_wrong_shape_changes_no_env():
    n = 2
    env = az.BalatroCoreVecEnv(n, threads=2)
    arrays = vec_env_arrays(n)
    env.reset_all(np.arange(n, dtype=np.uint64), *arrays)
    before = [a.copy() for a in arrays]
    actions = np.full(n, SELECT_BLIND, dtype=np.int32)
    picks = np.full((n, 5), -1, dtype=np.int32)
    seeds = np.arange(n, dtype=np.uint64) + 100
    for bad in [
        (actions[:1], picks, seeds),
        (actions, picks[:1], seeds),
        (actions, picks[:, :3], seeds),
        (actions, picks, seeds[:1]),
    ]:
        with pytest.raises(ValueError):
            env.step_all(*bad, *arrays)
    with pytest.raises(ValueError):
        env.step_all(actions, picks, seeds, arrays[0][:1], *arrays[1:])
    with pytest.raises(ValueError):
        env.reset_all(seeds[:1], *arrays)
    with pytest.raises(ValueError):
        env.reset_all(seeds, *arrays[:4], arrays[4][:1])
    assert all(np.array_equal(a, b) for a, b in zip(arrays, before))

    infos = env.step_all(actions, picks, seeds, *arrays)
    assert infos["steps"].tolist() == [1] * n and not infos["illegal"].any()
    assert not np.array_equal(arrays[2], before[2])


def test_a_refused_set_deck_changes_neither_the_deck_nor_the_stake():
    env = az.BalatroCoreEnv(record=True)
    arrays = core_env_arrays()
    for deck, stake in [("b_blue", 0), ("b_nope", 2)]:
        with pytest.raises(ValueError):
            env.set_deck(deck, stake)
    env.reset_into(None, "DECK", *arrays)
    assert env.script() == "seed DECK deck b_red stake 1\n"
    env.set_deck("b_blue", 2)
    env.reset_into(None, "DECK", *arrays)
    assert env.script() == "seed DECK deck b_blue stake 2\n"


def test_a_start_id_the_pool_does_not_hold_is_refused():
    pool = az.StartPool()
    assert list(pool.add("seed TEST deck b_red stake 1\nselect\n", [1])) == [0]
    env = az.BalatroCoreEnv()
    arrays = core_env_arrays()

    def started_from():
        env.reset_into(1, None, *arrays)
        return env.episode_start()

    env.set_start(pool, 0, False)
    for bad in (1, 5, -2):
        with pytest.raises(ValueError, match="not in the pool"):
            env.set_start(pool, bad, True)
    assert started_from() == 0
    for no_start in [(pool, -1), (None, -1), (None, 0)]:
        env.set_start(pool, 0, False)
        env.set_start(*no_start, True)
        assert started_from() == -1

    vec = az.BalatroCoreVecEnv(2, threads=1)
    with pytest.raises(ValueError, match="not in the pool"):
        vec.set_start(0, pool, 1, True)
    with pytest.raises(ValueError, match="index out of range"):
        vec.set_start(2, pool, 0, True)
    with pytest.raises(ValueError, match="index out of range"):
        vec.set_deck(2, "b_red", 1)
    with pytest.raises(ValueError, match="index out of range"):
        vec.episode_start(2)
    vec.set_start(1, pool, 0, False)
    vec.reset_all(np.zeros(2, dtype=np.uint64), *vec_env_arrays(2))
    assert [vec.episode_start(i) for i in range(2)] == [-1, 0]


@pytest.mark.parametrize(
    "options",
    [
        {"reward_money": True, "money_per_reward": 0.0},
        {"reward_money": True, "money_per_reward": -15.0},
        {"reward_money": True, "money_per_reward": float("nan")},
        {"reward_money": True, "money_per_reward": float("inf")},
        {"max_steps": 0},
    ],
)
def test_options_that_break_the_rewards_or_the_episode_are_refused(options):
    with pytest.raises(ValueError):
        az.BalatroCoreEnv(**options)
    with pytest.raises(ValueError):
        az.BalatroCoreVecEnv(1, **options)


def test_a_batch_of_no_envs_is_refused():
    with pytest.raises(ValueError, match="num_envs"):
        az.BalatroCoreVecEnv(0)
