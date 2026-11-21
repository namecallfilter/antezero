import argparse
import csv
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import antezero_sim as az
from antezero_agent.collect import Collector
from antezero_agent.config import Config, ModelConfig, parse_overrides, preset
from antezero_agent.critic import Critic
from antezero_agent.model import NEG, OP_BUY, OP_BUY_AND_USE, OP_END_SHOP, OP_PICK, OP_REDEEM, Net, Obs, noise_shapes, obs_from_numpy, sample
from antezero_agent.runtime import load_model, make_env, read_checkpoint, seeded
from antezero_agent.search import shaped_reward
from antezero_agent.stats import wilson
from antezero_agent.train import LOG_FIELDS, Rollout, gae, train

OP = {"select": 0, "cash_out": 14, "end_shop": 13, "skip_pack": 16, "sort_rank": 18, "sort_suit": 19}


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


def test_bf16_encoder_changes_nothing_on_cpu():
    env = make_env(preset("smoke"), num_envs=16)
    a, b = tiny(), tiny()
    b.cfg.bf16_encoder = True
    o = obs_from_numpy(env)
    assert all(torch.equal(x, y) for x, y in zip(a.encode(o), b.encode(o)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bfloat16 autocast runs on CUDA")
def test_bf16_encoder_acts_and_evaluates_alike():
    """With the encoder in bfloat16, teacher forcing still reproduces the sampled log-probs (PPO's ratio starts at
    1), everything leaving the network is float32, and the policy stays that of the float32 network."""
    dev = torch.device("cuda")
    cfg = preset("smoke")
    cfg.env.num_envs = 64
    env = make_env(cfg)
    fp, bf = tiny(64, 2).to(dev), tiny(64, 2).to(dev)
    bf.cfg.bf16_encoder = True
    for _ in range(20):
        o = obs_from_numpy(env, dev)
        out = bf.act(o)
        with torch.no_grad():
            logp, _ent, val = bf.evaluate(o, out["op"], out["card_seq"], out["dest"])
            ref = fp.evaluate(o, out["op"], out["card_seq"], out["dest"])[0]
        assert logp.dtype == val.dtype == out["logp"].dtype == torch.float32
        assert torch.allclose(logp, out["logp"], atol=1e-4), (logp - out["logp"]).abs().max()
        assert not torch.equal(logp, ref) and float((logp - ref).abs().max()) < 0.2
        env.step(out["op"].to(torch.int32).cpu().numpy(), out["cards"].cpu().numpy().astype(np.uint32), out["dest"].to(torch.int32).cpu().numpy())


def test_shaping_telescopes_to_true_return_plus_progress():
    """Over a whole episode the discounted shaped rewards sum to the
    discounted true return (only the final step has one) plus
    gamma^T * Phi(s_T) = gamma^T * w_blind * blinds cleared (Phi(start) = 0;
    Phi is kept at the terminal state so lost games keep their progress), or
    to the true return alone with terminal_phi=False (Phi(terminal) = 0)."""
    for gamma, tp in ((1.0, True), (0.99, True), (0.99, False)):
        cfg = preset("smoke")
        cfg.env.w_score = 0.0  # the lost blind's chip fraction (kept at game over) is not in the episode record
        env = make_env(cfg, num_envs=32, gamma=gamma, r_loss=0.5, play_candidates=4, terminal_phi=tp)
        progress = 0
        acc, t = np.zeros(32), np.zeros(32)
        checked = 0
        for _ in range(1500):
            r, done, trunc = heuristic(env)
            acc += gamma**t * r
            t += 1
            for ep in env.pop_episodes():
                i = ep[0]
                if not ep[9]:
                    want = gamma ** (ep[7] - 1) * ep[8] + (gamma ** ep[7] * cfg.env.w_blind * ep[6] if tp else 0.0)
                    assert abs(acc[i] - want) < 1e-3, (gamma, acc[i], want, ep)
                    checked += 1
                    progress += ep[6] > 0
                acc[i], t[i] = 0.0, 0
        assert checked >= 20 and progress >= 5


def test_a_loss_costs_no_progress():
    """With gamma = 1 the last step of a lost game is never negative: the
    game-over state keeps the blinds cleared and the lost blind's chip
    fraction, so a policy gains nothing by postponing a loss (stalling)."""
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=32, gamma=1.0, play_candidates=4)
    losses = 0
    for _ in range(1500):
        r, _done, _trunc = heuristic(env)
        for ep in env.pop_episodes():
            if not ep[5] and not ep[9]:
                assert r[ep[0]] >= -1e-6, (r[ep[0]], ep)
                losses += 1
    assert losses >= 5


def test_skipping_a_blind_earns_no_progress():
    """Skipped blinds count in the game's blinds_cleared but not in the
    shaping potential (else skipping is risk-free reward), and
    allow_skip=False stops offering the skip."""
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=1, gamma=1.0)
    kinds = np.asarray(env.ops)[0, : np.asarray(env.n_ops)[0], 0]
    skip = int(np.flatnonzero(kinds == 1)[0])
    r, _done, _trunc = env.step(np.array([skip], np.int32), np.zeros(1, np.uint32), np.zeros(1, np.int32))
    assert env.game(0).blinds_cleared == 1 and abs(float(r[0])) < 1e-6
    off = make_env(cfg, num_envs=1, allow_skip=False)
    assert 1 not in np.asarray(off.ops)[0, : np.asarray(off.n_ops)[0], 0]


def test_a_standalone_game_can_hide_skips_like_training():
    """play_real builds its own Game: a policy trained with allow_skip=False
    must not be offered skips there."""
    g = az.Game("SKIPTEST")
    assert g.allow_skip and env_game_skip_offered(g)
    g.allow_skip = False
    assert not g.allow_skip and not env_game_skip_offered(g)
    assert make_env(preset("smoke"), num_envs=1, allow_skip=False).game(0).allow_skip is False


def env_game_skip_offered(g) -> bool:
    d = g.encode(8, False, True)
    return 1 in d["ops"][: int(d["n_ops"]), 0]


# A real game (bridge, Steamodded) that diverged by one chip: Pair of Queens,
# Half Joker, Banner with 4 discards, Ramen at x1.93 -> 150 x 42.46.
RAMEN_GAME = (
    "select;discard 240;play 59;cash_out;buy 1;open 0;pick 0 0;end_shop;select;play 31;cash_out;open 1;pick 0 0;end_shop;select;play 155;cash_out;"
    "end_shop;select;play 115;cash_out;buy_use 0 0;reroll;buy 1;open 0;skip_pack;end_shop;select;play 27;play 240;cash_out;reroll;buy 0;"
    "move_joker 3 0;end_shop;select;discard 241;play 236;play 87;move_joker 4 1;cash_out;sell_joker 1;buy 1;move_joker 4 0;open 1;pick 0 0;"
    "open 0;pick 0 0;end_shop;select;play 192;cash_out;buy_use 0 0;open 1;pick 0 0;reroll;reroll;sell_joker 1;buy 1;end_shop;select;"
    "play 192;cash_out;buy_use 0 0;buy_use 0 0;end_shop;select;play 131"
)


def test_steamodded_rounds_xmult_like_the_real_game():
    """Steamodded applies xMult as mult + mult * (x - 1): the floored score can
    be one chip below vanilla's mult * x (play_real turns it on)."""
    chips = {}
    for steamodded in (False, True):
        g = az.Game("VL34I4WG", "b_red", 1)
        g.steamodded = steamodded
        for code in RAMEN_GAME.split(";"):
            g.step(code)
        chips[steamodded] = g.canonical()["chips"]
        assert g.clone().steamodded is steamodded
    assert chips == {False: "6369", True: "6368"}


def test_progress_potential_is_the_run_position():
    """The blind term of Phi is the run position (3 per finished ante plus
    this ante's defeated blinds), so neither skips nor ante-reducing vouchers
    (Hieroglyph, Petroglyph) earn free progress."""
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=4, play_candidates=4)
    checked, top_ante = 0, 1
    for _ in range(400):
        heuristic(env)
        for i in range(env.num_envs):
            g = env.game(i)
            c = g.canonical()
            defeated = sum(c[f"blind_states.{s}"] == "Defeated" for s in ("Small", "Big", "Boss"))
            want = 0.04 * (3 * max(g.ante - 1, 0) + defeated)
            assert abs(g.potential(0.04, 0.0) - want) < 1e-5, (g.ante, defeated, g.potential(0.04, 0.0))
            checked += 1
            top_ante = max(top_ante, g.ante)
    assert checked > 1000 and top_ante >= 2


@pytest.mark.parametrize("tp", [True, False])
def test_search_reward_matches_vecenv(tp):
    cfg = preset("smoke")
    cfg.env.play_candidates = 4
    cfg.env.r_loss = 0.5
    cfg.env.terminal_phi = tp
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


def test_gae_stops_before_an_explored_step():
    r = Rollout(3, 1, torch.device("cpu"))
    r.reward[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    r.value[:, 0] = 0.5
    cut = torch.tensor([[False], [True], [False]])
    adv, _ = gae(r, torch.tensor([10.0]), gamma=0.9, lam=0.8, cut=cut)
    # t=2 and the explored t=1 as without the cut (11.5; 2 + 0.9*0.5 - 0.5 + 0.9*0.8*11.5); t=0 sees only the value
    # of the state it led to: 1 + 0.9*0.5 - 0.5.
    assert torch.allclose(adv[:, 0], torch.tensor([0.95, 10.23, 11.5]))
    assert torch.equal(gae(r, torch.tensor([10.0]), gamma=0.9, lam=0.8, cut=torch.zeros_like(cut))[0], gae(r, torch.tensor([10.0]), gamma=0.9, lam=0.8)[0])


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


def gift_of(g):
    """What a run of the heuristic holds, which never buys: (jokers, top hand level, the first joker is eternal)."""
    c = g.canonical()
    held = int(c["jokers.len"])
    top = max(int(v) for k, v in c.items() if k.startswith("hands.") and k.endswith(".level"))
    return held, top, held > 0 and c["jokers.0.ab.eternal"] == "true"


def test_null_sorts_can_be_left_out():
    """A dealt hand is in rank order: with env.null_sorts off the rank sort is not offered, in the env and in its games."""
    cfg = preset("smoke")
    for on in (True, False):
        env = make_env(cfg, num_envs=1, null_sorts=on)
        heuristic(env)
        g = env.game(0)
        o = g.encode()
        assert g.phase == "round" and g.null_sorts is on
        for kinds in (o["ops"][: int(o["n_ops"]), 0].tolist(), np.asarray(env.ops)[0, : int(np.asarray(env.n_ops)[0]), 0].tolist()):
            assert (OP["sort_rank"] in kinds) is on and OP["sort_suit"] in kinds


def test_practice_restarts_hold_one_gift_and_feed_no_pool():
    cfg = preset("smoke")
    env = make_env(cfg, num_envs=64, play_candidates=4, snapshot_prob=0.3, snapshot_min_ante=4, practice_prob=0.4)
    restarted, jokers, levels = set(), 0, 0
    for _ in range(6000):
        heuristic(env)
        for e in env.pop_episodes():
            if e[0] in restarted:
                assert e[10], "a restarted episode is flagged as one"
            g = env.game(e[0])
            restarted.discard(e[0])
            if g.ante == 1:
                continue
            restarted.add(e[0])
            # Only runs with a gift get past ante 3 here, and they feed neither pool: no late restart, no second gift.
            assert 2 <= g.ante <= 3 and g.phase == "blind_select"
            held, top, eternal = gift_of(g)
            # One sellable joker or three levels of a hand type.
            assert (held, top) in ((1, 1), (0, 4)) and not eternal
            jokers += held
            levels += top == 4
    assert jokers > 0 and levels > 0
    assert env.invalid_actions() == 0


def test_practice_eternal_gifts_are_eternal_where_the_game_allows():
    env = make_env(preset("smoke"), num_envs=64, play_candidates=4, practice_prob=0.5, practice_eternal=True)
    eternal, plain = set(), set()
    for _ in range(4000):
        heuristic(env)
        for e in env.pop_episodes():
            g = env.game(e[0])
            if g.ante > 1 and gift_of(g)[0]:
                (eternal if gift_of(g)[2] else plain).add(g.canonical()["jokers.0.key"])
    assert len(eternal) > 20
    # The jokers the game never makes Eternal.
    assert plain <= {"j_gros_michel", "j_ice_cream", "j_cavendish", "j_turtle_bean", "j_ramen", "j_diet_cola", "j_popcorn", "j_selzer", "j_mr_bones", "j_invisible", "j_luchador"}
    assert env.invalid_actions() == 0


def test_restart_shares_are_checked():
    cfg = preset("smoke")
    for bad in (dict(snapshot_prob=0.1, snapshot_min_ante=3, practice_prob=0.1), dict(snapshot_prob=0.8, snapshot_min_ante=5, practice_prob=0.25)):
        with pytest.raises(ValueError):
            make_env(cfg, num_envs=2, **bad)
        for k, v in bad.items():
            setattr(cfg.env, k, v)
        with pytest.raises(ValueError):
            cfg.validate()
    make_env(preset("smoke"), num_envs=2, snapshot_prob=0.8, snapshot_min_ante=5, practice_prob=0.2)
    make_env(preset("smoke"), num_envs=2, practice_prob=1.0)


def test_explore_redeem_replaces_sampled_shop_exits():
    cfg = preset("smoke")
    cfg.env.num_envs = 64
    env = make_env(cfg)
    m = tiny()
    explored = 0
    for _ in range(600):
        o = obs_from_numpy(env)
        kinds = o.ops[..., 0]
        legal = torch.arange(kinds.shape[1])[None] < o.n_ops[:, None]
        red = ((kinds == OP_REDEEM) & legal).any(-1)
        out = m.act(o, explore=(1.0, 0.0))
        chosen = kinds.gather(1, out["op"][:, None]).squeeze(1)
        assert not (red & (chosen == OP_END_SHOP)).any()
        x = out["explored"]
        assert (red | ~x).all() and (chosen[x] == OP_REDEEM).all() and (out["cards"][x] == 0).all()
        logp, _, _ = m.evaluate(o, out["op"], out["card_seq"], out["dest"])
        assert torch.allclose(logp, out["logp"], atol=1e-4)  # the policy's own log-prob of the explored action
        assert torch.equal(m.act(o, deterministic=True, explore=(1.0, 1.0))["op"], m.act(o, deterministic=True)["op"])
        for i in np.flatnonzero(x.numpy()):  # the kind constants match the simulator's
            assert env.game(int(i)).factored_to_code(int(out["op"][i]), 0, 0, cfg.env.play_candidates).startswith("redeem")
        explored += int(x.sum())
        heuristic(env)  # leaves every shop, so money (and affordable vouchers) accumulate
    assert explored > 0


def test_explore_buy_tries_purchases_and_pack_cards():
    """A sampled shop exit buys a legal shop card instead, a sampled pack pick or skip takes a card of the pack; the
    replaced op is always legal and its log-probability the policy's own."""
    cfg = preset("smoke")
    cfg.env.num_envs = 64
    env = make_env(cfg)
    m = tiny()
    seen = {"buy": 0, "pick": 0}
    for _ in range(600):
        o = obs_from_numpy(env)
        kinds = o.ops[..., 0]
        legal = torch.arange(kinds.shape[1])[None] < o.n_ops[:, None]
        known = m.tryable[o.tok_i[..., 1].gather(1, o.ops[..., 1].clamp(min=0))]
        buys = (((kinds == OP_BUY) | (kinds == OP_BUY_AND_USE)) & legal & known).any(-1)
        out = m.act(o, explore=(0.0, 1.0))
        chosen, x = kinds.gather(1, out["op"][:, None]).squeeze(1), out["explored"]
        assert not (buys & (chosen == OP_END_SHOP)).any()  # every sampled exit with one of them to buy buys it
        assert legal.gather(1, out["op"][:, None]).all() and known.gather(1, out["op"][:, None]).squeeze(1)[x].all()
        assert ((chosen[x] == OP_BUY) | (chosen[x] == OP_BUY_AND_USE) | (chosen[x] == OP_PICK)).all()
        logp, _, _ = m.evaluate(o, out["op"], out["card_seq"], out["dest"])
        assert torch.allclose(logp, out["logp"], atol=1e-4)
        for i in np.flatnonzero(x.numpy()):  # the kind constants match the simulator's
            code = env.game(int(i)).factored_to_code(int(out["op"][i]), int(out["cards"][i]), int(out["dest"][i]), cfg.env.play_candidates)
            seen[code.split()[0].split("_")[0]] += 1
        step(env, out)
    assert seen["buy"] > 0 and seen["pick"] > 0, seen


@pytest.mark.parametrize("all_explored", [False, True])
def test_explore_redeem_trains(tmp_path, monkeypatch, all_explored):
    """The loss branch of explored samples and the KL without them run. A smoke policy never reaches a
    shop, so half the rows (or all: a chunk of explored rows only has no KL) are marked explored: finite, and
    not the loss of the same run unmarked."""
    act = Net.act
    losses = []
    for mark in (False, True):

        def marked(self, o, deterministic=False, explore=(0.0, 0.0), mark=mark):
            out = act(self, o, deterministic, explore)
            out["explored"] = ((torch.arange(o.batch, device=o.ops.device) % 2 == 0) | all_explored) & mark
            return out

        monkeypatch.setattr(Net, "act", marked)
        cfg = preset("smoke")
        cfg.run_dir, cfg.name = str(tmp_path), f"marked{int(mark)}"
        cfg.ppo.explore_redeem, cfg.ppo.learn_frac = 1.0, 0.5
        row = train(cfg, max_updates=2)
        assert np.isfinite(row["policy_loss"]) and np.isfinite(row["approx_kl"])
        losses.append(row["policy_loss"])
    assert losses[0] != losses[1]


def test_pre_drawn_noise_replays_sampling():
    """``sample`` is ``Categorical.sample`` without its host syncs, and ``act`` with its draws
    made beforehand in ``noise_shapes`` order (what ``ppo.fast_rollout``'s CUDA graphs
    replay) samples exactly what it draws itself, also when exploring."""
    lg = (torch.randn(64, 33) * 30).masked_fill(torch.rand(64, 33) < 0.5, NEG)
    lg[:, 0] = 0.0
    with seeded(2):
        x = torch.distributions.Categorical(logits=lg, validate_args=False).sample()
    with seeded(2):
        assert torch.equal(sample(lg), x)
    m, env = tiny(), make_env(preset("smoke"), num_envs=16, play_candidates=4)
    for _ in range(10):
        o = obs_from_numpy(env)
        for explore in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
            with seeded(1):
                a = m.act(o, explore=explore)
            with seeded(1):
                qs = [torch.empty(s).uniform_() if u else torch.empty(s).exponential_(1) for s, u in noise_shapes(o.batch, *explore)]
                b = m.act(o, explore=explore, noise=qs)
            assert all(torch.equal(a[k], b[k]) for k in a)
        step(env, a)


def test_deterministic_policy_is_reproducible():
    m = tiny()
    g = az.Game("REPRO", "b_red", 1)
    o = obs_from_numpy(g.encode())
    a = m.act(o, deterministic=True)
    b = m.act(o, deterministic=True)
    assert a["op"].item() == b["op"].item() and a["cards"].item() == b["cards"].item()


def test_tiny_overfit_every_head():
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


def test_training_logs_outcomes_per_deck(tmp_path):
    cfg = preset("smoke")
    cfg.run_dir, cfg.env.backs, cfg.ppo.rollout_len, cfg.ppo.total_steps = str(tmp_path), ["b_red", "b_blue"], 128, 8 * 128
    row = train(cfg, max_updates=1)
    cells = json.loads((tmp_path / "smoke" / "cells.jsonl").read_text())
    assert cells.pop("update") == 1 and set(cells) <= {"b_red/1", "b_blue/1"}
    assert sum(n for _, n in cells.values()) == row["episodes"] > 0


def test_smoke_train_resume_and_run_protection(tmp_path):
    cfg = preset("smoke")
    cfg.run_dir = str(tmp_path)
    row = train(cfg, max_updates=1)
    assert row["update"] == 1 and row["invalid_actions"] == 0
    ck = tmp_path / "smoke" / "last.pt"
    assert read_checkpoint(ck)["update"] == 1
    with pytest.raises(FileExistsError):
        train(cfg, max_updates=1)
    row = train(cfg, resume=str(ck), max_updates=1)
    assert row["update"] == 2
    with open(tmp_path / "smoke" / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == LOG_FIELDS and [r["update"] for r in rows] == ["1", "2"]
    model, cfg2, meta = load_model(ck)
    assert meta["update"] == 2 and not model.training and cfg2.name == "smoke"
    # Other env.num_envs / ppo.rollout_len with the same product resume; another product does not.
    cfg.env.num_envs, cfg.ppo.rollout_len, cfg.ppo.total_steps = 4, 16, 4 * 64
    assert train(cfg, resume=str(ck), max_updates=1)["update"] == 3
    cfg.env.num_envs = 2
    with pytest.raises(ValueError):
        train(cfg, resume=str(ck), max_updates=1)
    # A checkpoint keeps its model, and with it the effect table it reads.
    cfg.env.num_envs, cfg.model.effects = 4, False
    with pytest.raises(ValueError):
        train(cfg, resume=str(ck), max_updates=1)


def test_a_stop_file_ends_training_after_the_update(tmp_path):
    cfg = preset("smoke")
    cfg.run_dir, cfg.ppo.total_steps = str(tmp_path), 8 * 8 * 5
    train(cfg, max_updates=1)
    stop, ck = tmp_path / "smoke" / "STOP", tmp_path / "smoke" / "last.pt"
    stop.touch()
    assert train(cfg, resume=str(ck))["update"] == 2
    assert not stop.exists() and read_checkpoint(ck)["update"] == 2


def test_grad_accum_takes_the_same_steps(tmp_path):
    weights = []
    for accum in (1, 2, 4):  # 2, 4: chunks sorted by token count, each at its own lengths
        cfg = preset("smoke")
        cfg.run_dir, cfg.name, cfg.device = str(tmp_path), f"accum{accum}", "cpu"
        cfg.ppo.amp_cpu, cfg.ppo.grad_accum = False, accum
        train(cfg, max_updates=1)
        weights.append(read_checkpoint(tmp_path / cfg.name / "last.pt")["model"])
    # One update moves weights by ~lr (3e-4); chunking may only change rounding.
    assert max(float((weights[0][k] - w[k]).abs().max()) for w in weights[1:] for k in weights[0]) < 1e-5


def test_learn_frac_skips_the_surest_samples(tmp_path):
    """A forced action (one legal choice, log-prob 0) has no policy gradient, so skipping those leaves the step as it was."""
    weights = []
    for frac in (1.0, 0.875):  # 0.875 skips 4 of a minibatch's 32 samples; about a quarter of them are forced
        cfg = preset("smoke")
        cfg.run_dir, cfg.name, cfg.device = str(tmp_path), f"frac{frac}", "cpu"
        cfg.env.allow_skip, cfg.ppo.amp_cpu, cfg.ppo.learn_frac = False, False, frac
        cfg.ppo.vf_coef = cfg.ppo.ent_coef = cfg.ppo.ent_coef_final = 0.0  # the policy loss alone
        train(cfg, max_updates=1)
        weights.append(read_checkpoint(tmp_path / cfg.name / "last.pt")["model"])
    assert max(float((weights[0][k] - weights[1][k]).abs().max()) for k in weights[0]) < 1e-5


@pytest.mark.parametrize("device,fast", [("cpu", 0)] + ([("cuda", 0), ("cuda", 1), ("cuda", 2)] if torch.cuda.is_available() else []))
def test_rollout_obs_on_cpu_takes_the_same_steps(tmp_path, device, fast):
    """Tokens and ops in host memory: the same losses and weights, bitwise (the plain loop or the Collector, also
    with two halves, stores them; the recompute pass and the sorted grad_accum chunks gather them). A no-op on the CPU."""
    out = []
    for on in (False, True):
        cfg = preset("smoke")
        cfg.run_dir, cfg.name, cfg.device = str(tmp_path), f"host{int(on)}", device
        cfg.ppo.amp_cpu, cfg.ppo.grad_accum, cfg.ppo.fast_rollout, cfg.ppo.rollout_obs_on_cpu = False, 2, fast, on
        row = train(cfg, max_updates=2)
        out.append((row, read_checkpoint(tmp_path / cfg.name / "last.pt")["model"]))
    (r0, w0), (r1, w1) = out
    assert all(torch.equal(w0[k], w1[k]) for k in w0)
    assert all(r0[k] == r1[k] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "logp_drift"))


def test_act_reports_how_sure_the_state_is():
    """``sure`` (what ``ppo.learn_frac`` keeps samples by) depends on the state alone, is log 1 with one legal op,
    bounds the log-prob of every sampled action without a card or destination choice, and never counts an op that
    still has such a choice: a state led by one (which cards to discard) must not look settled."""
    m = tiny()
    env = make_env(preset("smoke"), seed=3, num_envs=64)
    torch.manual_seed(0)
    led = 0
    for _ in range(12):
        o = obs_from_numpy(env)
        out, lsm = m.act(o), m.encode(o)[3].log_softmax(-1)
        assert torch.equal(out["sure"], m.act(o)["sure"]) and torch.equal(out["sure"], m.act(o, deterministic=True)["sure"])
        assert bool((out["sure"][o.n_ops == 1] == 0).all()) and bool((out["sure"] <= 0).all())
        row = o.ops[torch.arange(64), out["op"]]
        whole = ((row[:, 3] == 0) | (row[:, 4] == row[:, 5])) & (row[:, 6] == 0)
        assert bool((out["sure"][whole] >= out["logp"][whole] - 1e-5).all())
        top = o.ops[torch.arange(64), lsm.argmax(-1)]
        choice = ((top[:, 3] > 0) & (top[:, 4] != top[:, 5])) | (top[:, 6] > 0)
        assert bool((out["sure"][choice] < lsm.amax(-1)[choice]).all())
        led += int(choice.sum())
        step(env, out)
    assert led > 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Collector needs CUDA")
def test_collector_fills_host_storage_the_same():
    """``ppo.fast_rollout`` writes the same rollout into host storage, and gathering it gives the same observations
    (trimmed to the rows' longest token and op lists: the same log-probs, entropies and values)."""
    cfg, dev = preset("smoke"), torch.device("cuda")
    model = tiny().to(dev)
    rolls = []
    critic = Critic(model.cfg).to(dev).eval().requires_grad_(False)
    for on in (False, True):
        torch.manual_seed(3)
        r = Rollout(cfg.ppo.rollout_len, cfg.env.num_envs, dev, on)
        Collector(cfg, model, dev, 5, 1, lambda o: torch.zeros(o.batch, device=dev), critic).collect(r, lambda eps: None)
        rolls.append(r)
    off, on = rolls
    assert on.host and on.tok_f.is_cpu and on.ops.is_cpu and not on.glob_f.is_cpu and not off.tok_f.is_cpu
    for k, x in vars(off).items():
        if isinstance(x, torch.Tensor):
            assert torch.equal(x.cpu(), getattr(on, k).cpu()), k
    idx = torch.randperm(off.T * off.N, device=dev)[:24]
    a, b, c = off.obs(idx), on.obs(idx, idx.cpu()), on.obs(idx)
    for f in Obs.__dataclass_fields__:
        assert getattr(b, f).is_cuda and torch.equal(getattr(a, f), getattr(b, f)) and torch.equal(getattr(a, f), getattr(c, f)), f
    T, O = int(off.n_tokens.flatten()[idx].max()), int(off.n_ops.flatten()[idx].max())
    acts = off.op.flatten()[idx], off.card_seq.flatten(0, 1)[idx], off.dest.flatten()[idx]
    t = on.obs(idx, idx.cpu(), T, O)  # only the first T tokens and O ops cross from host memory
    assert T < az.MAX_TOKENS and O < az.MAX_OPS and torch.equal(t.tok_f[:, :T], a.tok_f[:, :T]) and torch.equal(t.ops[:, :O], a.ops[:, :O])
    assert all(torch.equal(x, y) for x, y in zip(model.evaluate(a, *acts, T, O), model.evaluate(t, *acts, T, O)))
    for r in (on, off):  # O = 0 (the critic's pass): no op slots at all, the tokens as before
        e = r.obs(idx, idx.cpu(), T, 0)
        assert e.ops.shape == (24, 0, az.OP_INT) and e.ops_f.shape == (24, 0, az.OP_FLOAT) and torch.equal(e.tok_f[:, :T], a.tok_f[:, :T])
        # The critic's values the collector stored while acting are those of a pass over the stored states.
        assert torch.allclose(r.c_val.flatten()[idx], critic.value(e, T), atol=1e-4) and float(r.c_val.std()) > 0
    # As training runs it: the critic's encoder in float16 (values of other batches differ by rounding only).
    half = Critic(model.cfg).to(dev).eval().requires_grad_(False)
    half.load_state_dict(critic.state_dict())
    half.encoder.half()
    r = Rollout(cfg.ppo.rollout_len, cfg.env.num_envs, dev, True)
    Collector(cfg, model, dev, 5, 1, lambda o: torch.zeros(o.batch, device=dev), half).collect(r, lambda eps: None)
    assert torch.allclose(r.c_val.flatten()[idx], half.value(r.obs(idx, idx.cpu(), T, 0), T).float(), atol=5e-3)
    # Refilled storage: only the slots the last fill wrote are cleared, and no row keeps an older state's tokens
    # (deck tokens make the token count change from step to step).
    cfg.env.deck_tokens = True
    col, widths = Collector(cfg, model, dev, 5, 1, lambda o: torch.zeros(o.batch, device=dev)), set()
    for _ in range(3):
        col.collect(on, lambda eps: None)
        pad = torch.arange(az.MAX_TOKENS) >= on.n_tokens.cpu()[..., None]
        assert not on.tok_f[pad].any() and not on.tok_i[pad].any()
        widths |= set(on.n_tokens.amax(1).tolist())
    assert len(widths) > 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Collector needs CUDA")
@pytest.mark.parametrize("host", [False, True])
def test_act_chunks_fill_the_rollout_in_env_order(monkeypatch, host):
    """``ppo.act_chunks``: the policy runs on groups of envs sorted by token count, each at its own length, but the
    rollout is still what a plain loop stores: a twin env given the same actions shows the stored observations,
    rewards and ends, and the stored log-probs are those of the stored actions."""
    cfg, dev = preset("smoke"), torch.device("cuda")
    cfg.env.num_envs, cfg.env.deck_tokens, cfg.ppo.rollout_len, cfg.ppo.fast_rollout, cfg.ppo.act_chunks = 16, True, 24, 1, 4
    cfg.validate()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)  # train() turns TF32 on: acting and evaluating then differ by ~1e-3
    model, sent, pick = tiny().to(dev), [], Collector._pick
    monkeypatch.setattr(Collector, "_pick", lambda self, roll, t, h: sent.append([x.copy() for x in pick(self, roll, t, h)]) or sent[-1])
    roll = Rollout(24, 16, dev, host)
    col = Collector(cfg, model, dev, 5, 1, lambda o: torch.zeros(o.batch, device=dev))
    col.collect(roll, lambda eps: None)
    assert col.chunk == 4 and len(col.act.graphs) > 1 and col.invalid_actions() == 0
    twin = make_env(cfg, seed=5)
    for t, acts in enumerate(sent):
        for k in ("tok_i", "tok_f", "glob_i", "glob_f", "ops", "ops_f", "n_ops", "n_tokens", "n_hand"):
            assert np.array_equal(np.asarray(getattr(twin, k)), getattr(roll, k)[t].cpu().numpy()), (t, k)
        rew, done, _trunc = twin.step(*acts)
        assert np.array_equal(rew, roll.reward[t].cpu().numpy()) and np.array_equal(done, roll.done[t].cpu().numpy() > 0), t
    idx = torch.arange(roll.T * roll.N, device=dev)
    with torch.no_grad():
        lp, _ent, _val = model.evaluate(roll.obs(idx), roll.op.flatten(), roll.card_seq.flatten(0, 1), roll.dest.flatten())
    assert torch.allclose(lp, roll.logp.flatten(), atol=1e-4)
    cfg.ppo.act_chunks = 3
    with pytest.raises(ValueError):
        cfg.validate()


def test_every_saved_checkpoint_is_kept(tmp_path):
    cfg = preset("smoke")
    cfg.run_dir = str(tmp_path)
    train(cfg, max_updates=2)
    run = tmp_path / cfg.name
    assert (run / "u1.pt").exists() and (run / "u2.pt").exists() and not (run / "eval.jsonl").exists()
    assert read_checkpoint(run / "u2.pt")["update"] == 2


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


class _BotGame:
    """Stands in for the game in a bot run (the mod's ``az_bot`` and a
    ``GameEnv``, which keeps the reply to the last request) on a simulator of
    its own. A menu pauses the bot at the polls in ``pauses``, and opens
    between the bot's poll and its action at the actions in ``menus``;
    ``player(game)`` runs while a menu is open: the player changing the game.
    The win screen pauses the bot for ``win_polls`` polls, then the player
    chooses Endless Mode. ``deal(game, code)`` runs after each of the bot's
    actions: the game doing what the simulator would not (a locked card in
    its shop). ``misread(dump)`` changes the dump the bot loads, but not what
    the bridge reads of it."""

    def __init__(self, run, leave_at=None, pauses=(2,), player=None, deal=None, misread=None, g=None, refuse=False, menus=(), win_polls=3):
        self.g = g or az.Game(run["seed"], run["back"], run["stake"])
        self.run, self.leave_at, self.pauses, self.menus, self.refuse = run, leave_at, pauses, menus, refuse
        self.player, self.deal, self.misread, self.win_polls = player, deal, misread, win_polls
        self.polls, self.notes, self.changed, self.free, self.interrupted, self.tries = 0, [], 0, 0, False, 0
        self.refresh()

    def _menu(self):
        self.interrupted = True
        if self.player and self.player(self.g):
            self.changed += 1

    def az_bot(self, run=None, **params):
        self.polls += 1
        win_screen = self.g.run_won and not self.g.endless and self.g.phase == "round_eval"
        if win_screen:
            self.win_polls -= 1
            self.g.endless = self.win_polls < 0  # the player chooses Endless Mode
        paused = self.polls in self.pauses or win_screen
        if paused:
            self._menu()
        interrupted = False
        if run is not None:  # the bot's own poll reports it once
            interrupted, self.interrupted = self.interrupted, False
        active = self.polls != self.leave_at
        return {"active": active, "paused": paused, "interrupted": interrupted, "run": self.run, "search": True}

    def refresh(self):
        d = json.loads(self.g.state_json())
        d["az"] = {"steamodded": self.g.steamodded}  # what the mod adds
        self.reply = json.dumps(d)

    def state(self):
        return json.loads(self.reply)

    def state_json(self):
        return self.misread(self.reply) if self.misread else self.reply

    def canonical(self):
        from antezero_bridge import canon

        return canon.from_lua(self.reply)

    def step(self, code, run, note):
        from antezero_bridge.game_env import ActionRejected

        assert run == self.run["id"]
        self.tries += 1
        if self.tries in self.menus:
            self._menu()
        if self.refuse or self.interrupted:
            raise ActionRejected(code)
        self.notes.append(note)
        self.g.step(code)
        if self.deal and self.deal(self.g, code):
            self.changed += 1
        self.refresh()
        self.free = self.free + 1 if code.startswith(("move", "sort")) else 0  # the bot's rearranges in a row


@pytest.fixture
def bot_game(monkeypatch):
    """Makes a ``_BotGame`` whose bot must hold the game's state, and its
    count of rearranges in a row, at every decision."""
    from antezero_agent import bot

    monkeypatch.setattr(bot.time, "sleep", lambda s: None)
    games = []
    greedy = bot.greedy

    def checked(model, cfg, sim, dev):
        assert sim.canonical() == games[-1].g.canonical()
        assert sim.rearranges == games[-1].free
        assert sim.steamodded == games[-1].g.steamodded  # the game's rounding of ×Mult
        return greedy(model, cfg, sim, dev)

    monkeypatch.setattr(bot, "greedy", checked)

    def make(*a, **kw):
        games.append(_BotGame(*a, **kw))
        return games[-1]

    return make


BOT_RUN = {"id": 7, "seed": "BOTTEST1", "back": "b_red", "stake": 1}


def test_the_bot_plays_its_run_to_the_end_in_step_with_the_game(bot_game):
    from antezero_agent import bot

    cfg, net, run = preset("smoke"), tiny(), BOT_RUN
    critic = Critic(cfg.model).eval()
    game = bot_game(run)
    assert bot.play_run(game, game, run, net, cfg, "cpu", None, critic) == "lost the run"  # an untrained policy
    assert game.g.is_terminal() and len(game.notes) > 5 and all("sure, win " in n for n in game.notes)
    assert game.polls == len(game.notes) + 2  # one poll per action and at the end, and one more while a menu paused it
    # the player leaves the run: the bot stops without a result
    game = bot_game(run, leave_at=4)
    assert bot.play_run(game, game, run, net, cfg, "cpu", None, None) is None and len(game.notes) == 2
    # the game refuses the bot's action: the bot hands the run back
    game = bot_game(run, refuse=True)
    assert bot.play_run(game, game, run, net, cfg, "cpu", None, None).startswith("stopped")


def _in_a_shop_with_a_joker() -> az.Game:
    """A run that has reached a shop selling a joker (random play)."""
    rng = random.Random(0)
    for _ in range(1000):
        g = az.Game(BOT_RUN["seed"], BOT_RUN["back"], BOT_RUN["stake"])
        while not g.is_terminal():
            if g.phase == "shop" and any(c["ability"]["set"] == "Joker" for c in json.loads(g.state_json())["shop_jokers"]):
                return g
            g.step(rng.choice(g.legal_codes()))
    raise AssertionError("no run reached a shop")


def _blueprint_in_the_shop(g, code=None) -> bool:
    """The shop just dealt holds a Blueprint where the simulator dealt
    another joker (a locked card's slot, say)."""
    if g.phase != "shop" or code not in (None, "cash_out", "reroll"):
        return False
    d = json.loads(g.state_json())
    jokers = [c for c in d["shop_jokers"] if c["ability"]["set"] == "Joker"]
    if not jokers:
        return False
    jokers[0]["key"] = "j_blueprint"
    g.load_state(json.dumps(d))
    return True


def test_the_bot_plays_a_shop_the_simulator_would_not_deal(bot_game):
    from antezero_agent import bot

    g = _in_a_shop_with_a_joker()
    assert _blueprint_in_the_shop(g)
    cfg, net = preset("smoke"), tiny()
    game = bot_game(BOT_RUN, deal=_blueprint_in_the_shop, g=g)
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) == "lost the run"


def test_the_bot_reads_the_game_after_the_player_changed_it(bot_game):
    from antezero_agent import bot

    def player(g):
        if g.is_terminal():
            return False
        g.step(g.legal_codes()[-1])
        return True

    cfg, net = preset("smoke"), tiny()
    game = bot_game(BOT_RUN, pauses=range(2, 1000, 3), player=player)
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) in ("lost the run", "won the run")
    assert game.changed > 1


def test_the_bot_plays_the_rules_of_the_game_it_reads(bot_game):
    from antezero_agent import bot

    cfg, net = preset("smoke"), tiny()
    for steamodded in (False, True):
        g = az.Game(BOT_RUN["seed"], BOT_RUN["back"], BOT_RUN["stake"])
        g.steamodded = steamodded
        game = bot_game(BOT_RUN, g=g)
        assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) == "lost the run"


def test_the_bot_decides_again_on_a_menu_opened_before_its_action(bot_game):
    from antezero_agent import bot

    def player(g):
        if g.is_terminal():
            return False
        g.step(g.legal_codes()[-1])
        return True

    cfg, net = preset("smoke"), tiny()
    game = bot_game(BOT_RUN, menus=range(3, 1000, 4), player=player)
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) == "lost the run"
    assert game.changed > 1 and game.tries > len(game.notes)


def _won() -> az.Game:
    """A run at the round evaluation of a won boss (the win screen)."""
    rng = random.Random(0)
    g = az.Game(BOT_RUN["seed"], BOT_RUN["back"], BOT_RUN["stake"])
    while g.phase != "round_eval":
        if g.is_terminal():
            g = az.Game(BOT_RUN["seed"], BOT_RUN["back"], BOT_RUN["stake"])
        g.step(rng.choice(g.legal_codes()))
    d = json.loads(g.state_json())
    d["won"] = d["win_notified"] = True
    g.load_state(json.dumps(d))
    assert g.run_won and not g.endless
    return g


def test_the_bot_plays_on_in_endless_mode_once_the_player_chooses_it(bot_game):
    from antezero_agent import bot

    cfg, net = preset("smoke"), tiny()
    game = bot_game(BOT_RUN, pauses=(), g=_won())
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None).startswith("lost in Endless Mode at ante ")
    assert game.notes and game.polls > len(game.notes) + 3  # it waited on the win screen


def test_the_bot_plays_on_past_an_unlock_alert():
    """Steamodded marks the placeholder an unlock alert shows (``j_locked``)
    in ``used_jokers``: no card of any pool."""
    from antezero_agent import bot

    class Alerted(_BotGame):
        def refresh(self):
            super().refresh()
            d = json.loads(self.reply)
            d["used_jokers"] = sorted(d["used_jokers"] + ["j_locked", "j_undiscovered"])
            self.reply = json.dumps(d)

    cfg, net = preset("smoke"), tiny()
    game = Alerted(BOT_RUN)
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) == "lost the run"


def test_the_bot_stops_on_a_run_the_simulator_cannot_hold(bot_game):
    from antezero_agent import bot

    def challenge(dump):
        d = json.loads(dump)
        d["challenge"] = "c_omelette_1"
        return json.dumps(d)

    cfg, net = preset("smoke"), tiny()
    game = bot_game(BOT_RUN, misread=challenge)
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) == "stopped: the simulator cannot play this run"


def test_the_bot_stops_on_a_dump_its_simulator_misreads(bot_game):
    from antezero_agent import bot

    def misread(dump):
        d = json.loads(dump)
        d["dollars"] += 1
        return json.dumps(d)

    cfg, net = preset("smoke"), tiny()
    game = bot_game(BOT_RUN, misread=misread)
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) == "stopped: the simulator misread the game"
    assert not game.notes


def test_the_bot_stops_when_the_game_does_not_do_what_it_expected(bot_game):
    """A dump the simulator misreads where the bridge does not look (the
    discards the next blind adds): the next blind shows it."""
    from antezero_agent import bot

    def misread(dump):
        d = json.loads(dump)
        d["round_bonus"]["discards"] += 1
        return json.dumps(d)

    cfg, net = preset("smoke"), tiny()
    game = bot_game(BOT_RUN, misread=misread)
    assert bot.play_run(game, game, BOT_RUN, net, cfg, "cpu", None, None) == "stopped: the game did not do what the simulator expected"
    assert game.notes


def test_what_the_game_may_do_otherwise_is_no_surprise():
    from antezero_agent.bot import surprises

    before = {"jokers.len": "1", "jokers.0.id": "10", "jokers.0.key": "j_joker", "hand_limit": "8", "hand.0.id": "3"}
    expected = dict(before, **{"jokers.len": "2", "jokers.1.id": "11", "jokers.1.key": "j_troubadour", "hand_limit": "10"})
    # a locked card's place: the game made another joker, with other effects
    game = dict(expected, **{"jokers.1.key": "j_juggler", "hand_limit": "9"})
    assert surprises(before, expected, game) == []
    assert surprises(before, expected, dict(expected, hand_limit="9")) == [("hand_limit", "10", "9")]


def test_the_bot_hands_its_run_back_after_losing_contact(monkeypatch):
    import httpx
    from antezero_agent import bot
    from antezero_bridge import client

    class Client:
        def __init__(self, port):
            self.polls, self.stops = 0, []

        def az_bot(self, run=None, stop=None, status=None):
            if stop:
                self.stops.append((run, stop))
            self.polls += 1
            if self.polls > 4:
                raise httpx.ConnectError("the game closed")
            return {"active": True, "run": dict(BOT_RUN)}

    def lost(*a):
        raise httpx.ReadTimeout("no reply")

    made = []
    monkeypatch.setattr(client, "BridgeClient", lambda port: made.append(Client(port)) or made[-1])
    monkeypatch.setattr(bot, "load_model", lambda ckpt, dev: (None, None, None))
    monkeypatch.setattr(bot, "read_checkpoint", lambda ckpt: {})
    monkeypatch.setattr(bot, "play_run", lost)
    monkeypatch.setattr(bot.time, "sleep", lambda s: None)
    bot.serve(argparse.Namespace(port=1, device="cpu", checkpoint="ckpt"))
    assert made[0].stops == [(BOT_RUN["id"], "stopped: lost contact with the game (ReadTimeout)")]


@pytest.mark.skipif(os.name != "nt", reason="the mod starts the bot through cmd.exe")
def test_the_bot_command_keeps_its_arguments_through_cmd(tmp_path):
    from antezero_agent.bot import command_line

    ckpt, out = tmp_path / "a&b(1)^x" / "last.pt", tmp_path / "argv.json"
    script = "import json, sys; open(sys.argv[2], 'w').write(json.dumps(sys.argv[1:]))"
    cmd = command_line([sys.executable, "-c", script, str(ckpt), str(out)])
    subprocess.run(f'start "AnteZero bot" /b /wait {cmd}', shell=True, check=True)  # as bot.lua starts it
    assert json.loads(out.read_text()) == [str(ckpt), str(out)]
