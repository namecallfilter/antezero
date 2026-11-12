import csv

import numpy as np
import pytest
import torch

import antezero_sim as az
from antezero_agent.collect import Collector
from antezero_agent.config import Config, ModelConfig, parse_overrides, preset
from antezero_agent.critic import Critic
from antezero_agent.model import NEG, OP_END_SHOP, OP_REDEEM, Net, Obs, noise_shapes, obs_from_numpy, sample
from antezero_agent.runtime import load_model, make_env, read_checkpoint, seeded
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


def test_bf16_encoder_changes_nothing_on_cpu():
    """``model.bf16_encoder`` is a CUDA setting: on the CPU the same network gives the same numbers."""
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
        assert checked >= 20 and progress >= 5  # some checked episodes cleared blinds


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
    must not be offered skips there (it skipped blinds in the real game)."""
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
    """``search.shaped_reward`` is the reward ``VecEnv`` gives, terminal
    transitions included, with Phi kept or zeroed at terminal states."""
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
        out = m.act(o, explore=1.0)
        chosen = kinds.gather(1, out["op"][:, None]).squeeze(1)
        assert not (red & (chosen == OP_END_SHOP)).any()  # every sampled exit with a legal voucher redeems it
        x = out["explored"]
        assert (red | ~x).all() and (chosen[x] == OP_REDEEM).all() and (out["cards"][x] == 0).all()
        logp, _, _ = m.evaluate(o, out["op"], out["card_seq"], out["dest"])
        assert torch.allclose(logp, out["logp"], atol=1e-4)  # the policy's own log-prob of the explored action
        assert torch.equal(m.act(o, deterministic=True, explore=1.0)["op"], m.act(o, deterministic=True)["op"])
        for i in np.flatnonzero(x.numpy()):  # the kind constants match the simulator's
            assert env.game(int(i)).factored_to_code(int(out["op"][i]), 0, 0, cfg.env.play_candidates).startswith("redeem")
        explored += int(x.sum())
        heuristic(env)  # leaves every shop, so money (and affordable vouchers) accumulate
    assert explored > 0


@pytest.mark.parametrize("all_explored", [False, True])
def test_explore_redeem_trains(tmp_path, monkeypatch, all_explored):
    """The loss branch of explored samples and the KL without them run. A smoke policy never reaches a
    shop, so half the rows (or all: a chunk of explored rows only has no KL) are marked explored: finite, and
    not the loss of the same run unmarked."""
    act = Net.act
    losses = []
    for mark in (False, True):

        def marked(self, o, deterministic=False, explore=0.0, mark=mark):
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
        for explore in (0.0, 1.0):
            with seeded(1):
                a = m.act(o, explore=explore)
            with seeded(1):
                qs = [torch.empty(s).uniform_() if u else torch.empty(s).exponential_(1) for s, u in noise_shapes(o.batch, explore)]
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
    # Other env.num_envs / ppo.rollout_len with the same product resume; another product does not.
    cfg.env.num_envs, cfg.ppo.rollout_len, cfg.ppo.total_steps = 4, 16, 4 * 64
    assert train(cfg, resume=str(ck), max_updates=1)["update"] == 3
    cfg.env.num_envs = 2
    with pytest.raises(ValueError):
        train(cfg, resume=str(ck), max_updates=1)


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


def test_every_saved_checkpoint_is_kept(tmp_path):
    """A copy stays at every save (what gets averaged), with or without an in-run eval."""
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
