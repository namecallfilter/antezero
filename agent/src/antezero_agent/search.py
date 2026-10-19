"""Determinized lookahead on top of the policy/value net (test-time
policy improvement).

For a decision we take the policy's top actions (argmax plus samples),
and for each one average ``r + gamma * V(s')`` over several *determinized*
futures: copies of the game whose hidden state (deck order, face-down
cards, all future random outcomes) is resampled with
``Game.resample_hidden``. Every action is tried in the same sampled worlds,
so their comparison is not blurred by luck. The agent never uses
information a player would not have. Rewards are the training rewards
(same shaping), so ``V`` is used on the scale it was trained on.
"""

from __future__ import annotations

import antezero_sim as az
import torch

from .config import Config
from .model import Net, Obs, obs_from_numpy
from .runtime import seeded


def _stack(obs_list: list[Obs]) -> Obs:
    return Obs(*(torch.cat([getattr(o, f) for o in obs_list], 0) for f in Obs.__dataclass_fields__))


def shaped_reward(before: az.Game, after: az.Game, cfg: Config) -> float:
    """The reward ``VecEnv`` gives for the transition ``before -> after``."""
    e = cfg.env
    r = 0.0
    if after.run_won and not before.run_won:
        r += e.r_win
    elif before.run_won:
        r += e.r_endless * (after.blinds_cleared - before.blinds_cleared)
    if after.phase == "game_over":
        r -= e.r_loss
    phi_after = 0.0 if after.is_terminal() else after.potential(e.w_blind, e.w_score)
    return r + cfg.ppo.gamma * phi_after - before.potential(e.w_blind, e.w_score)


@torch.no_grad()
def choose(
    model: Net,
    game: az.Game,
    cfg: Config | None = None,
    n_actions: int = 8,
    n_samples: int = 4,
    seed: int = 0,
    device="cpu",
    play_candidates: int | None = None,
    deck_tokens: bool | None = None,
) -> str:
    """Best action code for ``game`` by determinized one-step lookahead
    (deterministic given ``seed``)."""
    cfg = cfg or Config()
    K = cfg.env.play_candidates if play_candidates is None else play_candidates
    dt = cfg.env.deck_tokens if deck_tokens is None else deck_tokens
    model.eval()
    o = obs_from_numpy(game.encode(K, dt), device)
    cands: dict[str, None] = {}
    greedy = model.act(o, deterministic=True)
    batch = _stack([o] * max(1, n_actions * 2))
    with seeded(seed, device):
        sampled = model.act(batch)
    for out, i in [(greedy, 0)] + [(sampled, j) for j in range(batch.batch)]:
        code = game.factored_to_code(int(out["op"][i]), int(out["cards"][i]), int(out["dest"][i]), K)
        if code is not None:
            cands.setdefault(code)
        if len(cands) >= n_actions:
            break
    codes = list(cands)
    if len(codes) == 1:
        return codes[0]
    nexts, rewards, owners = [], [], []
    for k, code in enumerate(codes):
        for s in range(n_samples):
            c = game.clone()
            c.resample_hidden(f"LA{seed}_{s}")
            c.step(code)
            rewards.append(shaped_reward(game, c, cfg))
            owners.append(k)
            nexts.append(c)
    vals = torch.zeros(len(nexts))
    live = [i for i, c in enumerate(nexts) if not c.is_terminal()]
    if live:
        ob = _stack([obs_from_numpy(nexts[i].encode(K, dt), device) for i in live])
        cls, _, _, _ = model.encode(ob)
        vals[live] = model.value(cls).squeeze(-1).float().cpu()
    score = torch.zeros(len(codes))
    for i, k in enumerate(owners):
        score[k] += (rewards[i] + cfg.ppo.gamma * float(vals[i])) / n_samples
    return codes[int(score.argmax())]
