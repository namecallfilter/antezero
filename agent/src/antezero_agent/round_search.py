"""Risk-gated within-round search: decision-time policy improvement over
determinized worlds, batched across games.

At a round decision (free rearranges, ``move_*`` / ``sort``, are never
searched) the greedy action is played in ``gate`` *worlds*: clones of the
game whose hidden state (deck order, face-down cards, every future random
outcome) is redrawn by ``Game.resample_hidden`` with a tag made of the game
and decision indices only, never of the seed or of hidden state. Each world
is rolled out with the greedy policy to the end of the round. If every world
clears the round, greedy is played. Otherwise the candidates (greedy,
``samples`` policy samples, the scored play candidates and, on the last
hand, every legal play) are scored on the same worlds; up to ``keep`` of
them that clear at least as often as greedy meet greedy again on ``final``
fresh worlds (after the first half only if greedy does not strictly lead
there). The most clears wins and greedy wins ties. The true game only acts;
it is rolled out only to attribute a finished round (statistics).

The opt-in endgame search (``endgame=True``, :meth:`RoundSearch.decide_endgame`)
runs the same stages, with its own world counts, at the shop and pack
decisions of ante 8 before the win, where the build for the last blinds is
settled. Its worlds play to the end of the game (won or lost), its
candidates are every non-free op of the observation (cards picked by the
greedy card pointer), a finalist needs a screen win, and a finalist
overrides greedy only if it wins significantly more of the final worlds (a
one-sided exact sign test on the worlds exactly one of them wins, at
``EG_ALPHA``): with ~10 candidates, a fixed margin of 2 wins out of 32
let noise override greedy in states it already won ~90% of the time.

Every game is a generator yielding policy requests; :meth:`RoundSearch.run`
serves the requests of all live games (main games, rollouts, samples) in
chunked forward passes and encodes the next chunk on a thread pool while the
device runs the current one. On u5200 (512 paired games, .scratch/search)
the search took the win rate from 42.6% to 53.5% at ~106x the forwards.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import os
from collections import Counter, deque

import antezero_sim as az
import numpy as np
import torch

from .config import Config
from .model import MAX_HAND, Net, Obs

FIELDS = {"tok_i": torch.long, "tok_f": torch.float32, "glob_i": torch.long, "glob_f": torch.float32, "ops": torch.long, "ops_f": torch.float32, "n_ops": torch.long, "n_tokens": torch.long, "n_hand": torch.long}
FREE = ("move", "sort")  # free rearranges: never searched
ROLL_MAX = 100  # steps per rollout; a world still in the round counts as not cleared
END_ANTE = 8  # the endgame search's ante (the run is won at its boss)
GAME_MAX = 1000  # steps per endgame rollout; a world still playing counts as lost
EG_ALPHA = 0.05  # significance an endgame override needs


def in_round(g: az.Game) -> bool:
    return g.phase == "round" and not g.is_terminal()


def in_game(g: az.Game) -> bool:
    return not g.is_terminal() and not g.run_won


def cleared(g: az.Game) -> bool:
    return g.run_won or g.phase not in ("round", "game_over")


def frac(g: az.Game) -> float:
    return min(1.0, g.chips / g.blind_chips) if g.blind_chips > 0 else 1.0


def stack(ds: list[dict], device) -> Obs:
    """``Game.encode`` dicts -> one batch, cut to its longest token / op
    lists (the model masks the padding)."""
    T = max(max(int(d["n_tokens"]) for d in ds), MAX_HAND, 1)
    O = max(max(int(d["n_ops"]) for d in ds), 1)
    cut = {"tok_i": T, "tok_f": T, "ops": O, "ops_f": O}
    return Obs(**{k: torch.from_numpy(np.stack([d[k][: cut[k]] if k in cut else d[k] for d in ds])).to(device, t) for k, t in FIELDS.items()})


def to_code(g: az.Game, d: dict, op: int, cards: int, dest: int) -> str | None:
    """The action code of a factored choice over ``d`` (``enc(g)``'s ops),
    ``None`` if illegal, without scoring the play candidates again: they come
    first and the other ops are ``encode(0)``'s in the same order."""
    if op < d["_k"]:
        return f"play {int(d['ops'][op, 4]) & 0xFFFFFFFF}"
    return g.factored_to_code(op - d["_k"], cards & 0xFFFFFFFF, max(dest, 0), 0)


def _fetch(out: dict) -> list[tuple]:
    return list(zip(out["op"].tolist(), out["cards"].tolist(), out["dest"].tolist(), out["card_seq"].tolist()))


def _parts(xs: list, n: int) -> list[list]:
    k = max(1, -(-len(xs) // n))
    return [xs[i : i + k] for i in range(0, len(xs), k)]


def _tally(codes: list[str], res: list[tuple], n: int) -> dict:
    """{code: (clears, chip fraction sum)} of results laid out code-major."""
    return {c: (sum(x[0] for x in res[i * n : (i + 1) * n]), sum(x[1] for x in res[i * n : (i + 1) * n])) for i, c in enumerate(codes)}


def _sign_p(a: list, b: list) -> float:
    """One-sided exact sign test of per-world wins: the chance that ``a``
    wins at least this many of the worlds exactly one of ``a``, ``b`` wins
    if both were as good."""
    x, n = sum(p > q for p, q in zip(a, b)), sum(p != q for p, q in zip(a, b))
    return sum(math.comb(n, k) for k in range(x, n + 1)) / 2**n


class RoundSearch:
    """The search for one policy. Requests a game generator may yield:
    ``("act", g)`` -> ``(greedy code, enc(g), (op, cards, dest, card_seq))``;
    ``("sample", g, d, S)`` -> S sampled codes; ``("ops", g, d)`` -> the
    legal non-free code of every op, its cards picked greedily;
    ``("step", worlds, label)`` -> every world takes one greedy step in
    place."""

    def __init__(self, model: Net, cfg: Config, device="cpu", gate: int = 16, final: int = 32, keep: int = 4, samples: int = 16, min_ante: int = 1, threads: int = 0, chunk: int = 1024, cap: int = 16384, endgame: bool = False, eg_gate: int = 8, eg_final: int = 32, eg_keep: int = 2, eg_min_ante: int = END_ANTE):
        if min(gate, keep, eg_gate, eg_keep) < 1 or min(final, eg_final) < 2 or samples < 0:
            raise ValueError("need gate, keep, eg_gate, eg_keep >= 1, final, eg_final >= 2, samples >= 0")
        self.model, self.device = model.eval(), torch.device(device)
        self.K, self.dt, self.pv, self.limit = cfg.env.play_candidates, cfg.env.deck_tokens, cfg.env.build_previews, cfg.env.step_limit
        self.gate, self.final, self.keep, self.samples, self.min_ante = gate, final, keep, samples, min_ante
        self.endgame, self.eg_gate, self.eg_final, self.eg_keep, self.eg_min_ante = endgame, eg_gate, eg_final, eg_keep, eg_min_ante
        self.threads = threads or os.cpu_count() or 1
        self.pool = cf.ThreadPoolExecutor(self.threads)
        self.chunk, self.cap = chunk, cap
        self.rows: Counter = Counter()  # policy rows by purpose (cost accounting)

    def enc(self, g: az.Game) -> dict:
        d = g.encode(self.K, self.dt, self.pv)
        kinds = d["ops"][: min(self.K, int(d["n_ops"])), 0].tolist()
        d["_k"] = next((i for i, x in enumerate(kinds) if x != az.OP_PLAY_TOP), len(kinds))  # leading scored candidates
        return d

    def searched(self, g: az.Game, code: str) -> bool:
        return g.phase == "round" and g.ante >= self.min_ante and not code.startswith(FREE)

    def endgame_at(self, g: az.Game, code: str) -> bool:
        """Whether the endgame search decides here: a shop or pack from ante
        ``eg_min_ante`` (default 8) to the final ante, before the win, or a
        blind select there that can reroll the boss."""
        if not (self.endgame and self.eg_min_ante <= g.ante <= END_ANTE and not g.run_won and not code.startswith(FREE)):
            return False
        return g.phase in ("shop", "pack") or (g.phase == "blind_select" and "reroll_boss" in g.legal_codes())

    # ------------------------------------------------------------- scheduler
    def run(self, gens: list) -> list:
        """Runs generators to completion -> their return values (in the order
        they finish)."""
        done, q = [], deque()
        for gen in gens:
            self._advance(gen, None, q, done)
        it = 0
        while q:
            reqs, n = [], 0
            while q and (n < self.cap or not reqs):
                gen, r = q.popleft()
                reqs.append((gen, r))
                n += len(r[1]) if r[0] == "step" else 1
            rows, samp, comp = [], [], []
            for ri, (_, r) in enumerate(reqs):
                if r[0] == "act":
                    rows.append((ri, r[1], False))
                    self.rows["act"] += 1
                elif r[0] == "step":
                    rows += [(ri, w, True) for w in r[1]]
                    self.rows[r[2]] += len(r[1])
                elif r[0] == "ops":
                    comp.append((ri, r))
                    self.rows["ops"] += int(r[2]["n_ops"])
                else:
                    samp.append((ri, r))
                    self.rows["sample"] += r[3]
            results = [None] * len(reqs)
            self._greedy(rows, results)
            if samp:
                self._sample(samp, results)
            if comp:
                self._complete(comp, results)
            for (gen, _), v in zip(reqs, results):
                self._advance(gen, v, q, done)
            it += 1
            if it % 50 == 0 and self.device.type == "cuda":
                torch.cuda.empty_cache()  # variable batch shapes fragment the cache
        return done

    @staticmethod
    def _advance(gen, v, q, done):
        try:
            q.append((gen, gen.send(v)))
        except StopIteration as e:
            done.append(e.value)

    def _encode(self, gs: list) -> list[dict]:
        return [d for p in self.pool.map(lambda xs: [self.enc(g) for g in xs], _parts(gs, self.threads * 4)) for d in p]

    def _greedy(self, rows: list, results: list):
        """Greedy actions of ``(request, game, is_world)`` rows: worlds step in
        place, main games get their result. Chunk i+1 is encoded and chunk
        i-1 post-processed on the CPU while the device runs chunk i."""
        chunks = [rows[i : i + self.chunk] for i in range(0, len(rows), self.chunk)]
        prev = None
        for ch in chunks + [None]:
            ds = self._encode([g for _, g, _ in ch]) if ch else None
            acts = _fetch(prev[2]) if prev else None
            nxt = (ch, ds, self.model.act(stack(ds, self.device), deterministic=True)) if ch else None
            if prev:
                self._post(prev[0], prev[1], acts, results)
            prev = nxt

    def _post(self, rows, ds, acts, results):
        def one(xs):
            for (ri, g, is_world), d, a in xs:
                if not is_world:
                    code = to_code(g, d, *a[:3])
                    assert code is not None, ("policy action not legal", a)
                    results[ri] = (code, d, a)
                elif a[0] < d["_k"]:
                    g.step(f"play {int(d['ops'][a[0], 4]) & 0xFFFFFFFF}")
                else:
                    g.step_factored(a[0] - d["_k"], a[1] & 0xFFFFFFFF, max(a[2], 0), 0)

        list(self.pool.map(one, _parts(list(zip(rows, ds, acts)), self.threads * 4)))

    def _sample(self, samp: list, results: list):
        flat = [r[2] for _, r in samp for _ in range(r[3])]
        out = [x for i in range(0, len(flat), self.chunk) for x in _fetch(self.model.act(stack(flat[i : i + self.chunk], self.device)))]
        k = 0
        for ri, (_, g, d, m) in samp:
            results[ri] = [to_code(g, d, *x[:3]) for x in out[k : k + m]]
            assert None not in results[ri], "policy sample not legal"
            k += m

    def _complete(self, comp: list, results: list):
        """Each op alone in its observation (an op's embedding and card
        pointer do not depend on the other ops), so greedy picks its cards."""
        one = [{**d, "ops": d["ops"][j : j + 1], "ops_f": d["ops_f"][j : j + 1], "n_ops": 1} for _, (_, _, d) in comp for j in range(int(d["n_ops"]))]
        out = iter([x for i in range(0, len(one), self.chunk) for x in _fetch(self.model.act(stack(one[i : i + self.chunk], self.device), deterministic=True))])
        for ri, (_, g, d) in comp:
            codes = [to_code(g, d, j, *next(out)[1:3]) for j in range(int(d["n_ops"]))]
            results[ri] = list(dict.fromkeys(c for c in codes if c and not c.startswith(FREE)))

    # ------------------------------------------------------------ generators
    def rollout(self, ws: list, label: str, to_end: bool = False):
        """Greedy policy until every world leaves the round -> [(cleared,
        chip fraction)]; ``to_end``: until the game is won or lost -> [(won,
        blinds cleared + chip fraction)]."""
        live = in_game if to_end else in_round
        act = [w for w in ws if live(w)]
        for _ in range(GAME_MAX if to_end else ROLL_MAX):
            if not act:
                break
            yield ("step", act, label)
            act = [w for w in act if live(w)]
        return [(w.run_won, w.blinds_cleared + frac(w)) if to_end else (cleared(w), frac(w)) for w in ws]

    @staticmethod
    def worlds(g: az.Game, n: int, tag: str) -> list:
        ws = []
        for j in range(n):
            w = g.clone()
            w.resample_hidden(f"{tag}_{j}")
            ws.append(w)
        return ws

    def branch(self, ws: list, codes: list[str]) -> tuple[list, list[str]]:
        """Each legal code applied to a clone of each world -> (worlds,
        code-major; the legal codes). Legality is visible state, so it is
        checked on the first world."""
        ok = []
        for code in codes:
            try:
                ws[0].clone().step(code)
                ok.append(code)
            except (RuntimeError, ValueError):
                pass

        def one(xs):
            out = []
            for code, w in xs:
                c = w.clone()
                c.step(code)
                out.append(c)
            return out

        jobs = [(c, w) for c in ok for w in ws]
        return [w for p in self.pool.map(one, _parts(jobs, self.threads * 2)) for w in p], ok

    def decide(self, tag: str, g: az.Game, d: dict, gcode: str, st: Counter):
        """Risk-gated search at one round decision (``d = enc(g)``, greedy
        code ``gcode``) -> (code, info); ``info["risky"]`` tells whether some
        gate world failed. ``tag`` must not depend on hidden state."""
        st["gated"] += 1
        w0 = self.worlds(g, self.gate, "G" + tag)
        s1 = _tally([gcode], (yield from self.rollout(self.branch(w0, [gcode])[0], "gate")), self.gate)
        info = {"risky": s1[gcode][0] < self.gate, "gate_clears": s1[gcode][0]}
        if not info["risky"]:
            return gcode, info
        st["risky"] += 1
        src = {gcode: "greedy"}
        for c in (yield ("sample", g, d, self.samples)):
            src.setdefault(c, "sample")
        for i in range(d["_k"]):
            src.setdefault(f"play {int(d['ops'][i, 4]) & 0xFFFFFFFF}", "top")
        last = int(g.canonical()["hands_left"]) == 1
        if last:
            for c in g.legal_codes():
                if c.startswith("play "):
                    src.setdefault(c, "all")
        cands = list(src)
        ws, others = self.branch(w0, cands[1:])
        st["illegal"] += len(cands) - 1 - len(others)
        s1.update(_tally(others, (yield from self.rollout(ws, "screen")), self.gate))
        g1 = s1[gcode][0]
        keep = sorted((c for c in others if s1[c][0] >= g1), key=lambda c: (-s1[c][0], -s1[c][1]))[: self.keep]
        info.update(cands=len(cands), last=last, finalists=len(keep))
        if not keep:
            return gcode, info
        st["finals"] += 1
        fin = [gcode] + keep
        w1 = self.worlds(g, self.final, "F" + tag)
        h = self.final // 2
        ws, ok = self.branch(w1[:h], fin)
        assert ok == fin, "a finalist is illegal in a fresh world"  # legality is visible state; keeps the tally aligned
        s2 = _tally(fin, (yield from self.rollout(ws, "final")), h)
        nw = h
        if any(s2[c][0] >= s2[gcode][0] for c in keep):
            ws, _ = self.branch(w1[h:], fin)
            s2b = _tally(fin, (yield from self.rollout(ws, "final")), self.final - h)
            s2 = {c: (s2[c][0] + s2b[c][0], s2[c][1] + s2b[c][1]) for c in fin}
            nw = self.final
        best = max(fin, key=lambda c: (s2[c][0], c == gcode, s2[c][1]))
        info.update(final_worlds=nw, final_greedy=s2[gcode][0], final_best=s2[best][0], src=src[best], rank=keep.index(best) if best != gcode else -1)
        st["overrides"] += best != gcode
        return best, info

    def decide_endgame(self, tag: str, g: az.Game, d: dict, gcode: str, st: Counter):
        """:meth:`decide` at an :meth:`endgame_at` decision, with worlds
        played to the end of the game -> (code, info). ``tag`` must not
        depend on hidden state."""
        st["eg_gated"] += 1
        w0 = self.worlds(g, self.eg_gate, "eg_G" + tag)
        s1 = _tally([gcode], (yield from self.rollout(self.branch(w0, [gcode])[0], "eg_gate", True)), self.eg_gate)
        info = {"risky": s1[gcode][0] < self.eg_gate, "gate_wins": s1[gcode][0]}
        if not info["risky"]:
            return gcode, info
        st["eg_risky"] += 1
        cands = list(dict.fromkeys([gcode] + (yield ("ops", g, d))))
        ws, others = self.branch(w0, cands[1:])
        st["illegal"] += len(cands) - 1 - len(others)
        s1.update(_tally(others, (yield from self.rollout(ws, "eg_screen", True)), self.eg_gate))
        keep = sorted((c for c in others if s1[c][0] >= max(s1[gcode][0], 1)), key=lambda c: (-s1[c][0], -s1[c][1]))[: self.eg_keep]
        info.update(cands=len(cands), finalists=len(keep))
        if not keep:
            return gcode, info
        st["eg_finals"] += 1
        fin = [gcode] + keep
        w1 = self.worlds(g, self.eg_final, "eg_F" + tag)
        won = {c: [] for c in fin}  # final outcomes, world by world
        for part in (w1[: self.eg_final // 2], w1[self.eg_final // 2 :]):
            ws, ok = self.branch(part, fin)
            assert ok == fin, "a finalist is illegal in a fresh world"
            res = yield from self.rollout(ws, "eg_final", True)
            for i, c in enumerate(fin):
                won[c] += [x[0] for x in res[i * len(part) : (i + 1) * len(part)]]
            if not any(sum(won[c]) > sum(won[gcode]) for c in keep):
                break  # no finalist leads greedy
        p = {c: _sign_p(won[c], won[gcode]) for c in keep}
        best = max((c for c in keep if p[c] <= EG_ALPHA), key=lambda c: sum(won[c]), default=gcode)
        info.update(final_worlds=len(won[gcode]), final_greedy=sum(won[gcode]), final_best=max(sum(won[c]) for c in keep), p=min(p.values()), rank=keep.index(best) if best != gcode else -1)
        st["eg_overrides"] += best != gcode
        return best, info

    def play(self, g: az.Game, gi: int, search: bool = True, max_ante: int | None = None, watch=None):
        """Plays ``g`` (in place) to the end of the game -> record. ``gi``
        names the game in world tags. Rounds whose actions differ from
        greedy are replayed greedily from their start in the true game
        (``greedy_clear``: statistics only). ``watch(g, n, d, act, greedy,
        code, info)`` sees every state before its step (``info``: the
        search's, ``None`` where it did not run)."""
        st: Counter = Counter()
        rec = {"slot": gi, "seed": g.seed, "search": search, "rounds": []}
        rs = None
        n = 0
        while not g.is_terminal() and n < self.limit and not (max_ante and g.endless and g.ante > max_ante):
            code, d, a = yield ("act", g)
            greedy, info = code, None
            if g.phase == "round":
                if rs is None:
                    rs = {"snap": g.clone(), "ante": g.ante, "blind": g.canonical()["blind.name"], "o0": st["overrides"], "n0": n}
                if search and self.searched(g, code):
                    code, info = yield from self.decide(f"{gi}_{n}", g, d, code, st)
            elif search and self.endgame_at(g, code):
                code, info = yield from self.decide_endgame(f"{gi}_{n}", g, d, code, st)
            if watch:
                watch(g, n, d, a, greedy, code, info)
            g.step(code)
            n += 1
            if rs is not None and not in_round(g):
                r = {"ante": rs["ante"], "blind": rs["blind"], "steps": [rs["n0"], n], "clear": cleared(g), "overrides": st["overrides"] - rs["o0"]}
                if r["overrides"]:
                    ((r["greedy_clear"], _),) = yield from self.rollout([rs["snap"].clone()], "counterfactual")
                rec["rounds"].append(r)
                rs = None
        rec.update(won=g.run_won, ante=g.ante, blinds=g.blinds_cleared, steps=n, truncated=not g.is_terminal())
        rec.update({k: st[k] for k in ("gated", "risky", "finals", "overrides", "illegal", "eg_gated", "eg_risky", "eg_finals", "eg_overrides")})
        return rec

    def choose(self, g: az.Game, tag: str) -> str:
        """The search's action for one state (``tag``: e.g. game and step
        counters, never the seed)."""

        def one():
            code, d, _a = yield ("act", g)
            if self.searched(g, code):
                code, _info = yield from self.decide(tag, g, d, code, Counter())
            elif self.endgame_at(g, code):
                code, _info = yield from self.decide_endgame(tag, g, d, code, Counter())
            return code

        return self.run([one()])[0]
