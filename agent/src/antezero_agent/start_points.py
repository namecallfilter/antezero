"""Start pools: mid-run states of recorded games that training episodes can start from (BalatroEnv.set_start)."""
import json
import re
from pathlib import Path

COND = re.compile(r"^\s*(!?)([a-z_0-9]+)\s*(>=|<=|!=|=|>|<|:)\s*(.+?)\s*$")


def parse_filter(spec):
    """'ante>=3,jokers:j_baron' -> a predicate over a tags dict."""
    conds = []
    for part in [p for p in (spec or "").split(",") if p.strip()]:
        m = COND.match(part)
        if not m:
            raise ValueError(f"bad start filter condition {part!r}")
        neg, key, op, val = m.groups()
        try:
            num = float(val)
        except ValueError:
            num = None
        conds.append((neg == "!", key, op, val, num))

    def ok(tags):
        for neg, key, op, val, num in conds:
            if key not in tags:
                raise KeyError(f"start filter: no tag {key!r} (tags: {', '.join(sorted(tags))})")
            t = tags[key]
            if op == ":":
                r = val in t
            else:
                x, y = (float(t), num) if num is not None and isinstance(t, (int, float)) else (str(t), val)
                r = {">=": x >= y, "<=": x <= y, ">": x > y, "<": x < y, "=": x == y, "!=": x != y}[op]
            if r == neg:
                return False
        return True
    return ok


def read_index(pool_dir):
    return [json.loads(l) for l in open(Path(pool_dir) / "index.jsonl") if l.strip()]


def load_pool(pool_dir, spec="", ante_min=0, ante_max=99, verbose=True, challenge=None):
    """Load the pool's points that pass the filter into a StartPool (one replay per recording)."""
    from .rl_env import StartPool
    pool_dir = Path(pool_dir)
    pred = parse_filter(spec)
    pts = [p for p in read_index(pool_dir) if ante_min <= p["tags"]["ante"] <= ante_max and pred(p["tags"])]
    pool = StartPool()
    by_rec = {}
    for p in pts:
        by_rec.setdefault(p["rec"], []).append(p)
    out = []
    n_other = 0
    for rec, ps in sorted(by_rec.items()):
        script = (pool_dir / "recordings" / rec).read_text()
        ids = pool.add(script, [p["k"] for p in ps])
        for p, sid in zip(ps, ids):
            if sid < 0:
                if verbose:
                    print(f"start pool: {rec} action {p['k']} not loadable: {pool.error()}", flush=True)
                continue
            ch = pool.challenge(sid)
            if challenge is not None and ch != challenge:
                n_other += 1
                continue
            out.append({**p, "sid": int(sid), "challenge": ch})
    if verbose:
        print(f"start pool {pool_dir}: {len(out)} of {len(read_index(pool_dir))} points pass "
              f"(ante {ante_min}..{ante_max}{', ' + spec if spec else ''}"
              f"{f', challenge {challenge or None}: {n_other} of another dropped' if challenge is not None else ''}), "
              f"from {len(by_rec)} recordings", flush=True)
    return pool, out
