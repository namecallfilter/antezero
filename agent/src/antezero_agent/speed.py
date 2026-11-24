"""Speed-ups that keep results the same up to float rounding (config: len_buckets / len_round, compile, pinned_gather,
fast_opt)."""
import os
import shutil
import sys
import sysconfig
import time
import warnings
from pathlib import Path

import torch


def bucket_plan(hist, k, rnd, cap, min_rows=2):
    """hist[l]: rows with l real tokens (l = 0..len(hist)-1). Returns [(rows, L), ...]: <= k buckets of the rows sorted
    by length (ascending), bucket b taking the next rows[b] of them, padded to L (its longest row rounded up to a
    multiple of rnd, at most cap)."""
    vals, cnts = [], []
    for ln, c in enumerate(hist):
        if not c:
            continue
        g = min(cap, -(-ln // rnd) * rnd) if ln else min(cap, rnd)
        if vals and vals[-1] == g:
            cnts[-1] += int(c)
        else:
            vals.append(g)
            cnts.append(int(c))
    n = len(vals)
    if n == 0:
        return []
    cum = [0]
    for c in cnts:
        cum.append(cum[-1] + c)
    tot = cum[-1]
    if k <= 1 or n == 1 or tot < 2 * min_rows:
        return [(tot, vals[-1])]
    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    arg = [[-1] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0
    for b in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(j):
                if best[b - 1][i] == INF or cum[j] - cum[i] < min_rows:
                    continue
                c = best[b - 1][i] + (cum[j] - cum[i]) * vals[j - 1]
                if c < best[b][j]:
                    best[b][j], arg[b][j] = c, i
    nb = min(range(1, k + 1), key=lambda b: (best[b][n], b))
    if best[nb][n] == INF:
        return [(tot, vals[-1])]
    out, j = [], n
    for b in range(nb, 0, -1):
        i = arg[b][j]
        out.append((cum[j] - cum[i], vals[j - 1]))
        j = i
    return out[::-1]


_STATE = {"off": None}
SITES = []


def toolchain_problem():
    """None, or why torch.compile (inductor / Triton) can't build its kernels here."""
    inc = sysconfig.get_paths().get("include") or ""
    dirs = [inc] + [d for v in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH") for d in os.environ.get(v, "").split(os.pathsep) if d]
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if not any((Path(d) / "Python.h").exists() for d in dirs):
        return (f"Python.h not found (looked in {', '.join(dirs)}). torch.compile's Triton / inductor kernels need the "
                f"Python {ver} dev headers: sudo apt install python{ver}-dev")
    cc = os.environ.get("CC") or "cc"
    if not (shutil.which(cc) or shutil.which("gcc") or shutil.which("clang")):
        return f"no C compiler found ({cc}, gcc, clang): torch.compile needs one (sudo apt install build-essential)"
    return None


def configure():
    """Global settings for compile=1."""
    import torch._dynamo.config as dc
    import torch._inductor.config as ic
    ic.fallback_random = True
    for k_ in ("recompile_limit", "cache_size_limit"):
        if hasattr(dc, k_):
            setattr(dc, k_, max(64, getattr(dc, k_)))
    try:
        import torch._inductor.cpp_builder as cb
        _orig = cb._get_python_include_dirs
        def _patched():
            dirs = list(_orig())
            for env_var in ("C_INCLUDE_PATH", "CPATH"):
                for d in os.environ.get(env_var, "").split(os.pathsep):
                    if d and os.path.exists(os.path.join(d, "Python.h")) and d not in dirs:
                        dirs.insert(0, d)
            return dirs
        cb._get_python_include_dirs = _patched
    except (ImportError, AttributeError):
        pass


def _snap():
    from torch._dynamo.utils import counters
    return (counters["frames"]["total"], counters["stats"]["unique_graphs"],
            sum(counters["graph_break"].values()))


def disable(reason):
    if _STATE["off"] is None:
        _STATE["off"] = reason
        hint = toolchain_problem()
        warnings.warn(f"torch.compile failed, running eager from here on (identical results, slower): {reason}"
                      + (f"\n  likely cause: {hint}" if hint else ""), stacklevel=3)
        print(f"WARNING: compile=1: compile failed, every site falls back to eager: {reason}"
              + (f"\n  likely cause: {hint}" if hint else ""), flush=True)


def is_off():
    return _STATE["off"] is not None


COMPILE_MODE = {"mode": "default"}


class CompiledSite:
    """fn compiled with torch.compile (default mode, dynamic shapes) behind a site name."""

    def __init__(self, name, fn, enabled=True):
        self.name, self.fn = name, fn
        mode = None if COMPILE_MODE["mode"] == "default" else COMPILE_MODE["mode"]
        self.cfn = torch.compile(fn, dynamic=True, mode=mode) if enabled else None
        self.calls = self.frames = self.graphs = self.breaks = self.recompiles = 0
        self.t_compile = 0.0
        if enabled:
            SITES.append(self)

    def __call__(self, *a, **kw):
        if self.cfn is None or is_off():
            return self.fn(*a, **kw)
        s0, t0 = _snap(), time.time()
        try:
            out = self.cfn(*a, **kw)
        except Exception as ex:   # noqa: BLE001
            disable(f"{self.name}: {type(ex).__name__}: {str(ex).strip().splitlines()[0][:300] if str(ex).strip() else ''}")
            return self.fn(*a, **kw)
        s1 = _snap()
        df = s1[0] - s0[0]
        if df:
            self.t_compile += time.time() - t0
            if self.calls == 0:
                self.frames += df
            else:
                self.recompiles += df
        self.graphs += s1[1] - s0[1]
        self.breaks += s1[2] - s0[2]
        self.calls += 1
        return out

    def summary(self):
        if self.cfn is None:
            return f"{self.name}: eager"
        if is_off():
            return f"{self.name}: eager (compile failed)"
        ok = "compiled" if self.graphs or self.frames else "no frame of its own (cached elsewhere, or eager)"
        return (f"{self.name}: {ok}, {self.calls} calls, {self.frames} frames + {self.recompiles} recompiles, "
                f"{self.graphs} graphs, {self.breaks} graph breaks, {self.t_compile:.1f} s in compiling calls")


def summary():
    if not SITES:
        return "compile: off"
    if is_off():
        return f"compile: FAILED, eager everywhere ({_STATE['off']})"
    return "compile: " + "; ".join(s.summary() for s in SITES)


class PinnedGather:
    """x = src[j] on dev for a host-side src (pinned, buf_device=cpu)."""

    def __init__(self, dev):
        self.dev, self.buf, self.ev = dev, None, None

    def __call__(self, src, j):
        jc = j.cpu()
        n = len(jc)
        if self.buf is None or self.buf.shape[0] < n or self.buf.shape[1:] != src.shape[1:] or self.buf.dtype != src.dtype:
            if self.ev is not None:
                self.ev.synchronize()
            self.buf = torch.empty((n, *src.shape[1:]), dtype=src.dtype, pin_memory=True)
        elif self.ev is not None:
            self.ev.synchronize()
        out = self.buf[:n]
        torch.index_select(src, 0, jc, out=out)
        xb = out.to(self.dev, non_blocking=True)
        self.ev = torch.cuda.Event()
        self.ev.record()
        return xb


class BatchedMuon(torch.optim.Muon):
    """torch.optim.Muon's update, with the matrices of one shape stepped together."""

    @torch.no_grad()
    def step(self, closure=None):
        from torch.optim._muon import _adjust_lr
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params, grads, bufs = [], [], []
            self._init_group(group, params, grads, bufs)
            if not params:
                continue
            lr, wd, mom = float(group["lr"]), group["weight_decay"], group["momentum"]
            a, b, c = group["ns_coefficients"]
            torch._foreach_lerp_(bufs, grads, 1 - mom)
            ups = torch._foreach_lerp(grads, bufs, mom) if group["nesterov"] else bufs
            by_shape = {}
            for i, p in enumerate(params):
                by_shape.setdefault(tuple(p.shape), []).append(i)
            for shape, idx in by_shape.items():
                G = torch.stack([ups[i] for i in idx]).bfloat16()
                tr = shape[0] > shape[1]
                if tr:
                    G = G.transpose(1, 2)
                G = G / G.norm(dim=(1, 2), keepdim=True).clamp(min=group["eps"])
                for _ in range(group["ns_steps"]):
                    gram = torch.bmm(G, G.transpose(1, 2))
                    upd = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                    G = torch.baddbmm(G, upd, G, beta=a)
                if tr:
                    G = G.transpose(1, 2)
                ps = [params[i] for i in idx]
                if wd:
                    torch._foreach_mul_(ps, 1 - lr * wd)
                torch._foreach_add_(ps, list(G.to(ps[0].dtype).unbind(0)),
                                    alpha=-_adjust_lr(lr, group["adjust_lr_fn"], torch.Size(shape)))
        return loss
