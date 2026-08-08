#!/usr/bin/env python
"""End-to-end validation of the cache-aware-chunk hypothesis on the REAL solver.

Runs a full `Amica.fit` twice on identical synthetic data — full-batch (chunk_size=None)
vs a small cache-resident chunk — and reports:
  * total fit wall time and approx ms/iter (the speedup a user actually sees), and
  * the log-likelihood trajectory difference (parity: the chunked path is an algebraic
    identity, so trajectories must agree to ~float64 reassociation noise, not diverge).

This grounds finding C4 on the solver's own code path, not just the kernel microbench.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from amica import Amica, AmicaConfig


def make_data(n_ch: int, T: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S = rng.laplace(size=(n_ch, T))          # Laplacian sources (AMICA's model)
    A = rng.standard_normal((n_ch, n_ch))    # random mixing
    return (A @ S).astype(np.float64)


def run(data: np.ndarray, chunk_size, max_iter: int, seed: int = 42):
    cfg = AmicaConfig(max_iter=max_iter, num_mix_comps=3, chunk_size=chunk_size)
    model = Amica(cfg, random_state=seed)
    t0 = time.perf_counter()
    res = model.fit(data)
    wall = time.perf_counter() - t0
    return wall, np.asarray(res.log_likelihood)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ch", type=int, default=32)
    ap.add_argument("--T", type=int, default=65536)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    data = make_data(args.n_ch, args.T)
    print(f"data: ({args.n_ch}, {args.T}), max_iter={args.max_iter}, chunk={args.chunk}\n")

    wall_full, ll_full = run(data, None, args.max_iter)
    print(f"full-batch : {wall_full:7.2f} s  ({1e3*wall_full/args.max_iter:6.1f} ms/iter)  "
          f"final LL={ll_full[-1]:.8f}  n_iter={len(ll_full)}")

    wall_chunk, ll_chunk = run(data, args.chunk, args.max_iter)
    print(f"chunk={args.chunk:<6}: {wall_chunk:7.2f} s  ({1e3*wall_chunk/args.max_iter:6.1f} ms/iter)  "
          f"final LL={ll_chunk[-1]:.8f}  n_iter={len(ll_chunk)}")

    n = min(len(ll_full), len(ll_chunk))
    max_abs = float(np.max(np.abs(ll_full[:n] - ll_chunk[:n])))
    denom = float(np.max(np.abs(ll_full[:n]))) or 1.0
    print(f"\nspeedup   : {wall_full / wall_chunk:.2f}x")
    print(f"LL parity : max|Δ| over trajectory = {max_abs:.3e}  (rel {max_abs/denom:.3e})")
    print(f"final ΔLL : {abs(ll_full[-1] - ll_chunk[-1]):.3e}")
    ok = max_abs / denom < 1e-4
    print(f"parity {'PASS' if ok else 'FAIL'} (rel < 1e-4, the docstring bar in _amica_step_fused)")


if __name__ == "__main__":
    main()
