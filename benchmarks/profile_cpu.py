#!/usr/bin/env python
"""Step-0 CPU-path profiling for the AMICA E-step (commit 450a63c, branch perf/cpu-path).

Answers the three questions the review panel's estimates hinged on:

  A. THREADING   — does the fused generalized-Gaussian E-step actually use all cores on
                   XLA-CPU, or does it run the elementwise/reduction fusion single-threaded?
                   (claude F2 — the highest-ceiling item, and the only one a reviewer flagged.)
                   Measured as CPU-time / wall-time over many calls of the real jitted
                   `compute_chunk_stats`, with a same-size GEMM as a positive control.
  B. HLO         — does XLA already CSE the duplicated transcendentals (|y|^rho computed
                   ~4x in per_mix) and hoist the per-chunk QR of W? If yes, the single-pass
                   GGD rewrite helps mainly the NumPy fallback; if no, it helps JAX-CPU too.
                   Measured by counting ops in the *optimized* HLO of the jitted kernel.
  C. CHUNK SIZE  — is "largest chunk that fits RAM" the speed optimum, or is there a
                   cache-resident sweet spot? (C4 / codex #1.) Steady-state ms per full
                   E-step pass over the whole recording, as a function of chunk size.

This is a KERNEL microbenchmark: it drives the exact jitted function the solver calls on
the hot path, with synthetic Laplacian-mixture-scale inputs. Values are chosen O(1) so the
math stays in-regime; compute cost of exp/log/pow is data-independent, so cost numbers are
representative. It does NOT depend on any real EEG dataset.

Run:  .venv/bin/python benchmarks/profile_cpu.py
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np


def _fmt(x: float) -> str:
    return f"{x:8.2f}"


def synth_inputs(n_comp: int, n_mix: int, T: int, seed: int = 0):
    """Synthetic pre-centered whitened data + in-regime mixture params (float64)."""
    rng = np.random.default_rng(seed)
    # Laplacian sources are exactly AMICA's generative model; whitened data ~ O(1).
    data_white = rng.laplace(size=(n_comp, T)).astype(np.float64)
    data_white -= data_white.mean(axis=1, keepdims=True)  # pre-centered (data - c)
    # A near-identity, well-conditioned unmixing matrix.
    W = np.eye(n_comp) + 0.01 * rng.standard_normal((n_comp, n_comp))
    W = W.astype(np.float64)
    alpha = np.full((n_mix, n_comp), 1.0 / n_mix, dtype=np.float64)
    mu = 0.1 * rng.standard_normal((n_mix, n_comp)).astype(np.float64)
    beta = np.ones((n_mix, n_comp), dtype=np.float64)
    rho = np.full((n_mix, n_comp), 1.5, dtype=np.float64)  # in (minrho=1, maxrho=2)
    log_det_sphere = 0.0
    return data_white, W, alpha, mu, beta, rho, log_det_sphere


def time_calls(fn, args, n_warmup: int, n_iter: int):
    """Return (ms_per_call, cores_used) for a JAX fn, blocking each call."""
    import jax

    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n_iter):
        jax.block_until_ready(fn(*args))
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    ms = 1e3 * wall / n_iter
    cores = cpu / wall if wall > 0 else float("nan")
    return ms, cores


def phase_A_threading(inputs, n_comp, n_mix, T):
    import jax
    import jax.numpy as jnp

    from amica.accumulators import compute_chunk_stats

    data_white, W, alpha, mu, beta, rho, lds = inputs
    args = tuple(
        jnp.asarray(a) if isinstance(a, np.ndarray) else a
        for a in (data_white, W, alpha, mu, beta, rho, lds)
    )
    estep_ms, estep_cores = time_calls(lambda *a: compute_chunk_stats(*a), args, 2, 10)

    # Positive control: a GEMM of comparable FLOPs, which XLA-CPU is known to thread.
    a = jnp.asarray(np.random.default_rng(1).standard_normal((2048, 2048)))
    b = jnp.asarray(np.random.default_rng(2).standard_normal((2048, 2048)))
    gemm = jax.jit(lambda x, y: x @ y)
    gemm_ms, gemm_cores = time_calls(gemm, (a, b), 2, 10)

    return {
        "estep_ms_per_call": estep_ms,
        "estep_cores_used": estep_cores,
        "gemm_ms_per_call": gemm_ms,
        "gemm_cores_used": gemm_cores,
        "n_cores_available": len(os.sched_getaffinity(0)),
    }


def phase_B_hlo(inputs):
    import jax
    import jax.numpy as jnp

    from amica.accumulators import compute_chunk_stats

    data_white, W, alpha, mu, beta, rho, lds = inputs
    args = tuple(
        jnp.asarray(a) if isinstance(a, np.ndarray) else a
        for a in (data_white, W, alpha, mu, beta, rho, lds)
    )
    compiled = compute_chunk_stats.lower(*args).compile()
    hlo = compiled.as_text().lower()

    def count(*needles):
        return sum(hlo.count(n) for n in needles)

    return {
        "exp_ops": count("exponential"),
        "log_ops": count("log("),          # 'log(' avoids matching 'logistic' etc.
        "pow_ops": count("power"),
        "rsqrt_ops": count("rsqrt"),
        # QR / cholesky / triangular-solve custom calls used by log|det W|:
        "qr_or_lu_calls": count("geqrf", "qr", "cholesky", "lu_", "custom-call"),
        "hlo_line_count": compiled.as_text().count("\n"),
    }


def phase_C_chunk_sweep(inputs, n_comp, n_mix, T, chunk_sizes):
    import jax
    import jax.numpy as jnp

    from amica.accumulators import add_stats, compute_chunk_stats, zero_stats

    data_white, W, alpha, mu, beta, rho, lds = inputs
    Wj = jnp.asarray(W)
    aj, mj, bj, rj = map(jnp.asarray, (alpha, mu, beta, rho))
    dwj = jnp.asarray(data_white)

    results = []
    for B in chunk_sizes:
        Beff = min(B, T)

        def one_full_pass():
            totals = zero_stats(n_comp, n_mix, dtype=jnp.float64)
            for start in range(0, T, Beff):
                stop = min(start + Beff, T)
                stats = compute_chunk_stats(
                    dwj[:, start:stop], Wj, aj, mj, bj, rj, lds
                )
                totals = add_stats(totals, stats)
            return totals

        # warmup compiles both the full-chunk shape and the ragged-tail shape
        jax.block_until_ready(one_full_pass())
        w0, c0 = time.perf_counter(), time.process_time()
        n_rep = 5
        for _ in range(n_rep):
            jax.block_until_ready(one_full_pass())
        wall = time.perf_counter() - w0
        cpu = time.process_time() - c0
        results.append(
            {
                "chunk_size": Beff,
                "n_chunks": (T + Beff - 1) // Beff,
                "ms_per_full_pass": 1e3 * wall / n_rep,
                "cores_used": cpu / wall if wall > 0 else float("nan"),
            }
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-comp", type=int, default=64)
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--T", type=int, default=262144)
    ap.add_argument("--out", type=str, default="benchmarks/profile_cpu_result.json")
    ap.add_argument(
        "--chunks",
        type=str,
        default="",
        help="comma-separated chunk sizes for phase C (default: 8192,32768,131072,524288,T)",
    )
    args = ap.parse_args()

    import jax

    env = {
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "(unset)"),
        "n_comp": args.n_comp,
        "n_mix": args.n_mix,
        "T": args.T,
    }
    print("=== environment ===")
    print(json.dumps(env, indent=2))

    inputs = synth_inputs(args.n_comp, args.n_mix, args.T)

    print("\n=== PHASE A: threading (cores used; ~1 => single-threaded fusion) ===")
    A = phase_A_threading(inputs, args.n_comp, args.n_mix, args.T)
    print(
        f"  E-step  compute_chunk_stats : {_fmt(A['estep_ms_per_call'])} ms/call   "
        f"cores≈{A['estep_cores_used']:.2f}"
    )
    print(
        f"  GEMM    2048x2048 (control) : {_fmt(A['gemm_ms_per_call'])} ms/call   "
        f"cores≈{A['gemm_cores_used']:.2f}"
    )
    print(f"  cores available: {A['n_cores_available']}")

    print("\n=== PHASE B: optimized-HLO op counts (does XLA CSE / hoist?) ===")
    B = phase_B_hlo(inputs)
    for k, v in B.items():
        print(f"  {k:20s}: {v}")

    print("\n=== PHASE C: chunk-size sweep (ms per full E-step pass over all T) ===")
    if args.chunks:
        chunk_sizes = [int(x) for x in args.chunks.split(",")]
    else:
        chunk_sizes = [8192, 32768, 131072, 524288, args.T]
    C = phase_C_chunk_sweep(inputs, args.n_comp, args.n_mix, args.T, chunk_sizes)
    print(f"  {'chunk':>10} {'n_chunks':>9} {'ms/pass':>10} {'cores':>7}")
    for r in C:
        print(
            f"  {r['chunk_size']:>10} {r['n_chunks']:>9} "
            f"{r['ms_per_full_pass']:>10.2f} {r['cores_used']:>7.2f}"
        )

    maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\n  peak RSS: {maxrss_mb:.0f} MB")

    report = {"env": env, "phase_A": A, "phase_B": B, "phase_C": C, "peak_rss_mb": maxrss_mb}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
