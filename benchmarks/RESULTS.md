# CPU-path optimization — measured results

Host: 8-core x86, `jax 0.11.0` CPU (no CUDA jaxlib), `numpy 2.5.1`, float64.
Branch `perf/cpu-path`. Reproduce with the scripts in this directory.

Changes measured (all algebraic identities; `chunk_size=None` reproduces the exact
pre-change full-batch trace):
- **C4** cache-aware chunking, default `chunk_size="auto"` (JAX-CPU only)
- **C1** single-pass generalized-Gaussian E-step accumulator
- **C2a** hoist per-chunk `log|det W|` QR

## Synthetic Laplacian mixtures (`validate_chunk.py`, n_ch=32, T=65536, 60 iter)

| path | ms/iter | speedup | LL parity vs full-batch |
|------|---------|---------|-------------------------|
| original full-batch (pre-work) | 518 | 1.00x | — |
| full-batch + C1 | 405 | 1.28x | identical |
| **auto default (C1+C4+C2a)** | **243** | **2.13x** | `max|Δ| ~1e-14` |

## Real EEG — PhysioNet EEGBCI subject 1, runs 1-4 (`bench_mne_real.py`)

64-channel motor-imagery EEG, 59,520 samples, 1 Hz high-pass, PCA→32 components, 60 iter
(includes the Newton phase, iters 50-60):

| path | ms/iter | final LL | 
|------|---------|----------|
| full-batch (`chunk_size=None`) | 359 | 9.45332559 |
| **auto (default)** | **222** | 9.45332603 |

- **speedup: 1.62x** (lower than synthetic because the Newton phase does per-iteration
  matrix work that does not chunk; E-step-dominated fits are closer to ~2x).
- **LL parity: `max|Δ| = 4.7e-7` (rel `5e-8`)** over the trajectory — PASS (bar is `1e-4`).
  Larger than the synthetic `1e-14` because reassociated reductions accumulate through the
  Newton update across 60 iterations. Still ~4 orders under the bar.
- peak RSS 1290 MB.

## Kernel microbench (`profile_cpu.py`, n_comp=64, T=262144)

- E-step `compute_chunk_stats`: 3814 ms (pre-C1) -> 2799 ms (post-C1), pow HLO ops 6 -> 0.
- Threading: E-step ~3.4/8 cores; GEMM control ~7.4/8 (⇒ ~2x threading headroom remains).
- Chunk knee: full-batch 3792 ms/pass vs 1024-sample chunk 1864 ms (2.03x); optimum is a
  cache-resident chunk (~1-2 MiB working set), not the largest that fits RAM.

## NumPy fallback (`AMICA_NO_JAX=1`)

- E-step is ~1.7x slower than JAX-CPU (653 vs 375 ms at n_comp=64, chunk=32768), not orders
  of magnitude — the loop-`vmap` Python overhead amortizes over chunk-length NumPy work.
- Cache chunking is intentionally NOT applied here (measured ~5% slower with small chunks,
  no XLA fusion to benefit); the fallback keeps RAM-based sizing.
