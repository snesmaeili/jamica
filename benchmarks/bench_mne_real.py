#!/usr/bin/env python
"""Real-EEG benchmark: cache-aware chunking + single-pass GGD on the JAX-CPU path.

Loads real 64-channel motor-imagery EEG (PhysioNet EEGBCI via MNE), high-pass filters it
as AMICA expects, then fits AMICA twice on identical data — full-batch (chunk_size=None)
vs the new default cache-aware "auto" — and reports steady-state ms/iter, speedup, peak RSS,
and the log-likelihood-trajectory parity between the two.

Real EEG (temporally correlated, non-stationary, genuine mixing) is a stronger test than
synthetic Laplacian mixtures for both the speedup and the algebraic-identity parity claim.
"""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np


def load_real_eeg(subject: int, runs: list[int], hp: float) -> np.ndarray:
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    mne.set_log_level("WARNING")
    paths = eegbci.load_data(subject, runs, update_path=True)
    raw = concatenate_raws([read_raw_edf(p, preload=True) for p in paths])
    eegbci.standardize(raw)  # fix channel names for a standard montage
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.pick("eeg")
    raw.filter(l_freq=hp, h_freq=None)  # high-pass; AMICA expects HP-filtered data
    data = raw.get_data().astype(np.float64)  # (n_channels, n_samples), Volts
    return data


def run(data, chunk_size, pcakeep, max_iter, seed=42):
    from amica import Amica, AmicaConfig

    cfg = AmicaConfig(
        max_iter=max_iter, num_mix_comps=3, pcakeep=pcakeep, chunk_size=chunk_size
    )
    t0 = time.perf_counter()
    res = Amica(cfg, random_state=seed).fit(data)
    return time.perf_counter() - t0, np.asarray(res.log_likelihood)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--runs", type=str, default="1,2,3,4")
    ap.add_argument("--hp", type=float, default=1.0)
    ap.add_argument("--pcakeep", type=int, default=32)
    ap.add_argument("--max-iter", type=int, default=60)
    args = ap.parse_args()

    runs = [int(r) for r in args.runs.split(",")]
    data = load_real_eeg(args.subject, runs, args.hp)
    from amica.backend import HAS_JAX
    from amica.solver import _choose_chunk_size

    auto_chunk = _choose_chunk_size(data.shape[1], args.pcakeep, 3, dtype=np.dtype(np.float64))
    print(f"real EEG: {data.shape[0]} ch x {data.shape[1]} samples, HP={args.hp} Hz")
    print(f"HAS_JAX={HAS_JAX}  pcakeep={args.pcakeep}  max_iter={args.max_iter}  "
          f"auto chunk -> {auto_chunk} ({'chunks' if auto_chunk < data.shape[1] else 'full-batch'})\n")

    wf, ll_f = run(data, None, args.pcakeep, args.max_iter)
    print(f"full-batch : {wf:7.2f} s  ({1e3*wf/args.max_iter:7.1f} ms/iter)  "
          f"final LL={ll_f[-1]:.8f}  n_iter={len(ll_f)}")

    wa, ll_a = run(data, "auto", args.pcakeep, args.max_iter)
    print(f"auto       : {wa:7.2f} s  ({1e3*wa/args.max_iter:7.1f} ms/iter)  "
          f"final LL={ll_a[-1]:.8f}  n_iter={len(ll_a)}")

    n = min(len(ll_f), len(ll_a))
    max_abs = float(np.max(np.abs(ll_f[:n] - ll_a[:n])))
    rel = max_abs / (float(np.max(np.abs(ll_f[:n]))) or 1.0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\nspeedup    : {wf / wa:.2f}x")
    print(f"LL parity  : max|Δ| over trajectory = {max_abs:.3e}  (rel {rel:.3e})")
    print(f"peak RSS   : {rss:.0f} MB")
    print(f"parity {'PASS' if rel < 1e-4 else 'FAIL'} (rel < 1e-4)")


if __name__ == "__main__":
    main()
