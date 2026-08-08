"""Chunked E-step accumulators for CPU memory scalability.

The AMICA E-step materializes (n_comp, n_samples) tensors for y, g,
responsibilities, u*fp etc. On real EEG this exceeds RAM on CPU nodes.

Every quantity the M-step needs is a *sample sum* along the time axis
(verified in the audit trace). Splitting the time axis into chunks,
accumulating partial sums, and dividing by the total sample count once
at the end is an algebraic identity — identical fixed point to full-batch
within O(eps*T) float64 rounding.

This module provides the chunk-level accumulator. The outer loop in
solver.py sums the accumulators across chunks and hands the totals to
the M-step.
"""

from __future__ import annotations

from typing import NamedTuple

from .backend import jax, jnp
from .likelihood import compute_log_det_W
from .pdf import compute_responsibilities_with_loglik


class ChunkStats(NamedTuple):
    """Sufficient statistics accumulated for one time-chunk.

    All fields are sums (not means) — the outer loop divides by the
    total n_samples AFTER summing all chunk contributions.

    Shapes reference n_comp = number of components, n_mix = num_mix_comps.
    """

    gy_partial: jnp.ndarray  # (n_comp, n_comp)  = g_chunk @ y_chunk.T
    sigma2_partial: jnp.ndarray  # (n_comp,)         = sum(y^2, axis=time)
    data_sum: jnp.ndarray  # (n_comp,)         = sum(data_white_chunk, axis=time) [for c update]

    resp_sum: jnp.ndarray  # (n_mix, n_comp)   = sum(resp, axis=time) [for alpha]

    mu_numer: jnp.ndarray  # (n_mix, n_comp)   = sum(u*fp)
    mu_denom_le2: jnp.ndarray  # (n_mix, n_comp)   = sum(u*fp / y_scaled)    [for rho <= 2.0]
    mu_denom_gt2: jnp.ndarray  # (n_mix, n_comp)   = sum(u*fp * fp)          [for rho > 2.0]

    beta_denom_le2: jnp.ndarray  # (n_mix, n_comp)   = sum(u*fp * y_scaled)    [for rho <= 2.0]
    beta_denom_gt2: jnp.ndarray  # (n_mix, n_comp)   = sum(u * |y_scaled|^rho) [for rho > 2.0]

    rho_numer: jnp.ndarray  # (n_mix, n_comp)   = sum(u * |y|^rho * rho*log|y|)

    kappa_numer: jnp.ndarray  # (n_mix, n_comp)   = sum(u * fp^2)
    lambda_numer: jnp.ndarray  # (n_mix, n_comp)   = sum(u * (fp*y_scaled - 1)^2)

    ll_sum: jnp.ndarray  # scalar            = sum of per-sample source_ll
    n_chunk: jnp.ndarray  # scalar            = y_chunk.shape[1]


@jax.jit
def _chunk_stats_one_component(i, y_chunk, alpha, mu, beta, rho, sample_weight=None):
    """Compute per-component partial stats for one chunk.

    The outer caller vmaps this over components. ``sample_weight`` (an optional
    ``(n_chunk,)`` 0/1 vector) implements likelihood-based sample rejection by
    zero-weighting rejected samples in every M-step sum; it is ``None`` on the
    validated no-rejection path, so that branch is removed at trace time and the
    resulting graph is byte-identical to the original.

    Parameters
    ----------
    i : int or jnp.ndarray
        Component index.
    y_chunk : jnp.ndarray, shape (n_comp, n_chunk)
        Source activations for the current chunk.
    alpha : jnp.ndarray, shape (n_mix, n_comp)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_comp)
        Mixture centers.
    beta : jnp.ndarray, shape (n_mix, n_comp)
        Mixture scales.
    rho : jnp.ndarray, shape (n_mix, n_comp)
        Mixture shape parameters.

    Returns
    -------
    stats : tuple
        A tuple of 9 arrays, each with shape (n_mix,), representing the
        accumulated sufficient statistics for component i.
    """
    y_i = y_chunk[i]  # (n_chunk,)
    alpha_i = alpha[:, i]  # (n_mix,)
    mu_i = mu[:, i]
    beta_i = beta[:, i]
    rho_i = rho[:, i]

    # Single fused pass: responsibilities (n_mix, n_chunk) AND the per-sample
    # source log-likelihood (n_chunk,). The score g and the LL are derived from
    # these same quantities below, so this component is touched exactly once per
    # chunk per iteration (was: 1 pass here + 1 in compute_all_scores + 1 in
    # compute_source_loglikelihood).
    resp, source_ll_i = compute_responsibilities_with_loglik(y_i, alpha_i, mu_i, beta_i, rho_i)

    n_mix = alpha_i.shape[0]

    def per_mix(j):
        # Weighting the responsibility once propagates the good-mask to every
        # downstream M-step numerator/denominator (all derive from u).
        u = resp[j] if sample_weight is None else resp[j] * sample_weight  # (n_chunk,)
        m = mu_i[j]
        b = beta_i[j]
        r = rho_i[j]

        y_scaled = b * (y_i - m)  # (n_chunk,)
        abs_y = jnp.abs(y_scaled)
        sign_y = jnp.where(y_scaled >= 0.0, 1.0, -1.0)

        # Single-pass generalized-Gaussian core. The old code evaluated the shape
        # power/exponential ~3x per mixture: pow(|y|, rho-1) for fp, pow(|y|, rho) for
        # beta_denom_gt2, and exp(rho*log|y|) for the rho numerator. XLA does not CSE these
        # (they are distinct HLO with different floors — verified via optimized-HLO op
        # counts, benchmarks/profile_cpu.py), and the NumPy fallback runs each for real.
        # Evaluate log|y| once, get |y|^(rho-1) with a single exp, and |y|^rho as an exact
        # product |y|^(rho-1)*|y| — no extra transcendental. Every accumulator below reuses
        # these, so per (component, mixture) it is 1 log + 1 exp, down from ~1 log + 1 exp +
        # 2 pow. Reductions keep their original form → algebraic identity (ulp-level).
        safe_abs = jnp.maximum(abs_y, 1e-300)
        log_abs = jnp.log(safe_abs)
        p1 = jnp.exp((r - 1.0) * log_abs)  # |y|^(rho-1)
        fp = r * sign_y * p1
        tmpy = p1 * abs_y  # |y|^rho, exact (no transcendental)

        ufp = u * fp

        # Score contribution of this mixture to component i:
        #   g_i = Σ_j β_j · u_j · fp_j  (matches compute_weighted_score exactly)
        g_contrib = b * ufp  # (n_chunk,)

        # Newton kappa; mu_denom_gt2 is exactly beta * kappa (same sum) — compute once.
        kappa_n = jnp.sum(ufp * fp)

        # mu numer/denom
        mu_n = jnp.sum(ufp)
        safe_y = jnp.where(jnp.abs(y_scaled) < 1e-12, 1e-12, y_scaled)
        mu_d_le2 = b * jnp.sum(ufp / safe_y)
        mu_d_gt2 = b * kappa_n

        # beta numer/denom
        u_sum = jnp.sum(u)
        beta_d_le2 = jnp.sum(ufp * y_scaled)
        beta_d_gt2 = jnp.sum(u * tmpy)

        # rho numer (denom is u_sum)
        logab = r * log_abs
        rho_n = jnp.sum(u * tmpy * logab)

        # Newton lambda accumulator
        lambda_tmp = fp * y_scaled - 1.0
        lambda_n = jnp.sum(u * lambda_tmp * lambda_tmp)

        return (
            u_sum,
            mu_n,
            mu_d_le2,
            mu_d_gt2,
            beta_d_le2,
            beta_d_gt2,
            rho_n,
            kappa_n,
            lambda_n,
            g_contrib,
        )

    outs = jax.vmap(per_mix)(jnp.arange(n_mix))
    stats9 = outs[:9]  # tuple of 9 arrays, each (n_mix,)
    g_i = jnp.sum(outs[9], axis=0)  # (n_chunk,) score for component i
    return stats9, g_i, source_ll_i


@jax.jit
def compute_chunk_stats(
    data_chunk: jnp.ndarray,
    W: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    log_det_sphere: float,
    sample_weight: jnp.ndarray | None = None,
) -> ChunkStats:
    """Compute all sufficient statistics for one time-chunk.

    Parameters
    ----------
    data_chunk : jnp.ndarray, shape (n_comp, n_chunk)
        The pre-centered chunk slice of data (data_white - c).
    W : jnp.ndarray, shape (n_comp, n_comp)
        Unmixing matrix.
    alpha : jnp.ndarray, shape (n_mix, n_comp)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_comp)
        Mixture centers.
    beta : jnp.ndarray, shape (n_mix, n_comp)
        Mixture scales.
    rho : jnp.ndarray, shape (n_mix, n_comp)
        Mixture shape parameters.
    log_det_sphere : float
        Log determinant of the sphering matrix, added to per-sample LL.

    Returns
    -------
    stats : ChunkStats
        Sufficient statistics with all partial sums (not divided by n).
    """
    n_comp = W.shape[0]
    n_chunk = data_chunk.shape[1]

    # Sources
    y = jnp.dot(W, data_chunk)  # (n_comp, n_chunk)

    # Per-sample good-mask weighting for sample rejection (M=1). sample_weight is
    # None on the validated no-rejection path → the None branch is taken at trace
    # time, so that graph is byte-identical to the original.
    if sample_weight is None:
        # sigma2 partial (sum of y^2 over time)
        sigma2_partial = jnp.sum(y * y, axis=1)  # (n_comp,)
        # data sum for c update (placeholder; true data_white sum tracked in solver.py)
        data_sum = jnp.sum(data_chunk, axis=1)
    else:
        sigma2_partial = jnp.sum(sample_weight * y * y, axis=1)  # (n_comp,)
        data_sum = jnp.sum(sample_weight * data_chunk, axis=1)

    # Single fused E-step pass: per-component (M-step stats, score g_i, source LL).
    # The score and the source log-likelihood are derived from the SAME
    # responsibilities used for the M-step stats, so the generalized-Gaussian
    # log-pdf is evaluated once per chunk (previously: compute_all_scores +
    # compute_source_loglikelihood each re-evaluated it).
    stats9, g, source_ll_per_comp = jax.vmap(
        lambda i: _chunk_stats_one_component(i, y, alpha, mu, beta, rho, sample_weight)
    )(jnp.arange(n_comp))
    # stats9: tuple of 9 arrays each (n_comp, n_mix) -> transpose to (n_mix, n_comp)
    (u_sum, mu_n, mu_d_le2, mu_d_gt2, beta_d_le2, beta_d_gt2, rho_n, kappa_n, lambda_n) = [
        a.T for a in stats9
    ]
    # g: (n_comp, n_chunk) score; source_ll_per_comp: (n_comp, n_chunk)

    # Natural-gradient numerator (sum over time — NOT mean yet)
    gy_partial = jnp.dot(g, y.T)  # (n_comp, n_comp)

    # Per-sample log-likelihood sum (sum source LL across components)
    source_ll = jnp.sum(source_ll_per_comp, axis=0)  # (n_chunk,)
    log_det_W = compute_log_det_W(W)
    ll_per_sample = source_ll + log_det_W + log_det_sphere
    if sample_weight is None:
        ll_sum = jnp.sum(ll_per_sample)
        n_eff = jnp.asarray(n_chunk, dtype=jnp.float64)
    else:
        ll_sum = jnp.sum(sample_weight * ll_per_sample)
        n_eff = jnp.sum(sample_weight).astype(jnp.float64)

    return ChunkStats(
        gy_partial=gy_partial,
        sigma2_partial=sigma2_partial,
        data_sum=data_sum,
        resp_sum=u_sum,
        mu_numer=mu_n,
        mu_denom_le2=mu_d_le2,
        mu_denom_gt2=mu_d_gt2,
        beta_denom_le2=beta_d_le2,
        beta_denom_gt2=beta_d_gt2,
        rho_numer=rho_n,
        kappa_numer=kappa_n,
        lambda_numer=lambda_n,
        ll_sum=ll_sum,
        n_chunk=n_eff,
    )


def zero_stats(n_comp: int, n_mix: int, dtype=jnp.float64) -> ChunkStats:
    """Zero-initialized accumulator matching the ChunkStats shapes.

    Parameters
    ----------
    n_comp : int
        Number of components.
    n_mix : int
        Number of mixture components.
    dtype : jnp.dtype, optional
        Data type for the arrays. Default is jnp.float64.

    Returns
    -------
    stats : ChunkStats
        Zero-initialized chunk statistics.
    """
    z_cc = jnp.zeros((n_comp, n_comp), dtype=dtype)
    z_c = jnp.zeros((n_comp,), dtype=dtype)
    z_mc = jnp.zeros((n_mix, n_comp), dtype=dtype)
    z_s = jnp.asarray(0.0, dtype=dtype)
    return ChunkStats(
        gy_partial=z_cc,
        sigma2_partial=z_c,
        data_sum=z_c,
        resp_sum=z_mc,
        mu_numer=z_mc,
        mu_denom_le2=z_mc,
        mu_denom_gt2=z_mc,
        beta_denom_le2=z_mc,
        beta_denom_gt2=z_mc,
        rho_numer=z_mc,
        kappa_numer=z_mc,
        lambda_numer=z_mc,
        ll_sum=z_s,
        n_chunk=z_s,
    )


def add_stats(a: ChunkStats, b: ChunkStats) -> ChunkStats:
    """Element-wise sum of two ChunkStats (for accumulating across chunks).

    Parameters
    ----------
    a : ChunkStats
        First chunk statistics.
    b : ChunkStats
        Second chunk statistics.

    Returns
    -------
    stats : ChunkStats
        The element-wise sum of `a` and `b`.
    """
    return ChunkStats(*(getattr(a, f) + getattr(b, f) for f in ChunkStats._fields))
