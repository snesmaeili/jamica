"""Main AMICA solver class using JAX/NumPy."""

from __future__ import annotations

import logging
import time
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .backend import jax, jnp
from .config import AmicaConfig
from .likelihood import (
    compute_log_det_W,
    compute_total_loglikelihood,
)
from .pdf import (
    compute_all_scores,
    compute_source_loglikelihood,
)
from .preprocessing import (
    preprocess_data,
)
from .updates import (
    apply_full_newton_correction,
    compute_newton_terms,
    update_all_pdf_params,
)

logger = logging.getLogger(__name__)

# Lightweight namedtuple to pass config parameters into the JIT-compiled step.
# Defined at the module level to avoid re-registering a new type on every JIT trace.
ParamConfig = namedtuple(
    "ParamConfig",
    [
        "invsigmin",
        "invsigmax",
        "minrho",
        "maxrho",
        "update_alpha",
        "update_mu",
        "update_beta",
        "update_rho",
        "rholrate",
    ],
)


def _reject_threshold(sample_lls, good_mask, rejsig):
    """Fortran-faithful one-sided likelihood-rejection threshold (amica17 ``reject_data``).

    Computes the mean and (population, ddof=0) std of the per-sample log-likelihood
    over the **current good samples only**, then rejects samples whose LL is below
    ``mean - rejsig*std``. The returned mask is the cumulative-AND of the prior mask
    with the keep test, so a once-rejected sample is **never re-accepted** — matching
    the monotone behavior of Fortran AMICA 1.7 (its ``reject_data`` loop iterates only
    the current good indices and only ever flips ``gooddata`` to false).

    Parameters
    ----------
    sample_lls : np.ndarray, shape (n_samples,)
        Per-sample total log-likelihood, evaluated over all samples.
    good_mask : np.ndarray of bool, shape (n_samples,)
        Current good-sample mask (True = kept).
    rejsig : float
        Rejection threshold expressed in standard deviations below the mean.

    Returns
    -------
    new_mask : np.ndarray of bool
        Updated good mask (a monotone subset of ``good_mask``).
    n_newly_rejected : int
        Number of samples rejected in this round.
    """
    good_lls = sample_lls[good_mask]
    ll_mean = float(good_lls.mean())
    ll_std = float(good_lls.std())  # ddof=0 → sqrt(E[x^2]-E[x]^2), matches Fortran
    threshold = ll_mean - rejsig * ll_std
    new_mask = good_mask & (sample_lls >= threshold)
    n_newly_rejected = int(good_mask.sum() - new_mask.sum())
    return new_mask, n_newly_rejected


@partial(
    jax.jit,
    # Stage 3F: donate the previous-iteration state buffers (W,A,c,alpha,mu,beta,
    # rho — positions 0-6; NOT gm, which is returned unchanged) so XLA reuses them
    # in-place. Safe because the fit loop always accepts the returned (guarded)
    # state, so these inputs are never read again after the call.
    donate_argnums=(0, 1, 2, 3, 4, 5, 6),
    static_argnames=[
        "do_newton",
        "do_mean",
        "do_sphere",
        "doscaling",
        "update_alpha",
        "update_mu",
        "update_beta",
        "update_rho",
    ],
)
def _amica_step(
    # State variables
    W: jnp.ndarray,
    A: jnp.ndarray,
    c: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    gm: jnp.ndarray,
    # Per-iter step size
    lrate_step: float,
    rholrate: float,
    # Static data
    data_white: jnp.ndarray,
    log_det_sphere: float,
    # Config scalars
    newt_start_iter: int,
    iteration: int,
    invsigmin: float,
    invsigmax: float,
    minrho: float,
    maxrho: float,
    # Config flags (static)
    do_newton: bool,
    do_mean: bool,
    do_sphere: bool,
    doscaling: bool,
    update_alpha: bool,
    update_mu: bool,
    update_beta: bool,
    update_rho: bool,
):
    """Single JIT-compiled AMICA iteration step.

    Parameters
    ----------
    W : jnp.ndarray, shape (n_components, n_components)
        Current unmixing matrix.
    A : jnp.ndarray, shape (n_components, n_components)
        Current mixing matrix.
    c : jnp.ndarray, shape (n_components,)
        Current model centers.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Current mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Current mixture locations.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Current mixture inverse scales.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Current mixture shapes.
    gm : jnp.ndarray, shape (n_models,)
        Current model weights (placeholder for future multi-model).
    lrate_step : float
        Learning rate for the natural gradient step.
    rholrate : float
        Learning rate for the shape parameter updates.
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Whitened input data.
    log_det_sphere : float
        Log determinant of the sphering matrix.
    newt_start_iter : int
        Iteration to start applying Newton corrections.
    iteration : int
        Current iteration index.
    invsigmin : float
        Minimum allowed inverse scale parameter.
    invsigmax : float
        Maximum allowed inverse scale parameter.
    minrho : float
        Minimum allowed shape parameter.
    maxrho : float
        Maximum allowed shape parameter.
    do_newton : bool
        Whether to apply Newton correction.
    do_mean : bool
        Whether to center the data via model centers.
    do_sphere : bool
        Whether to use sphered data (affects log-likelihood).
    doscaling : bool
        Whether to re-scale mixing columns to unit norm.
    update_alpha : bool
        Whether to update mixture weights.
    update_mu : bool
        Whether to update mixture locations.
    update_beta : bool
        Whether to update mixture scales.
    update_rho : bool
        Whether to update mixture shapes.

    Returns
    -------
    W_new : jnp.ndarray
        Updated unmixing matrix.
    A_new : jnp.ndarray
        Updated mixing matrix.
    c_new : jnp.ndarray
        Updated model centers.
    alpha_new : jnp.ndarray
        Updated mixture weights.
    mu_new : jnp.ndarray
        Updated mixture locations.
    beta_new : jnp.ndarray
        Updated mixture inverse scales.
    rho_new : jnp.ndarray
        Updated mixture shapes.
    gm_new : jnp.ndarray
        Updated model weights (unchanged for single-model).
    ll : float
        Total log-likelihood of the current state.
    is_good : bool
        Whether the updated parameters are valid (no NaNs).
    newton_used : bool
        Whether the Newton correction was actively applied.
    """

    # 1. E-step: Compute sources
    # y = W * (x - c)
    y = jnp.dot(W, data_white - c[:, None])
    n_samples = y.shape[1]
    n_components = W.shape[0]

    # 2. Compute scores
    g = compute_all_scores(y, alpha, mu, beta, rho)

    # 3. Compute Log-Likelihood
    ll = compute_total_loglikelihood(y, W, alpha, mu, beta, rho, log_det_sphere)

    # 4. Natural Gradient on A
    gy = jnp.dot(g, y.T) / n_samples
    dA_local = jnp.eye(n_components) - gy

    # 5. Newton Correction
    Wtmp = dA_local
    newton_used = jnp.array(False)

    def apply_newton(operands):
        y_, alpha_, mu_, beta_, rho_ = operands
        sigma2, kappa, lambda_ = compute_newton_terms(y_, alpha_, mu_, beta_, rho_)
        lambda_pos = jnp.all(lambda_ > 0)
        Wtmp_newt, posdef_newt = apply_full_newton_correction(dA_local, sigma2, kappa, lambda_)
        is_valid = lambda_pos & posdef_newt
        return jnp.where(is_valid, Wtmp_newt, dA_local), is_valid

    if do_newton:

        def try_newton(operands):
            return apply_newton(operands)

        def skip_newton(operands):
            return dA_local, jnp.array(False)

        Wtmp, newton_used = jax.lax.cond(
            iteration >= newt_start_iter, try_newton, skip_newton, (y, alpha, mu, beta, rho)
        )
    else:
        Wtmp = dA_local
        newton_used = jnp.array(False)

    # 6. Update A using step-local lrate (halved natgrad or ramped Newton)
    dAk = A @ Wtmp
    A_new = A - lrate_step * dAk

    # 7. Update W = inv(A) — exact inverse via LU.
    def invert_A(A_):
        return jnp.linalg.pinv(A_).astype(W.dtype)

    A_ok = jnp.all(jnp.isfinite(A_new))
    W_new = jax.lax.cond(
        A_ok,
        invert_A,
        lambda x: W,  # Fallback to old W if A has NaN/Inf
        A_new,
    )
    # Check BOTH A and W for NaN/Inf
    is_good = A_ok & jnp.all(jnp.isfinite(W_new))

    # 8. M-step: Update PDF parameters
    pconfig = ParamConfig(
        invsigmin,
        invsigmax,
        minrho,
        maxrho,
        update_alpha,
        update_mu,
        update_beta,
        update_rho,
        rholrate,
    )

    alpha_new, mu_new, beta_new, rho_new = update_all_pdf_params(
        y, alpha, mu, beta, rho, pconfig, rholrate
    )

    # 9. Update model center.
    c_new = jnp.mean(data_white, axis=1) if do_mean else c

    # 10. Scaling
    if doscaling:
        col_norms = jnp.linalg.norm(A_new, axis=0)
        col_norms = jnp.where(col_norms > 0.0, col_norms, 1.0)
        A_new = A_new / col_norms
        mu_new = mu_new * col_norms[None, :]
        beta_new = beta_new / col_norms[None, :]
        # Avoid a 2nd pinv: column-scaling A by diag(1/col_norms) row-scales its
        # inverse by diag(col_norms). For the square invertible A maintained here,
        # pinv == inv, so this is exact (up to FP reassociation) and skips an SVD.
        #   inv(A · diag(1/c)) = diag(c) · inv(A)
        W_new = W_new * col_norms[:, None]

    # Stage 3F: guard state outputs so a bad step (is_good False) returns the
    # UNCHANGED input. Bit-exact with the caller's former discard-on-bad path,
    # and required for donate_argnums — the caller then always accepts, so the
    # donated input buffers are never reused on the recovery path.
    W_new = jnp.where(is_good, W_new, W)
    A_new = jnp.where(is_good, A_new, A)
    c_new = jnp.where(is_good, c_new, c)
    alpha_new = jnp.where(is_good, alpha_new, alpha)
    mu_new = jnp.where(is_good, mu_new, mu)
    beta_new = jnp.where(is_good, beta_new, beta)
    rho_new = jnp.where(is_good, rho_new, rho)
    return (W_new, A_new, c_new, alpha_new, mu_new, beta_new, rho_new, gm, ll, is_good, newton_used)


def _amica_step_chunked(
    W,
    A,
    c,
    alpha,
    mu,
    beta,
    rho,
    gm,
    lrate_step,
    rholrate,
    data_white,
    log_det_sphere,
    newt_start_iter,
    iteration,
    invsigmin,
    invsigmax,
    minrho,
    maxrho,
    do_newton,
    do_mean,
    do_sphere,
    doscaling,
    update_alpha,
    update_mu,
    update_beta,
    update_rho,
    chunk_size: int,
    sample_weight=None,
):
    """Chunked-accumulator version of _amica_step for CPU memory scalability.

    Loops over the time axis in chunks of `chunk_size`, accumulates
    sufficient statistics only, then performs a single M-step on the
    totals. Mathematically equivalent to _amica_step up to float64
    rounding (O(eps*T) ≈ 1e-10).

    Not JIT-compiled at the outer level (Python for-loop), but the
    per-chunk `compute_chunk_stats` IS JIT-compiled.

    Parameters
    ----------
    W : jnp.ndarray, shape (n_components, n_components)
        Current unmixing matrix.
    A : jnp.ndarray, shape (n_components, n_components)
        Current mixing matrix.
    c : jnp.ndarray, shape (n_components,)
        Current model centers.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Current mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Current mixture locations.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Current mixture inverse scales.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Current mixture shapes.
    gm : jnp.ndarray, shape (n_models,)
        Current model weights (placeholder for future multi-model).
    lrate_step : float
        Learning rate for the natural gradient step.
    rholrate : float
        Learning rate for the shape parameter updates.
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Whitened input data.
    log_det_sphere : float
        Log determinant of the sphering matrix.
    newt_start_iter : int
        Iteration to start applying Newton corrections.
    iteration : int
        Current iteration index.
    invsigmin : float
        Minimum allowed inverse scale parameter.
    invsigmax : float
        Maximum allowed inverse scale parameter.
    minrho : float
        Minimum allowed shape parameter.
    maxrho : float
        Maximum allowed shape parameter.
    do_newton : bool
        Whether to apply Newton correction.
    do_mean : bool
        Whether to center the data via model centers.
    do_sphere : bool
        Whether to use sphered data (affects log-likelihood).
    doscaling : bool
        Whether to re-scale mixing columns to unit norm.
    update_alpha : bool
        Whether to update mixture weights.
    update_mu : bool
        Whether to update mixture locations.
    update_beta : bool
        Whether to update mixture scales.
    update_rho : bool
        Whether to update mixture shapes.
    chunk_size : int
        Number of samples to process in each chunk.

    Returns
    -------
    W_new : jnp.ndarray
        Updated unmixing matrix.
    A_new : jnp.ndarray
        Updated mixing matrix.
    c_new : jnp.ndarray
        Updated model centers.
    alpha_new : jnp.ndarray
        Updated mixture weights.
    mu_new : jnp.ndarray
        Updated mixture locations.
    beta_new : jnp.ndarray
        Updated mixture inverse scales.
    rho_new : jnp.ndarray
        Updated mixture shapes.
    gm_new : jnp.ndarray
        Updated model weights (unchanged for single-model).
    ll : float
        Total log-likelihood of the current state.
    is_good : bool
        Whether the updated parameters are valid (no NaNs).
    newton_used : bool
        Whether the Newton correction was actively applied.
    """
    from .accumulators import add_stats, compute_chunk_stats, zero_stats
    from .likelihood import compute_log_det_W
    from .updates import (
        apply_alpha_update_from_stats,
        apply_beta_update_from_stats,
        apply_full_newton_correction,
        apply_mu_update_from_stats,
        apply_rho_update_from_stats,
        compute_newton_terms_from_stats,
    )

    n_samples = data_white.shape[1]
    n_components = W.shape[0]
    n_mix = alpha.shape[0]
    dtype = W.dtype

    # log|det W| is invariant across chunks within this iteration — compute the QR once
    # and pass it into every chunk instead of recomputing it per chunk.
    log_det_W = compute_log_det_W(W)

    # --- E-step: accumulate sufficient statistics over chunks ---
    totals = zero_stats(n_components, n_mix, dtype=dtype)
    data_sum_total = jnp.zeros((n_components,), dtype=dtype)

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        data_chunk = data_white[:, start:stop] - c[:, None]  # pre-center
        # sample_weight (None on the no-rejection path) zero-weights rejected
        # samples; slice it with the data so the partial sums stay aligned.
        w_chunk = None if sample_weight is None else sample_weight[start:stop]
        stats = compute_chunk_stats(
            data_chunk, W, alpha, mu, beta, rho, log_det_sphere, w_chunk, log_det_W
        )
        totals = add_stats(totals, stats)
        # Track uncentered data sum separately for c update
        if sample_weight is None:
            data_sum_total = data_sum_total + jnp.sum(data_white[:, start:stop], axis=1)
        else:
            data_sum_total = data_sum_total + jnp.sum(w_chunk * data_white[:, start:stop], axis=1)

    n_total = float(n_samples) if sample_weight is None else jnp.sum(sample_weight)
    ll = totals.ll_sum / n_total / n_components  # match compute_average_loglikelihood

    # --- Natural gradient ---
    gy = totals.gy_partial / n_total
    dA_local = jnp.eye(n_components, dtype=dtype) - gy

    # --- Newton correction ---
    newton_used = jnp.array(False)
    Wtmp = dA_local
    if do_newton and iteration >= newt_start_iter:
        sigma2, kappa, lambda_ = compute_newton_terms_from_stats(
            totals.sigma2_partial,
            totals.resp_sum,
            totals.kappa_numer,
            totals.lambda_numer,
            mu,
            beta,
            n_total,
        )
        lambda_pos = jnp.all(lambda_ > 0)
        Wtmp_newt, posdef_newt = apply_full_newton_correction(dA_local, sigma2, kappa, lambda_)
        is_valid = lambda_pos & posdef_newt
        Wtmp = jnp.where(is_valid, Wtmp_newt, dA_local)
        newton_used = is_valid

    # --- A/W update (same as _amica_step) ---
    dAk = A @ Wtmp
    A_new = A - lrate_step * dAk
    A_ok = jnp.all(jnp.isfinite(A_new))
    W_new = jnp.where(A_ok, jnp.linalg.pinv(A_new).astype(dtype), W)
    is_good = A_ok & jnp.all(jnp.isfinite(W_new))

    # --- M-step on PDF params using accumulated stats ---
    alpha_new = apply_alpha_update_from_stats(totals.resp_sum, n_total) if update_alpha else alpha

    if update_mu:
        mu_new = apply_mu_update_from_stats(
            mu,
            totals.mu_numer,
            totals.mu_denom_le2,
            totals.mu_denom_gt2,
            rho,
        )
    else:
        mu_new = mu

    if update_beta:
        beta_new = apply_beta_update_from_stats(
            beta,
            totals.resp_sum,
            totals.beta_denom_le2,
            totals.beta_denom_gt2,
            rho,
            invsigmin,
            invsigmax,
        )
    else:
        beta_new = beta

    if update_rho:
        rho_new = apply_rho_update_from_stats(
            rho,
            totals.rho_numer,
            totals.resp_sum,
            rholrate,
            minrho,
            maxrho,
        )
    else:
        rho_new = rho

    # --- c update ---
    c_new = data_sum_total / n_total if do_mean else c

    # --- Column scaling ---
    if doscaling:
        col_norms = jnp.linalg.norm(A_new, axis=0)
        col_norms = jnp.where(col_norms > 0.0, col_norms, 1.0)
        A_new = A_new / col_norms
        mu_new = mu_new * col_norms[None, :]
        beta_new = beta_new / col_norms[None, :]
        # Same identity as the full-batch step: row-scale the existing inverse
        # instead of recomputing pinv. inv(A·diag(1/c)) = diag(c)·inv(A).
        W_new = W_new * col_norms[:, None]

    # Stage 3F: guard state outputs (return unchanged input on a bad step) so the
    # caller can always accept — bit-exact with the old discard-on-bad path. The
    # chunked step is not jitted/donated, but the shared fit loop now always
    # accepts, so its outputs must follow the same keep-old-on-bad contract.
    W_new = jnp.where(is_good, W_new, W)
    A_new = jnp.where(is_good, A_new, A)
    c_new = jnp.where(is_good, c_new, c)
    alpha_new = jnp.where(is_good, alpha_new, alpha)
    mu_new = jnp.where(is_good, mu_new, mu)
    beta_new = jnp.where(is_good, beta_new, beta)
    rho_new = jnp.where(is_good, rho_new, rho)
    return (
        W_new,
        A_new,
        c_new,
        alpha_new,
        mu_new,
        beta_new,
        rho_new,
        gm,
        ll,
        is_good,
        newton_used,
    )


@partial(
    jax.jit,
    # Stage 3F: donate previous-iteration state buffers (W,A,c,alpha,mu,beta,rho —
    # positions 0-6; NOT gm) for in-place reuse. Safe: the fit loop always accepts
    # the returned (guarded) state, so these inputs are never reused post-call.
    donate_argnums=(0, 1, 2, 3, 4, 5, 6),
    static_argnames=[
        "do_newton",
        "do_mean",
        "do_sphere",
        "doscaling",
        "update_alpha",
        "update_mu",
        "update_beta",
        "update_rho",
    ],
)
def _amica_step_fused(
    W,
    A,
    c,
    alpha,
    mu,
    beta,
    rho,
    gm,
    lrate_step,
    rholrate,
    data_white,
    log_det_sphere,
    newt_start_iter,
    iteration,
    invsigmin,
    invsigmax,
    minrho,
    maxrho,
    do_newton,
    do_mean,
    do_sphere,
    doscaling,
    update_alpha,
    update_mu,
    update_beta,
    update_rho,
    sample_weight=None,
):
    """Single-graph fused full-batch step (Stage 3D).

    Same contract as ``_amica_step`` but computes responsibilities ONCE via the
    fused accumulator (``compute_chunk_stats`` on the whole batch) and derives
    the score, log-likelihood, Newton terms, and M-step sufficient statistics
    from that single pass — all inside ONE ``@jax.jit`` graph.

    ``_amica_step`` recomputes the generalized-Gaussian log-pdf 3-4x per
    iteration (compute_all_scores + compute_total_loglikelihood +
    compute_newton_terms + update_all_pdf_params). This version does it once.
    Numerically equivalent up to float64 rounding — the chunked path it reuses
    is parity-tested against ``_amica_step`` (``rel_err < 1e-4``).

    Unlike the eager ``_amica_step_chunked`` (Python loop + per-call dispatch),
    this is one fused XLA program, so it keeps the GPU launch profile of
    ``_amica_step`` while dropping the recompute.
    """
    from .accumulators import compute_chunk_stats
    from .updates import (
        apply_alpha_update_from_stats,
        apply_beta_update_from_stats,
        apply_full_newton_correction,
        apply_mu_update_from_stats,
        apply_rho_update_from_stats,
        compute_newton_terms_from_stats,
    )

    n_samples = data_white.shape[1]  # static (shape) → Python int
    n_components = W.shape[0]
    dtype = W.dtype

    # --- E-step: single fused pass over the whole batch ---
    data_centered = data_white - c[:, None]
    # sample_weight is None on the validated no-rejection path → these branches
    # resolve at trace time and that graph is byte-identical to the original.
    if sample_weight is None:
        n_total = float(n_samples)
        totals = compute_chunk_stats(data_centered, W, alpha, mu, beta, rho, log_det_sphere)
        data_sum_total = jnp.sum(data_white, axis=1)
    else:
        n_total = jnp.sum(sample_weight)
        totals = compute_chunk_stats(
            data_centered, W, alpha, mu, beta, rho, log_det_sphere, sample_weight
        )
        data_sum_total = jnp.sum(sample_weight * data_white, axis=1)

    ll = totals.ll_sum / n_total / n_components

    # --- Natural gradient ---
    # The fused accumulator carries some float64 fields, so cast back to the
    # compute dtype (matters in float32 mode; a no-op in float64).
    gy = (totals.gy_partial / n_total).astype(dtype)
    dA_local = (jnp.eye(n_components, dtype=dtype) - gy).astype(dtype)

    # --- Newton correction (lax.cond: iteration is traced under jit; both
    # branches must return identical dtypes — hence the explicit casts). ---
    def _apply_newton_stats(_):
        sigma2, kappa, lambda_ = compute_newton_terms_from_stats(
            totals.sigma2_partial,
            totals.resp_sum,
            totals.kappa_numer,
            totals.lambda_numer,
            mu,
            beta,
            n_total,
        )
        lambda_pos = jnp.all(lambda_ > 0)
        Wtmp_newt, posdef_newt = apply_full_newton_correction(dA_local, sigma2, kappa, lambda_)
        is_valid = lambda_pos & posdef_newt
        return jnp.where(is_valid, Wtmp_newt, dA_local).astype(dtype), is_valid

    def _skip_newton(_):
        return dA_local.astype(dtype), jnp.array(False)

    if do_newton:
        Wtmp, newton_used = jax.lax.cond(
            iteration >= newt_start_iter, _apply_newton_stats, _skip_newton, None
        )
    else:
        Wtmp = dA_local
        newton_used = jnp.array(False)

    # --- A / W update (row-scaled inverse identity applied in scaling below) ---
    dAk = A @ Wtmp
    A_new = A - lrate_step * dAk
    A_ok = jnp.all(jnp.isfinite(A_new))
    W_new = jnp.where(A_ok, jnp.linalg.pinv(A_new).astype(dtype), W)
    is_good = A_ok & jnp.all(jnp.isfinite(W_new))

    # --- M-step on PDF params from accumulated stats ---
    alpha_new = apply_alpha_update_from_stats(totals.resp_sum, n_total) if update_alpha else alpha
    mu_new = (
        apply_mu_update_from_stats(
            mu, totals.mu_numer, totals.mu_denom_le2, totals.mu_denom_gt2, rho
        )
        if update_mu
        else mu
    )
    beta_new = (
        apply_beta_update_from_stats(
            beta,
            totals.resp_sum,
            totals.beta_denom_le2,
            totals.beta_denom_gt2,
            rho,
            invsigmin,
            invsigmax,
        )
        if update_beta
        else beta
    )
    rho_new = (
        apply_rho_update_from_stats(
            rho, totals.rho_numer, totals.resp_sum, rholrate, minrho, maxrho
        )
        if update_rho
        else rho
    )

    # --- c update ---
    c_new = data_sum_total / n_total if do_mean else c

    # --- Column scaling (Stage 3B row-scaled inverse: no 2nd pinv) ---
    if doscaling:
        col_norms = jnp.linalg.norm(A_new, axis=0)
        col_norms = jnp.where(col_norms > 0.0, col_norms, 1.0)
        A_new = A_new / col_norms
        mu_new = mu_new * col_norms[None, :]
        beta_new = beta_new / col_norms[None, :]
        W_new = W_new * col_norms[:, None]

    # Stage 3F: guard state outputs (return unchanged input on a bad step) so the
    # caller can always accept — bit-exact with the old discard-on-bad path, and
    # required for donate_argnums (donated input buffers are never reused).
    W_new = jnp.where(is_good, W_new, W)
    A_new = jnp.where(is_good, A_new, A)
    c_new = jnp.where(is_good, c_new, c)
    alpha_new = jnp.where(is_good, alpha_new, alpha)
    mu_new = jnp.where(is_good, mu_new, mu)
    beta_new = jnp.where(is_good, beta_new, beta)
    rho_new = jnp.where(is_good, rho_new, rho)
    return (
        W_new,
        A_new,
        c_new,
        alpha_new,
        mu_new,
        beta_new,
        rho_new,
        gm,
        ll,
        is_good,
        newton_used,
    )


def _amica_step_multimodel(
    W,
    A,
    c,
    alpha,
    mu,
    beta,
    rho,
    gm,
    lrate_step,
    rholrate,
    data_white,
    log_det_sphere,
    newt_start_iter,
    iteration,
    invsigmin,
    invsigmax,
    minrho,
    maxrho,
    do_newton,
    do_mean,
    do_sphere,
    doscaling,
    update_alpha,
    update_mu,
    update_beta,
    update_rho,
    sample_weight=None,
):
    """Full-batch multi-model step (num_models > 1).

    Same call contract / return tuple as ``_amica_step_fused``, but the params
    carry a leading model axis and the E/M-step are the v-weighted multimodel
    versions. ``do_sphere`` and the ``update_*`` flags are accepted for signature
    compatibility but not used (multimodel always updates the pdf params;
    sphering is handled in preprocessing).
    """
    from .multimodel import compute_chunk_stats_mm, m_step_mm

    totals = compute_chunk_stats_mm(
        data_white, W, c, alpha, mu, beta, rho, gm, log_det_sphere, sample_weight
    )
    n_total = totals.n_chunk
    return m_step_mm(
        totals,
        W,
        A,
        c,
        alpha,
        mu,
        beta,
        rho,
        gm,
        n_total,
        lrate_step,
        rholrate,
        iteration,
        newt_start_iter,
        invsigmin,
        invsigmax,
        minrho,
        maxrho,
        do_newton,
        do_mean,
        doscaling,
    )


def _amica_step_multimodel_chunked(
    W,
    A,
    c,
    alpha,
    mu,
    beta,
    rho,
    gm,
    lrate_step,
    rholrate,
    data_white,
    log_det_sphere,
    newt_start_iter,
    iteration,
    invsigmin,
    invsigmax,
    minrho,
    maxrho,
    do_newton,
    do_mean,
    do_sphere,
    doscaling,
    update_alpha,
    update_mu,
    update_beta,
    update_rho,
    chunk_size,
    sample_weight=None,
):
    """Chunked multi-model step: Python time-loop + one M-step on the totals.

    Mirrors ``_amica_step_chunked``; each per-chunk ``compute_chunk_stats_mm``
    is jitted, the outer loop is plain Python.
    """
    from .multimodel import (
        add_stats_mm,
        compute_chunk_stats_mm,
        m_step_mm,
        zero_stats_mm,
    )

    n_models = W.shape[0]
    n_comp = W.shape[1]
    n_mix = alpha.shape[1]
    n_samples = data_white.shape[1]

    totals = zero_stats_mm(n_models, n_comp, n_mix, dtype=W.dtype)
    start = 0
    while start < n_samples:
        end = min(start + chunk_size, n_samples)
        w_chunk = None if sample_weight is None else sample_weight[start:end]
        totals = add_stats_mm(
            totals,
            compute_chunk_stats_mm(
                data_white[:, start:end], W, c, alpha, mu, beta, rho, gm, log_det_sphere, w_chunk
            ),
        )
        start = end

    # n_chunk now carries the effective good-sample count: == n_samples with no
    # rejection, == sum(sample_weight) when rejecting (so gm/ll normalize correctly).
    n_total = totals.n_chunk
    return m_step_mm(
        totals,
        W,
        A,
        c,
        alpha,
        mu,
        beta,
        rho,
        gm,
        n_total,
        lrate_step,
        rholrate,
        iteration,
        newt_start_iter,
        invsigmin,
        invsigmax,
        minrho,
        maxrho,
        do_newton,
        do_mean,
        doscaling,
    )


@dataclass
class AmicaResult:
    """Container for AMICA results.

    Notes
    -----
    **Matrix naming convention**

    AMICA operates in whitened space. Matrices are stored in both spaces
    with explicit suffixes to avoid ambiguity:

    - ``*_white_`` — whitened space (after sphering)
    - ``*_sensor_`` — original sensor space

    The relationship is::

        sources = unmixing_matrix_white_ @ whitener_ @ (data - mean_)
        data = mixing_matrix_sensor_ @ sources + mean_

    Attributes
    ----------
    unmixing_matrix_white_ : np.ndarray, shape (n_components, n_components)
        Unmixing matrix W in whitened space: ``sources = W @ x_white``.
    mixing_matrix_white_ : np.ndarray, shape (n_components, n_components)
        Mixing matrix A in whitened space: ``x_white = A @ sources + c``.
    unmixing_matrix_sensor_ : np.ndarray, shape (n_components, n_channels)
        Full unmixing in sensor space: ``W @ sphere``.
    mixing_matrix_sensor_ : np.ndarray, shape (n_channels, n_components)
        Full mixing in sensor space: ``desphere @ A``.
    whitener_ : np.ndarray, shape (n_components, n_channels)
        Sphering/whitening matrix S.
    dewhitener_ : np.ndarray, shape (n_channels, n_components)
        Dewhitening matrix (pseudo-inverse of sphere).
    mean_ : np.ndarray, shape (n_channels,)
        Data mean removed during preprocessing.
    alpha_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Mixture weights for each component.
    mu_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Location parameters.
    rho_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Shape parameters.
    sbeta_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Scale parameters (inverse beta).
    c_ : np.ndarray, shape (n_components,) or (n_models, n_components)
        Model centers.
    gm_ : np.ndarray, shape (n_models,)
        Model weights (for multi-model).
    log_likelihood : np.ndarray, shape (n_iter,)
        Log-likelihood per iteration.
    iteration_times : np.ndarray, shape (n_iter,)
        Wall-clock time per iteration in seconds.
    elapsed_times : np.ndarray, shape (n_iter,)
        Cumulative wall-clock time in seconds.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged.
    sample_mask_ : np.ndarray of bool or None, shape (n_samples,)
        Likelihood-based sample-rejection mask over the fit-input samples
        (True = kept). ``None`` when ``do_reject`` was off or never fired.
        Single-model only; indexes the data passed to ``fit`` (post-decim /
        post-epoch-reject), not the original recording times.
    n_rejected_ : int
        Number of samples rejected (``(~sample_mask_).sum()``); 0 if no rejection.
    """

    unmixing_matrix_white_: np.ndarray
    mixing_matrix_white_: np.ndarray
    unmixing_matrix_sensor_: np.ndarray
    mixing_matrix_sensor_: np.ndarray
    whitener_: np.ndarray
    dewhitener_: np.ndarray
    mean_: np.ndarray
    alpha_: np.ndarray
    mu_: np.ndarray
    rho_: np.ndarray
    sbeta_: np.ndarray
    c_: np.ndarray
    gm_: np.ndarray
    log_likelihood: np.ndarray
    n_iter: int
    iteration_times: np.ndarray = field(default_factory=lambda: np.array([]))
    elapsed_times: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = False
    data_scale: float = 1.0
    # Multi-model only (num_models > 1): the model-posterior time-course
    # p(h|t), shape (n_models, n_samples). None for single-model fits.
    model_posteriors_: np.ndarray | None = None
    # Likelihood-based sample rejection (single-model do_reject). Boolean mask
    # over the fit-input samples, True = kept; None when rejection was not run.
    sample_mask_: np.ndarray | None = None
    n_rejected_: int = 0

    def to_mne(self, info):
        """Convert results to MNE ICA object.

        AMICA decomposes as: ``sources = W @ S @ (x - mean)``
        where W is the unmixing matrix in whitened space and S is the
        sphering/whitening matrix.

        MNE's ICA reconstructs via:
        ``unmixing_full = unmixing_ @ pca_components_[:n_comp]``
        ``mixing_full = pca_components_.T @ mixing_``

        To make these equivalent we use QR decomposition on the combined
        transform ``W @ S`` to extract an orthonormal ``pca_components_``
        (Q.T) and a square ``unmixing_matrix_`` (R), satisfying MNE's
        requirement that ``pca_components_`` has orthonormal rows.

        Parameters
        ----------
        info : mne.Info
            Measurement info (from the Raw/Epochs used for fitting).

        Returns
        -------
        ica : mne.preprocessing.ICA
            Fitted MNE ICA object compatible with plot_components(),
            get_sources(), apply(), and ICLabel.
        """
        try:
            from mne.preprocessing import ICA
        except ImportError as err:
            raise ImportError("MNE-Python is required for to_mne().") from err

        W = np.asarray(self.unmixing_matrix_white_)  # (n_comp, n_comp)
        S = np.asarray(self.whitener_)  # (n_comp, n_ch)
        n_components = W.shape[0]
        n_channels = S.shape[1]

        # Combined transform: sources = WS @ (x - mean)
        WS = W @ S  # (n_comp, n_ch)

        # QR decomposition: WS.T = Q @ R  =>  WS = R.T @ Q.T
        # Q.T is orthonormal (n_comp, n_ch) — use as pca_components_
        # R.T is square (n_comp, n_comp) — use as unmixing_matrix_ (before norms)
        Q, R = np.linalg.qr(WS.T, mode="reduced")  # Q: (n_ch, n_comp), R: (n_comp, n_comp)
        pca_components = Q.T  # (n_comp, n_ch) — orthonormal rows
        unmixing_raw = R.T  # (n_comp, n_comp) — square

        # Verify: WS ≈ unmixing_raw @ pca_components
        # (this is exact by QR construction)

        ica = ICA(n_components=n_components, method="infomax")

        # Build full orthonormal pca_components (n_ch, n_ch).
        # First n_comp rows = Q.T from QR. Complete to orthonormal basis
        # using SVD of Q to get its orthogonal complement.
        U_full, _, _Vt_full = np.linalg.svd(Q, full_matrices=True)
        # U_full: (n_ch, n_ch) orthonormal columns
        # First n_comp columns span same space as Q
        # Remaining columns span the null space
        pca_full = U_full.T  # (n_ch, n_ch) — orthonormal rows
        # But we need the first n_comp rows to be exactly Q.T (= pca_components)
        # SVD may reorder/flip signs. Use Q directly and append null space.
        null_space = U_full[:, n_components:]  # (n_ch, n_ch - n_comp)
        pca_full = np.vstack([pca_components, null_space.T])
        ica.pca_components_ = pca_full

        ica.pca_mean_ = np.asarray(self.mean_)
        ica.pre_whitener_ = np.ones((n_channels, 1))

        # pca_explained_variance_ — MNE divides unmixing by sqrt(variance)
        # during fit(). We need to match that convention.
        # Since our pca_components are orthonormal, the "variance" each
        # component explains is the squared column norm of unmixing_raw.
        col_var = np.sum(unmixing_raw**2, axis=0)
        col_var[col_var == 0] = 1.0
        pca_explained_variance = np.ones(n_channels)
        pca_explained_variance[:n_components] = col_var
        ica.pca_explained_variance_ = pca_explained_variance

        # Apply MNE's normalization: unmixing /= sqrt(variance)
        norms = np.sqrt(col_var)
        ica.unmixing_matrix_ = unmixing_raw / norms
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

        # Metadata
        ica.n_components_ = n_components
        ica.n_pca_components = n_channels
        ica.info = info
        ica.ch_names = info["ch_names"][:n_channels]
        ica.n_iter_ = self.n_iter
        ica.current_fit = "raw"
        ica.method = "amica"
        ica._is_fitted = True
        ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_components)]

        return ica


def _gpu_memory_budget_bytes(memory_fraction: float) -> float | None:
    """Free VRAM (bytes) x memory_fraction on the active GPU, or None.

    Returns None when JAX is unavailable, no GPU device is present, or the
    backend does not expose memory_stats() (e.g. some ROCm / older jaxlib).
    The caller falls back to the system-RAM (psutil) path in that case.
    """
    try:
        gpus = [d for d in jax.devices() if getattr(d, "platform", "") in ("gpu", "cuda", "rocm")]
    except Exception:
        return None
    if not gpus:
        return None
    try:
        stats = gpus[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit")
    if limit is None:
        return None
    in_use = int(stats.get("bytes_in_use", 0) or 0)
    free = max(0, int(limit) - in_use)
    return float(free) * memory_fraction


def _active_device_is_gpu() -> bool:
    """True when JAX's default device is a GPU."""
    try:
        return any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in jax.devices())
    except Exception:
        return False


def _choose_chunk_size(
    n_samples: int,
    n_components: int,
    n_mix_comps: int,
    dtype: np.dtype[Any] | None = None,
    memory_fraction: float = 0.25,
    device: str | None = None,
) -> int:
    """Return chunk_size that keeps hot-buffer allocation within available memory.

    Hot buffers per chunk: y and g (n_comp x B each) plus per-mixture
    intermediates (~5 x n_comp x n_mix x B). Formula mirrors scott-huberty's
    choose_batch_size() adapted for our single-model NumPy/JAX buffers.

    Memory budget source:
      - On GPU (``device='gpu'`` or auto-detected), the budget is derived from
        the active device's free VRAM via ``jax.devices()[0].memory_stats()``.
        This prevents the OOM that a system-RAM estimate would cause on a
        consumer GPU.
      - On CPU (or when VRAM stats are unavailable), the budget is system RAM
        via psutil, falling back to a 2 GiB budget when psutil is missing.

    The budget is capped at 4 GiB of hot buffers either way. Returns n_samples
    when everything fits (caller treats as full-batch).
    """
    dtype_size = np.dtype(np.float64 if dtype is None else dtype).itemsize
    bytes_per_sample = int(
        (1 + 2 * n_components + 5 * n_components * n_mix_comps) * dtype_size * 1.2
    )

    on_gpu = (device == "gpu") or (device is None and _active_device_is_gpu())
    budget = None
    budget_source = "system-RAM"
    if on_gpu:
        gpu_budget = _gpu_memory_budget_bytes(memory_fraction)
        if gpu_budget is not None:
            budget = min(gpu_budget, 4.0 * 1024**3)
            budget_source = "GPU-VRAM"
    if budget is None:
        try:
            import psutil

            avail = psutil.virtual_memory().available
            budget = min(float(avail) * memory_fraction, 4.0 * 1024**3)
        except Exception:
            budget = 2.0 * 1024**3  # 2 GiB fallback

    max_chunk = int(budget // bytes_per_sample)
    if max_chunk < 1:
        raise MemoryError(
            f"Cannot fit even 1 sample in {budget / 1024**3:.1f} GiB {budget_source} budget. "
            f"Per-sample cost: {bytes_per_sample / 1024:.1f} KB."
        )

    ram_chunk = min(n_samples, max_chunk)
    min_chunk = min(max(8192, n_components * 32), n_samples)
    if ram_chunk < min_chunk:
        logger.warning(
            "Auto chunk_size %d (from %s budget) is below recommended minimum %d. "
            "Free memory or reduce n_components to avoid very small chunks.",
            ram_chunk,
            budget_source,
            min_chunk,
        )

    # On CPU, RAM headroom is an UPPER bound, not the throughput optimum. The E-step is
    # memory-bandwidth-bound: a chunk whose (n_comp, n_mix, B) hot buffer stays resident in
    # last-level cache runs ~2x faster than the largest-that-fits-RAM chunk (and uses far
    # less memory). Measured on 8-core x86, jax CPU: ~1.98x end-to-end at B~2048, LL
    # identical to 1e-14 — see benchmarks/profile_cpu.py / validate_chunk.py. GPU is left
    # filling its VRAM budget (large chunks feed occupancy; the GPU path is out of scope).
    if not on_gpu:
        # Target ~2 MiB for the (n_comp, n_mix, B) hot buffer — the throughput knee measured
        # on this class of machine sits at an L2-resident working set (~1.5-2 MiB); 4 MiB was
        # measurably slower. It is a heuristic, not a hardware probe — re-tune with
        # benchmarks/profile_cpu.py if the LLC differs materially.
        target_bytes = 2 * 1024**2
        cache_chunk = int(target_bytes // max(1, n_components * n_mix_comps * dtype_size))
        cache_chunk = max(2048, cache_chunk)  # floor: below this per-chunk dispatch dominates
        return min(ram_chunk, cache_chunk)
    return ram_chunk


class Amica:
    """Native JAX implementation of AMICA algorithm.

    Adaptive Mixture Independent Component Analysis (AMICA) performs ICA
    with adaptive source density modeling using mixtures of generalized
    Gaussians.

    Parameters
    ----------
    config : AmicaConfig, optional
        Configuration object with all algorithm parameters.
        If None, uses default configuration.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    config : AmicaConfig
        Algorithm configuration.
    result_ : AmicaResult
        Fitted model (available after calling fit).

    Examples
    --------
    >>> from amica import Amica, AmicaConfig
    >>> config = AmicaConfig(max_iter=500, num_mix_comps=3)
    >>> amica = Amica(config, random_state=42)
    >>> result = amica.fit(data)  # data: (n_channels, n_samples)
    >>> activations = (
    ...     result.unmixing_matrix_white_ @ result.whitener_ @ (data - result.mean_[:, None])
    ... )

    Notes
    -----
    The AMICA algorithm was developed by Jason Palmer at UCSD.
    This is a native Python/JAX implementation for GPU acceleration.

    References
    ----------
    .. [1] Palmer et al. (2008). Newton method for the ICA mixture model.
           Proc. IEEE ICASSP.
    .. [2] Palmer et al. (2011). AMICA: An adaptive mixture of independent
           component analyzers with shared components. UCSD Technical Report.
    """

    def __init__(
        self,
        config: AmicaConfig | None = None,
        random_state: int | None = None,
    ):
        self.config = config if config is not None else AmicaConfig()
        self.random_state = random_state
        self.rng = jax.random.PRNGKey(random_state if random_state is not None else 0)
        self.result_: AmicaResult | None = None

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"config": self.config, "random_state": self.random_state}

    def set_params(self, **params) -> Amica:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : Amica
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray, shape (n_channels, n_samples)
            Input data.
        y : None
            Ignored.

        Returns
        -------
        X_new : np.ndarray, shape (n_components, n_samples)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def fit(
        self,
        data: np.ndarray,
        init_mean: np.ndarray | None = None,
        init_sphere: np.ndarray | None = None,
        init_weights: np.ndarray | None = None,
        init_params: dict | None = None,
    ) -> AmicaResult:
        """Fit AMICA model to data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            Input EEG/MEG data. Should be high-pass filtered.
        init_mean : np.ndarray, optional
            Precomputed mean vector to use instead of computing from data.
        init_sphere : np.ndarray, optional
            Precomputed sphering matrix to use instead of computing from data.
        init_weights : np.ndarray, optional
            Precomputed unmixing matrix (W) to use for initialization.
        init_params : dict, optional
            Dictionary containing initial values for 'alpha', 'mu', 'beta', 'rho'.

        Returns
        -------
        result : AmicaResult
            Fitted model containing mixing/unmixing matrices and
            all model parameters.
        """
        # Multi-model (num_models > 1) is handled by a separate, fully-isolated
        # step path selected below; the single-model path is left untouched so
        # M=1 results stay bit-for-bit identical.

        # Determine target JAX dtype
        dtype = jnp.float32 if self.config.dtype == "float32" else jnp.float64

        # Preprocessing usually done in float64 for stability, then cast to target dtype
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        if nan_count or inf_count:
            raise ValueError(
                f"Input data contains non-finite values: {nan_count} NaN, {inf_count} Inf. "
                "Interpolate or remove bad channels/epochs before fitting."
            )

        # Check for potential unit mismatch (uV vs Volts)
        # Amica (and Engine parity) works best with Volts (std ~ 1e-5 to 1e-4)
        # If std > 0.5, assume uV or similar large unit and auto-scale.
        scaling_factor = 1.0
        data_std = np.std(data)
        if data_std > 1e2:
            # Very large values suggest microvolts or similar units.
            # EEG in Volts: std ~ 1e-5 to 1e-4. In uV: std ~ 10 to 100.
            scaling_factor = 1.0 / data_std
            logger.info(
                "Data std (%.2e) very large. Auto-scaling by %.2e for stability.",
                data_std,
                scaling_factor,
            )
            data = data * scaling_factor

        n_channels, n_samples = data.shape
        logger.info(
            "Fitting %d channels x %d samples using %s", n_channels, n_samples, self.config.dtype
        )

        # Determine number of components
        if self.config.pcakeep is not None:
            n_components = min(self.config.pcakeep, n_channels)
        else:
            n_components = n_channels

        # ========== Preprocessing ==========
        logger.info("Preprocessing (mean removal, sphering)...")
        data_white, mean, sphere, desphere, n_components, eigenvalues = preprocess_data(
            data,
            do_mean=self.config.do_mean,
            do_sphere=self.config.do_sphere,
            pcakeep=self.config.pcakeep,
            mineig=self.config.mineig,
            do_approx=self.config.do_approx_sphere,
            sphere_type=self.config.sphere_type,
            init_mean=init_mean,
            init_sphere=init_sphere,
        )

        # Compute (log, det)(S)| from kept eigenvalues (Fortran sldet)
        safe_eigs = np.maximum(np.asarray(eigenvalues[:n_components]), self.config.mineig)
        self.log_det_sphere = float(-0.5 * np.sum(np.log(safe_eigs)))

        log_det_sphere = self.log_det_sphere

        logger.info("Using %d components", n_components)

        # ========== Initialize Parameters ==========
        W, A, alpha, mu, beta, rho, c, gm = self._initialize_params(
            n_components,
            self.config.num_mix_comps,
            self.config.num_models,
            dtype=dtype,
            init_weights=init_weights,
            init_params=init_params,
        )

        # Store learning rates (can be modified during optimization)
        lrate0 = self.config.lrate
        lrate = lrate0
        newtrate = self.config.newtrate
        rholrate0 = self.config.rholrate
        rholrate = rholrate0
        numdecs = 0
        numincs = 0

        # ========== Main EM Loop ==========
        LL: list[float] = []
        iteration_times: list[float] = []
        elapsed_times: list[float] = []
        converged = False
        newton_count = 0  # Track how many iterations actually used Newton
        natgrad_fallback_count = 0  # Track Newton fallbacks
        start_time = time.perf_counter()

        # Initial ll_prev for first iteration
        ll_prev_val = -np.inf

        # Sample rejection state (single-model likelihood rejection)
        rej_count = 0  # Number of rejection passes done
        n_samples_orig = data_white.shape[1]  # mask is indexed over these samples
        good_mask = None  # bool (n_samples_orig,), lazily created on the first rejection
        sample_weight_jax = None  # jnp 0/1 mask passed to the M=1 step; None = no rejection

        # Convert config flags to static arguments once
        do_newton_static = self.config.do_newton
        do_mean_static = self.config.do_mean
        do_sphere_static = self.config.do_sphere
        doscaling_static = self.config.doscaling
        update_alpha_static = self.config.update_alpha
        update_mu_static = self.config.update_mu
        update_beta_static = self.config.update_beta
        update_rho_static = self.config.update_rho

        # Resolve effective chunk size once (avoids psutil call each iteration).
        # Pass the actual compute dtype so float32 gets a correspondingly larger
        # chunk, and let _choose_chunk_size auto-detect GPU vs CPU to size the
        # budget against VRAM (GPU) or system RAM (CPU).
        _cfg_cs = self.config.chunk_size
        _eff_chunk_size: int | None
        if _cfg_cs == "auto":
            _chunk_dtype = np.dtype(np.float32 if self.config.dtype == "float32" else np.float64)
            # Multi-model multiplies the per-sample E-step tensors by num_models;
            # inflate the per-sample cost estimate so auto-chunking sizes against it.
            _eff_nmix = self.config.num_mix_comps * max(1, self.config.num_models)
            _eff_chunk_size = _choose_chunk_size(
                n_samples,
                n_components,
                _eff_nmix,
                dtype=_chunk_dtype,
            )
            if _eff_chunk_size >= n_samples:
                _eff_chunk_size = None  # everything fits — full batch
            else:
                logger.info("Auto chunk_size: %d samples", _eff_chunk_size)
        elif isinstance(_cfg_cs, int):
            _eff_chunk_size = _cfg_cs
        else:
            _eff_chunk_size = None  # None = full batch

        # Likelihood rejection passes a per-sample good-mask as the trailing
        # `sample_weight` kwarg; the single-model fused/chunked AND the multimodel
        # paths use it (a None mask = no rejection; classic rejects a non-None mask).
        _multimodel = self.config.num_models > 1
        _step_fn: Callable[..., Any]
        if _eff_chunk_size is not None:
            _cs = _eff_chunk_size
            if _multimodel:

                def _step_fn(*args, sample_weight=None, _cs=_cs):
                    return _amica_step_multimodel_chunked(
                        *args, chunk_size=_cs, sample_weight=sample_weight
                    )
            else:
                # Chunked single-model path is the fused single-pass accumulator.
                def _step_fn(*args, sample_weight=None, _cs=_cs):
                    return _amica_step_chunked(*args, chunk_size=_cs, sample_weight=sample_weight)
        elif _multimodel:
            # Full-batch multi-model: v-weighted per-model E/M-step; a rejection mask
            # is applied globally across models (folded into the per-sample posteriors).
            def _step_fn(*args, sample_weight=None):
                return _amica_step_multimodel(*args, sample_weight=sample_weight)
        else:
            # Full-batch single-model: choose fused (default) vs classic.
            # "auto" resolves to fused — numerically equivalent to classic
            # (matches to ~1e-15 per step) but computes responsibilities once.
            # "classic" is the parity oracle / exact-reproduction escape hatch.
            _estep = self.config.estep
            if _estep == "classic":

                def _step_fn(*args, sample_weight=None):
                    if sample_weight is not None:
                        raise NotImplementedError(
                            "Likelihood-based sample rejection requires the fused "
                            "E-step (estep='auto' or 'fused'); it is not implemented "
                            "for estep='classic'."
                        )
                    return _amica_step(*args)
            else:  # "auto" or "fused"

                def _step_fn(*args, sample_weight=None):
                    return _amica_step_fused(*args, sample_weight=sample_weight)

        # Ensure initial state is JAX array with correct dtype
        W = jnp.asarray(W, dtype=dtype)
        A = jnp.asarray(A, dtype=dtype)
        c = jnp.asarray(c, dtype=dtype)
        alpha = jnp.asarray(alpha, dtype=dtype)
        mu = jnp.asarray(mu, dtype=dtype)
        beta = jnp.asarray(beta, dtype=dtype)
        rho = jnp.asarray(rho, dtype=dtype)
        gm = jnp.asarray(gm, dtype=dtype)
        data_white = jnp.asarray(data_white, dtype=dtype)

        for iteration in range(self.config.max_iter):
            iter_start = time.perf_counter()

            # Stage 1 — decay on LL decrease
            if iteration > 0 and len(LL) >= 2 and LL[-1] < LL[-2]:
                if lrate <= self.config.minlrate:
                    logger.info("Converged at iteration %d (lrate <= minlrate)", iteration)
                    converged = True
                    break
                lrate = lrate * self.config.lratefact
                rholrate = rholrate * self.config.rholratefact
                numdecs += 1
                if numdecs >= self.config.max_decs:
                    lrate0 = lrate0 * self.config.lratefact
                    if iteration > self.config.newt_start:
                        rholrate0 = rholrate0 * self.config.rholratefact
                    if self.config.do_newton and iteration > self.config.newt_start:
                        newtrate = newtrate * self.config.lratefact
                    numdecs = 0
            elif iteration > 0 and len(LL) >= 2 and LL[-1] > LL[-2]:
                numdecs = 0

            # Stage 2 — per-iter ramp toward ceiling
            in_newton = self.config.do_newton and (iteration >= self.config.newt_start)
            ceiling = newtrate if in_newton else lrate0
            lrate = min(ceiling, lrate + min(1.0 / self.config.newt_ramp, lrate))

            (
                W_new,
                A_new,
                c_new,
                alpha_new,
                mu_new,
                beta_new,
                rho_new,
                gm_new,
                ll_curr,
                is_good,
                newton_used,
            ) = _step_fn(
                W,
                A,
                c,
                alpha,
                mu,
                beta,
                rho,
                gm,
                lrate,
                rholrate,
                data_white,
                log_det_sphere,
                # Config scalars
                self.config.newt_start,
                iteration,
                self.config.invsigmin,
                self.config.invsigmax,
                self.config.minrho,
                self.config.maxrho,
                # Config flags (static)
                do_newton_static,
                do_mean_static,
                do_sphere_static,
                doscaling_static,
                update_alpha_static,
                update_mu_static,
                update_beta_static,
                update_rho_static,
                sample_weight=sample_weight_jax,
            )

            # Stage 3F: always accept the (guarded) state. On a bad step each
            # step function now returns the UNCHANGED input, so this reproduces
            # the former discard-on-bad behavior exactly, while letting
            # donate_argnums reuse the step's input buffers safely.
            W, A, c, alpha, mu, beta, rho, gm = (
                W_new,
                A_new,
                c_new,
                alpha_new,
                mu_new,
                beta_new,
                rho_new,
                gm_new,
            )

            # Block until scalars are ready (synchronize for checking)
            is_good_val = bool(is_good)
            ll_val = float(ll_curr)
            newton_used_val = bool(newton_used)

            if not is_good_val:
                lrate *= 0.5
                if iteration % 10 == 0:
                    logger.warning(
                        "Iter %d: NaN/Inf detected, reducing lrate to %.2e", iteration, lrate
                    )
                iteration_times.append(time.perf_counter() - iter_start)
                elapsed_times.append(time.perf_counter() - start_time)
                continue

            LL.append(ll_val)

            # ========== Sample Rejection (Fortran 1.7 do_reject) ==========
            # Monotone cumulative mask following amica17 reject_data: each round
            # recomputes the threshold mean - rejsig*std of the per-sample TOTAL LL over
            # the CURRENT good samples and drops those below it. Rejected samples are
            # never re-accepted and are zero-weighted in the EM via `sample_weight` (the
            # data is NOT subsetted). For M>1 rejection is GLOBAL across models, on the
            # mixture LL (matches Fortran's single global good-mask).
            if (
                self.config.do_reject
                and iteration >= self.config.rejstart
                and rej_count < self.config.numrej
                and (iteration - self.config.rejstart) % self.config.rejint == 0
            ):
                # Per-sample TOTAL LL over ALL samples (data space). M>1 = the mixture LL
                # logsumexp_h[log gm_h + log p(x_t|h)]; M=1 = the single-model LL.
                if self.config.num_models == 1:
                    y_rej = jnp.dot(W, data_white - c[:, None])
                    sample_lls = compute_source_loglikelihood(y_rej, alpha, mu, beta, rho)
                    sample_lls = np.asarray(sample_lls + compute_log_det_W(W) + log_det_sphere)
                else:
                    from .multimodel import compute_model_sample_loglik

                    # Same (M, n_components, n_samples) intermediate as the post-fit
                    # posterior pass, but this runs once per rejection round, so it is
                    # chunked on the same budget.
                    sample_lls = np.asarray(
                        compute_model_sample_loglik(
                            data_white,
                            W,
                            c,
                            alpha,
                            mu,
                            beta,
                            rho,
                            gm,
                            log_det_sphere,
                            chunk_size=_eff_chunk_size,
                        )
                    )

                if good_mask is None:
                    good_mask = np.ones(n_samples_orig, dtype=bool)
                good_mask, _n_newly = _reject_threshold(sample_lls, good_mask, self.config.rejsig)
                rej_count += 1
                # Flip on the JAX weight (None -> 0/1 array on the first rejection round).
                sample_weight_jax = jnp.asarray(good_mask.astype(np.float64), dtype=dtype)
                n_rej_total = int((~good_mask).sum())
                if iteration % 10 == 0 or rej_count == 1:
                    logger.info(
                        "Iter %d: rejection round %d — %d/%d samples rejected (%.2f%%)",
                        iteration,
                        rej_count,
                        n_rej_total,
                        n_samples_orig,
                        100.0 * n_rej_total / n_samples_orig,
                    )

            dll = ll_val - ll_prev_val
            if iteration > 0 and self.config.use_min_dll:
                if dll < self.config.min_dll:
                    numincs += 1
                    if numincs > self.config.max_incs:
                        logger.info("Converged at iteration %d (dll < min_dll)", iteration)
                        converged = True
                        iteration_times.append(time.perf_counter() - iter_start)
                        elapsed_times.append(time.perf_counter() - start_time)
                        break
                else:
                    numincs = 0

            # Track Newton usage
            if iteration >= self.config.newt_start and do_newton_static:
                if newton_used_val:
                    newton_count += 1
                else:
                    natgrad_fallback_count += 1

            # Progress output
            if iteration % 10 == 0:
                if iteration >= self.config.newt_start and do_newton_static:
                    mode = "N" if newton_used_val else "ng"
                    logger.info(
                        "Iter %4d: LL = %.6f, lrate = %.2e [%s]", iteration, ll_val, lrate, mode
                    )
                else:
                    logger.info("Iter %4d: LL = %.6f, lrate = %.2e", iteration, ll_val, lrate)

            # Checkpoint
            if (
                self.config.outdir is not None
                and self.config.writestep > 0
                and (iteration + 1) % self.config.writestep == 0
            ):
                # Need to bring back to CPU for saving
                self.result_ = AmicaResult(
                    unmixing_matrix_white_=np.asarray(W),
                    mixing_matrix_white_=np.asarray(A),
                    unmixing_matrix_sensor_=np.asarray(W @ sphere),
                    mixing_matrix_sensor_=np.asarray(desphere @ A),
                    whitener_=np.asarray(sphere),
                    dewhitener_=np.asarray(desphere),
                    mean_=np.asarray(mean),
                    alpha_=np.asarray(alpha),
                    mu_=np.asarray(mu),
                    rho_=np.asarray(rho),
                    sbeta_=np.asarray(beta),
                    c_=np.asarray(c),
                    gm_=np.asarray(gm),
                    log_likelihood=np.array(LL),
                    iteration_times=np.array(iteration_times),
                    elapsed_times=np.array(elapsed_times),
                    n_iter=len(LL),
                    converged=converged,
                    data_scale=scaling_factor,
                )
                self.save(self.config.outdir)
                logger.info("Saved checkpoint to %s", self.config.outdir)

            ll_prev_val = ll_val
            iteration_times.append(time.perf_counter() - iter_start)
            elapsed_times.append(time.perf_counter() - start_time)

        if not converged:
            logger.info("Reached max_iter (%d)", self.config.max_iter)

        # Newton diagnostic summary
        if do_newton_static and self.config.newt_start < len(LL):
            total_newton_iters = newton_count + natgrad_fallback_count
            if total_newton_iters > 0:
                pct = 100.0 * newton_count / total_newton_iters
                logger.info(
                    "Newton: %d/%d iterations used Newton (%.0f%%), "
                    "%d fell back to natural gradient",
                    newton_count,
                    total_newton_iters,
                    pct,
                    natgrad_fallback_count,
                )

        # ========== Multi-model: model-posterior time-course p(h|t) ==========
        model_posteriors = None
        if self.config.num_models > 1:
            from .multimodel import compute_model_posteriors

            # Reuse the E-step's chunk size: the intermediate per-model sources here
            # are (M, n_components, n_samples), so a full-batch pass can dominate peak
            # memory on a long recording even when the fit itself was chunked.
            model_posteriors = np.asarray(
                compute_model_posteriors(
                    data_white,
                    W,
                    c,
                    alpha,
                    mu,
                    beta,
                    rho,
                    gm,
                    log_det_sphere,
                    chunk_size=_eff_chunk_size,
                )
            )

        # ========== Construct Result ==========
        self.result_ = AmicaResult(
            unmixing_matrix_white_=np.asarray(W),
            mixing_matrix_white_=np.asarray(A),
            unmixing_matrix_sensor_=np.asarray(W @ sphere),
            mixing_matrix_sensor_=np.asarray(desphere @ A),
            whitener_=np.asarray(sphere),
            dewhitener_=np.asarray(desphere),
            mean_=np.asarray(mean),
            alpha_=np.asarray(alpha),
            mu_=np.asarray(mu),
            rho_=np.asarray(rho),
            sbeta_=np.asarray(beta),
            c_=np.asarray(c),
            gm_=np.asarray(gm),
            log_likelihood=np.array(LL),
            iteration_times=np.array(iteration_times),
            elapsed_times=np.array(elapsed_times),
            n_iter=len(LL),
            converged=converged,
            data_scale=scaling_factor,
            model_posteriors_=model_posteriors,
            sample_mask_=good_mask,
            n_rejected_=int((~good_mask).sum()) if good_mask is not None else 0,
        )

        assert self.result_ is not None
        return self.result_

    def _initialize_params(
        self,
        n_components: int,
        n_mix: int,
        n_models: int,
        dtype: Any = jnp.float64,
        init_weights: np.ndarray | None = None,
        init_params: dict | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        """Initialize model parameters.

        Parameters
        ----------
        n_components : int
            Number of ICA components.
        n_mix : int
            Number of Gaussian mixture components.
        n_models : int
            Number of ICA models.
        dtype : Any
            JAX dtype for parameters (jnp.float32 or jnp.float64).
        init_weights : np.ndarray, optional
            Precomputed unmixing matrix (W) to use.
        init_params : dict, optional
            Dictionary containing initial values for 'alpha', 'mu', 'beta', 'rho'.

        Returns
        -------
        W : jnp.ndarray, shape (n_components, n_components)
            Initial unmixing matrix.
        A : jnp.ndarray, shape (n_components, n_components)
            Initial mixing matrix (inverse of W).
        alpha : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture weights.
        mu : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture locations.
        beta : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture inverse scales (sbeta).
        rho : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture shape parameters.
        c : jnp.ndarray, shape (n_components,)
            Initial model centers.
        gm : jnp.ndarray, shape (n_models,)
            Initial model probabilities.
        """
        rng = np.random.default_rng(self.random_state)

        if n_models > 1:
            # Multi-model init: per-model arrays with INDEPENDENT jitter so the
            # models start distinct (identical inits never separate under EM).
            # init_weights/init_params are not applied per-model here; use a 3D
            # warm-start path in future if needed.
            def _rand_A():
                if self.config.fix_init:
                    return np.eye(n_components, dtype=np.float64)
                A_h = np.eye(n_components, dtype=np.float64) + 0.01 * (
                    0.5 - rng.random((n_components, n_components))
                )
                cn = np.linalg.norm(A_h, axis=0)
                cn = np.where(cn > 0.0, cn, 1.0)
                return A_h / cn

            A = jnp.asarray(np.stack([_rand_A() for _ in range(n_models)]), dtype=dtype)
            W = jax.vmap(jnp.linalg.pinv)(A)

            alpha = jnp.ones((n_models, n_mix, n_components), dtype=dtype) / n_mix
            base = np.arange(n_mix, dtype=np.float64) - (n_mix - 1) / 2.0
            mu_np = (
                np.broadcast_to(base[None, :, None], (n_models, n_mix, n_components))
                .astype(np.float64)
                .copy()
            )
            if not self.config.fix_init:
                mu_np = mu_np + 0.05 * (1.0 - 2.0 * rng.random((n_models, n_mix, n_components)))
            mu = jnp.asarray(mu_np, dtype=dtype)
            if self.config.fix_init:
                beta = jnp.ones((n_models, n_mix, n_components), dtype=dtype)
            else:
                beta = jnp.asarray(
                    1.0 + 0.1 * (0.5 - rng.random((n_models, n_mix, n_components))),
                    dtype=dtype,
                )
            rho = jnp.full((n_models, n_mix, n_components), self.config.rho0, dtype=dtype)
            c = jnp.zeros((n_models, n_components), dtype=dtype)
            gm = jnp.ones(n_models, dtype=dtype) / n_models
            return W, A, alpha, mu, beta, rho, c, gm

        # Initialize mixing matrix then invert to get W.
        if init_weights is not None:
            W = jnp.asarray(init_weights, dtype=dtype)
            A = jnp.linalg.pinv(W)
        else:
            if self.config.fix_init:
                A_np = np.eye(n_components, dtype=np.float64)
            else:
                noise = rng.random((n_components, n_components))
                A_np = 0.01 * (0.5 - noise)
                A_np += np.eye(n_components, dtype=np.float64)
                col_norms = np.linalg.norm(A_np, axis=0)
                col_norms = np.where(col_norms > 0.0, col_norms, 1.0)
                A_np = A_np / col_norms
            A = jnp.asarray(A_np, dtype=dtype)
            W = jnp.linalg.pinv(A)

        # Initialize mixture parameters
        if init_params is not None and "alpha" in init_params:
            alpha = jnp.asarray(init_params["alpha"], dtype=dtype)
        else:
            # alpha: uniform mixture weights
            alpha = jnp.ones((n_mix, n_components), dtype=dtype) / n_mix

        if init_params is not None and "mu" in init_params:
            mu = jnp.asarray(init_params["mu"], dtype=dtype)
        else:
            base = np.arange(n_mix, dtype=np.float64) - (n_mix - 1) / 2.0
            mu_np = base[:, None] * np.ones((n_mix, n_components), dtype=np.float64)
            if not self.config.fix_init:
                noise = rng.random((n_mix, n_components))
                mu_np = mu_np + 0.05 * (1.0 - 2.0 * noise)
            mu = jnp.asarray(mu_np, dtype=dtype)

        if init_params is not None and ("beta" in init_params or "sbeta" in init_params):
            beta_key = "beta" if "beta" in init_params else "sbeta"
            beta = jnp.asarray(init_params[beta_key], dtype=dtype)
        else:
            if self.config.fix_init:
                beta = jnp.ones((n_mix, n_components), dtype=dtype)
            else:
                noise = rng.random((n_mix, n_components))
                beta = jnp.asarray(1.0 + 0.1 * (0.5 - noise), dtype=dtype)

        if init_params is not None and "rho" in init_params:
            rho = jnp.asarray(init_params["rho"], dtype=dtype)
        else:
            # rho: start at middle value
            rho = jnp.full((n_mix, n_components), self.config.rho0, dtype=dtype)

        # Model center: zero
        c = jnp.zeros(n_components, dtype=dtype)

        # Model weights: uniform
        gm = jnp.ones(n_models, dtype=dtype) / n_models

        return W, A, alpha, mu, beta, rho, c, gm

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted unmixing to new data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            New data to transform.

        Returns
        -------
        sources : np.ndarray, shape (n_components, n_samples)
            Source activations.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        data = np.asarray(data, dtype=np.float64) * self.result_.data_scale

        # Apply full unmixing: W @ sphere @ (x - mean)
        centered = data - self.result_.mean_[:, None]
        whitened = self.result_.whitener_ @ centered
        sources = self.result_.unmixing_matrix_white_ @ whitened

        return sources

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Reconstruct data from sources.

        Parameters
        ----------
        sources : np.ndarray, shape (n_components, n_samples)
            Source activations.

        Returns
        -------
        data : np.ndarray, shape (n_channels, n_samples)
            Reconstructed data.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        sources = np.asarray(sources, dtype=np.float64)

        # data = A @ sources + mean = desphere @ A_white @ sources + mean
        data = self.result_.mixing_matrix_sensor_ @ sources + self.result_.mean_[:, None]

        return data / self.result_.data_scale

    def save(self, outdir: str | Path) -> None:
        """Save model to directory in AMICA-compatible format.

        Parameters
        ----------
        outdir : str or Path
            Output directory.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Save in Fortran-compatible binary format (column-major)
        def save_binary(name: str, arr: np.ndarray):
            arr.astype("<f8").T.tofile(outdir / name)

        save_binary("A", self.result_.mixing_matrix_sensor_)
        save_binary("W", self.result_.unmixing_matrix_white_)
        save_binary("S", self.result_.whitener_)
        save_binary("mean", self.result_.mean_)
        save_binary("alpha", self.result_.alpha_)
        save_binary("mu", self.result_.mu_)
        save_binary("rho", self.result_.rho_)
        save_binary("sbeta", self.result_.sbeta_)
        save_binary("c", self.result_.c_)
        save_binary("gm", self.result_.gm_)
        save_binary("LL", self.result_.log_likelihood)

        logger.info("Saved model to %s", outdir)

    @classmethod
    def load(cls, outdir: str | Path) -> Amica:
        """Load model from AMICA-compatible directory.

        Parameters
        ----------
        outdir : str or Path
            Input directory containing AMICA binary files.

        Returns
        -------
        model : Amica
            Loaded AMICA model.
        """
        outdir = Path(outdir)
        if not outdir.exists():
            raise FileNotFoundError(f"Directory {outdir} does not exist")

        # Helper to read binary double files
        def read_bin(name, shape=None):
            p = outdir / name
            if not p.exists():
                return None
            # AMICA writes doubles (float64) in column-major order
            data = np.fromfile(p, dtype=np.float64)
            if shape is not None:
                data = data.reshape(shape, order="F")
            return data

        # Load parameters
        W = read_bin("W")
        if W is None:
            raise FileNotFoundError(f"Could not find W in {outdir}")

        # W is n_components x n_components
        n_components = int(np.sqrt(W.size))
        W = W.reshape((n_components, n_components), order="F")

        S = read_bin("S", (n_components, n_components))
        if S is None:
            S = np.eye(n_components)

        A = read_bin("A", (n_components, n_components))
        if A is None:
            A = np.linalg.pinv(W)

        mean = read_bin("mean", (n_components,))
        if mean is None:
            mean = np.zeros(n_components)

        c = read_bin("c", (n_components,))
        if c is None:
            c = np.zeros(n_components)

        # Check size of alpha to infer n_mix
        alpha_raw = read_bin("alpha")
        if alpha_raw is not None:
            n_mix = alpha_raw.size // n_components
            alpha = alpha_raw.reshape((n_mix, n_components), order="F")
            mu = read_bin("mu", (n_mix, n_components))
            sbeta = read_bin("sbeta", (n_mix, n_components))
            rho = read_bin("rho", (n_mix, n_components))
        else:
            n_mix = 1
            alpha = np.ones((n_mix, n_components))
            mu = np.zeros((n_mix, n_components))
            sbeta = np.ones((n_mix, n_components))
            rho = np.ones((n_mix, n_components)) * 1.5

        gm = read_bin("gm", (1,))
        if gm is None:
            gm = np.ones(1)

        LL = read_bin("LL")
        if LL is None:
            LL = np.array([])

        result = AmicaResult(
            unmixing_matrix_white_=W,
            mixing_matrix_white_=np.linalg.pinv(W),
            unmixing_matrix_sensor_=W @ S,
            mixing_matrix_sensor_=A,
            whitener_=S,
            dewhitener_=np.linalg.pinv(S),
            mean_=mean,
            alpha_=alpha,
            mu_=mu,
            rho_=rho,
            sbeta_=sbeta,
            c_=c,
            gm_=gm,
            log_likelihood=LL,
            n_iter=len(LL),
            converged=True,
            data_scale=1.0,
        )

        # Reconstruct the Amica object
        config = AmicaConfig(num_mix_comps=n_mix)
        model = cls(config=config)
        model.result_ = result
        return model


def amica(
    X,
    n_components=None,
    whiten=False,
    return_n_iter=False,
    random_state=None,
    max_iter=2000,
    num_mix=3,
    **kwargs,
):
    """Adaptive Mixture ICA (AMICA).

    Returns the ``(K, W, Y)`` tuple MNE-Python's ICA dispatch expects (the
    calling convention shared by its fastica/infomax/picard methods), used in MNE-Python's
    ``ICA`` dispatch (``method='amica'``).

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
        Pre-whitened data, features x samples.  This matches MNE's
        ICA-method convention; MNE passes ``data[:, sel].T`` which gives
        (n_components, n_samples).
    n_components : int or None
        Number of components. If None, uses X.shape[0].
    whiten : bool
        If True, whiten the data internally. MNE always passes False
        (data is pre-whitened by MNE's PCA step).
    return_n_iter : bool
        If True, return n_iter as a fourth element: ``K, W, Y, n_iter``.
    random_state : int or None
        Random seed for reproducibility.
    max_iter : int
        Maximum number of EM iterations.
    num_mix : int
        Number of generalized Gaussian mixture components per source.
    **kwargs
        Additional parameters passed to AmicaConfig.

    Returns
    -------
    K : None
        Pre-whitening matrix.  Always None when ``whiten=False``.
        Included for the MNE ICA-method signature; MNE discards this value.
    W : ndarray, shape (n_components, n_components)
        Unmixing matrix (operates on whitened data).
    Y : ndarray, shape (n_components, n_samples)
        Source matrix: ``W @ X``.
    n_iter : int
        Number of iterations. Only returned when ``return_n_iter=True``,
        as the fourth element.
    """
    from .config import AmicaConfig

    cfg_kwargs = {
        "max_iter": max_iter,
        "num_mix_comps": num_mix,
        "do_sphere": whiten,
        "do_mean": whiten,
    }
    cfg_kwargs.update(kwargs)
    config = AmicaConfig(**cfg_kwargs)

    # X is (n_features, n_samples) — same shape the AMICA solver expects.
    solver = Amica(config, random_state=random_state)
    result = solver.fit(X)

    K = None  # whiten=False: MNE pre-whitens; kept for the MNE ICA-method signature
    W = result.unmixing_matrix_white_  # (n_components, n_components)
    Y = W @ X  # (n_components, n_samples)

    if return_n_iter:
        return K, W, Y, result.n_iter
    return K, W, Y
