"""Tests for amica package."""

import os
import subprocess
import sys

import numpy as np
import pytest


def test_fit_random_data():
    """Test basic fitting on random data."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 500
    data = rng.randn(n_channels, n_samples)

    config = AmicaConfig(
        max_iter=20,
        num_mix_comps=2,
        do_newton=False,
    )
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    assert result.unmixing_matrix_white_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_white_.shape == (n_channels, n_channels)
    assert result.unmixing_matrix_sensor_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_sensor_.shape == (n_channels, n_channels)
    assert len(result.log_likelihood) > 0


def test_transform_inverse():
    """Test that transform + inverse_transform reconstructs data."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 500
    data = rng.randn(n_channels, n_samples)

    config = AmicaConfig(max_iter=20, num_mix_comps=2, do_newton=False)
    model = Amica(config=config, random_state=42)
    model.fit(data)

    sources = model.transform(data)
    assert sources.shape == (n_channels, n_samples)

    recon = model.inverse_transform(sources)
    assert recon.shape == (n_channels, n_samples)

    # Linear unmix→remix is algebraically exact; residual should be at machine-eps level.
    fro_norm_rel = np.linalg.norm(data - recon, "fro") / np.linalg.norm(data, "fro")
    assert fro_norm_rel < 1e-12, (
        f"Round-trip Frobenius relative residual too high: {fro_norm_rel:.2e}"
    )


def test_ll_increases():
    """Test that log-likelihood generally increases over iterations."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(123)
    n_channels, n_samples = 4, 1000
    # Create data with some structure (mixed sources)
    S = rng.laplace(size=(n_channels, n_samples))
    A = rng.randn(n_channels, n_channels)
    data = A @ S

    config = AmicaConfig(max_iter=100, num_mix_comps=3, do_newton=True)
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    ll = result.log_likelihood
    assert len(ll) > 10
    # LL at end should be higher than at start (allowing some tolerance)
    assert ll[-1] > ll[5], "Log-likelihood should increase over training"


"""Test the Picard-compatible functional API."""


def test_amica_function():
    """Test amica() functional API returns correct shapes."""
    from amica import amica

    rng = np.random.RandomState(42)
    n_components, n_samples = 4, 500
    # picard convention: (n_features, n_samples)
    X = rng.randn(n_components, n_samples)

    K, W, Y = amica(X, max_iter=20, num_mix=2)
    assert K is None
    assert W.shape == (n_components, n_components)
    assert Y.shape == (n_components, n_samples)


def test_amica_return_n_iter():
    """Test return_n_iter flag."""
    from amica import amica

    rng = np.random.RandomState(42)
    X = rng.randn(4, 500)

    K, W, Y, n_iter = amica(X, max_iter=20, num_mix=2, return_n_iter=True)
    assert K is None
    assert W.shape == (4, 4)
    assert Y.shape == (4, 500)
    assert isinstance(n_iter, int)
    assert n_iter > 0


"""Test actual source separation quality."""


def test_separate_laplacian_sources():
    """Test separation of known Laplacian sources."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(0)
    n_sources, n_samples = 3, 5000

    # Generate independent Laplacian sources
    S = rng.laplace(size=(n_sources, n_samples))

    # Random mixing
    A_true = rng.randn(n_sources, n_sources)
    X = A_true @ S

    config = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=True)
    model = Amica(config=config, random_state=42)
    result = model.fit(X)

    # Recover sources
    _ = model.transform(X)

    # Check Amari index (permutation-invariant separation quality)
    # Perfect separation: each row/col of W @ A_true has one dominant entry
    C = result.unmixing_matrix_white_ @ result.whitener_ @ A_true
    # Normalize rows and columns
    C = C / np.max(np.abs(C), axis=1, keepdims=True)

    # Amari index: sum of (sum/max - 1) for rows and cols
    row_ratios = np.sum(np.abs(C), axis=1) / np.max(np.abs(C), axis=1) - 1
    col_ratios = np.sum(np.abs(C), axis=0) / np.max(np.abs(C), axis=0) - 1
    amari = (np.mean(row_ratios) + np.mean(col_ratios)) / 2

    assert amari < 0.3, f"Amari index too high ({amari:.3f}), poor separation"


"""Test explicit matrix naming and consistency."""


def test_matrix_shapes_and_consistency():
    """Test all four matrices have correct shapes and are consistent."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 6, 2000
    S = rng.laplace(size=(n_channels, n_samples))
    A_true = rng.randn(n_channels, n_channels)
    data = A_true @ S

    config = AmicaConfig(max_iter=50, num_mix_comps=2, do_newton=False)
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    # White-space matrices are square (n_comp x n_comp)
    assert result.unmixing_matrix_white_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_white_.shape == (n_channels, n_channels)

    # Sensor-space matrices bridge channels and components
    assert result.unmixing_matrix_sensor_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_sensor_.shape == (n_channels, n_channels)

    # W_white @ A_white ≈ I (algebraic identity; machine-eps level)
    WA = result.unmixing_matrix_white_ @ result.mixing_matrix_white_
    np.testing.assert_allclose(WA, np.eye(n_channels), atol=1e-12)

    # sensor unmixing = W_white @ sphere
    expected_sensor = result.unmixing_matrix_white_ @ result.whitener_
    np.testing.assert_allclose(result.unmixing_matrix_sensor_, expected_sensor, atol=1e-10)

    # sensor mixing = desphere @ A_white
    expected_mix = result.dewhitener_ @ result.mixing_matrix_white_
    np.testing.assert_allclose(result.mixing_matrix_sensor_, expected_mix, atol=1e-10)


def test_sensor_roundtrip():
    """Test mixing_sensor @ unmixing_sensor ≈ I for full-rank data."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 1000
    data = rng.randn(n_channels, n_samples)

    config = AmicaConfig(max_iter=20, num_mix_comps=2, do_newton=False)
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    # mixing_sensor @ unmixing_sensor should be identity to machine precision
    product = result.mixing_matrix_sensor_ @ result.unmixing_matrix_sensor_
    np.testing.assert_allclose(product, np.eye(n_channels), atol=1e-12)


"""Test configuration validation."""


def test_defaults():
    """Test default config values match literature recommendations."""
    from amica import AmicaConfig

    cfg = AmicaConfig()
    assert cfg.max_iter == 2000
    assert cfg.num_mix_comps == 3
    assert cfg.do_newton
    assert cfg.newt_start == 50
    assert cfg.rejstart == 2
    assert cfg.rejint == 3
    assert cfg.rejsig == 3.0


def test_invalid_config():
    """Test that invalid config raises errors."""
    from amica import AmicaConfig

    with pytest.raises(ValueError):
        AmicaConfig(num_models=0)
    with pytest.raises(ValueError):
        AmicaConfig(minrho=0.5)
    with pytest.raises(ValueError):
        AmicaConfig(maxrho=3.0)


@pytest.mark.parametrize(
    "bad_data,label",
    [
        (np.full((4, 100), np.nan), "all-NaN"),
        (np.full((4, 100), np.inf), "all-Inf"),
        (np.full((4, 100), -np.inf), "all-neg-Inf"),
        # one bad sample injected into otherwise clean data
        (lambda: _inject(np.random.RandomState(0).randn(4, 100), 0, 50, np.nan), "one-NaN"),
        (lambda: _inject(np.random.RandomState(0).randn(4, 100), 1, 20, np.inf), "one-Inf"),
    ],
    ids=["all-NaN", "all-Inf", "all-neg-Inf", "one-NaN", "one-Inf"],
)
def test_nonfinite_input_raises(bad_data, label):
    """Fit must raise ValueError on NaN or Inf input."""
    from amica import Amica, AmicaConfig

    data = bad_data() if callable(bad_data) else bad_data
    model = Amica(AmicaConfig(max_iter=2), random_state=0)
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(data)


def _inject(arr, row, col, value):
    arr[row, col] = value
    return arr


"""Test sample rejection feature."""


def test_rejection_enabled():
    """Test that rejection runs without errors when enabled."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 1000
    data = rng.randn(n_channels, n_samples)
    # Add some outliers
    data[:, :10] *= 100

    config = AmicaConfig(
        max_iter=30,
        num_mix_comps=2,
        do_reject=True,
        rejstart=2,
        rejint=3,
        rejsig=3.0,
        numrej=3,
        do_newton=False,
    )
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    assert result is not None
    assert len(result.log_likelihood) > 0


"""Chunked E-step should match full-batch within float64 rounding."""


def test_chunked_loglik_additivity():
    """sum(compute_loglik_chunk) across halves == compute_total_loglikelihood."""
    from amica.backend import jnp
    from amica.likelihood import (
        compute_loglik_chunk,
        compute_total_loglikelihood,
    )

    rng = np.random.RandomState(0)
    n_comp, n_mix, n_samp = 4, 3, 10000
    y = rng.randn(n_comp, n_samp)
    W = np.eye(n_comp) + 0.01 * rng.randn(n_comp, n_comp)
    alpha = np.ones((n_mix, n_comp)) / n_mix
    mu = rng.randn(n_mix, n_comp) * 0.1
    beta = np.ones((n_mix, n_comp)) + 0.05 * rng.randn(n_mix, n_comp)
    rho = np.full((n_mix, n_comp), 1.5)

    ll_full = float(
        compute_total_loglikelihood(
            jnp.asarray(y),
            jnp.asarray(W),
            jnp.asarray(alpha),
            jnp.asarray(mu),
            jnp.asarray(beta),
            jnp.asarray(rho),
            log_det_sphere=0.3,
        )
    )
    ll_h1, n1 = compute_loglik_chunk(
        jnp.asarray(y[:, :5000]),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_sphere=0.3,
    )
    ll_h2, n2 = compute_loglik_chunk(
        jnp.asarray(y[:, 5000:]),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_sphere=0.3,
    )
    ll_merged = float((ll_h1 + ll_h2) / (n1 + n2) / n_comp)
    assert abs(ll_full - ll_merged) < 1e-12


def test_chunked_matches_fullbatch_synthetic():
    """Chunked vs full-batch: W and LL agree within rounding after 50 iters."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_src, n_samp = 4, 5000
    srcs = np.stack(
        [
            rng.laplace(size=n_samp),
            rng.standard_t(df=3, size=n_samp),
            rng.laplace(size=n_samp) * 1.5,
            np.sign(rng.randn(n_samp)) * rng.exponential(size=n_samp),
        ]
    )[:n_src]
    srcs = srcs / srcs.std(axis=1, keepdims=True)
    A_true = rng.randn(n_src, n_src)
    A_true = A_true / np.linalg.norm(A_true, axis=0, keepdims=True)
    x = A_true @ srcs

    cfg_kw = {
        "num_models": 1,
        "num_mix_comps": 3,
        "max_iter": 50,
        "dtype": "float64",
        "pcakeep": n_src,
    }
    res_full = Amica(AmicaConfig(**cfg_kw, chunk_size=None), random_state=42).fit(x)
    res_chunk = Amica(AmicaConfig(**cfg_kw, chunk_size=1024), random_state=42).fit(x)

    W_full = np.asarray(res_full.unmixing_matrix_white_)
    W_chunk = np.asarray(res_chunk.unmixing_matrix_white_)
    rel_err = np.max(np.abs(W_chunk - W_full)) / np.max(np.abs(W_full))
    assert rel_err < 1e-4, f"Chunked W diverged from full-batch: rel_err={rel_err:.2e}"

    ll_full = float(np.asarray(res_full.log_likelihood)[-1])
    ll_chunk = float(np.asarray(res_chunk.log_likelihood)[-1])
    assert abs(ll_full - ll_chunk) < 1e-5, (
        f"Final LL diverged: full={ll_full:.8f} chunk={ll_chunk:.8f}"
    )


@pytest.fixture
def tiny_data():
    """Very small data array for fast testing: 4 channels, 20 samples."""
    rng = np.random.RandomState(42)
    return rng.randn(4, 20)


def test_newton_path(tiny_data):
    """Test the full Newton correction path."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(
        do_newton=True,
        newt_start=1,
        newt_ramp=1,
        max_iter=3,
    )
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.n_iter == 3


def test_chunked_path(tiny_data):
    """Test accumulator path with time-axis chunking and Newton."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(
        chunk_size=5,  # Smaller than n_samples (20)
        max_iter=2,
        do_newton=True,
        newt_start=1,
    )
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.n_iter == 2


def test_chunked_path_no_updates(tiny_data):
    """Test accumulator path with no parameter updates."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(
        chunk_size=5,
        max_iter=1,
        update_alpha=False,
        update_mu=False,
        update_beta=False,
        update_rho=False,
        do_mean=False,
        doscaling=False,
    )
    solver = Amica(config, random_state=42)
    solver.fit(tiny_data)


# ---------------------------------------------------------------------------
# Auto chunk_size tests
# ---------------------------------------------------------------------------


def test_auto_chunk_size_returns_valid_int():
    """_choose_chunk_size returns int in [1, n_samples]."""
    from amica.solver import _choose_chunk_size

    cs = _choose_chunk_size(n_samples=10000, n_components=32, n_mix_comps=3)
    assert isinstance(cs, int)
    assert 1 <= cs <= 10000


def test_auto_chunk_size_bounded_by_n_samples():
    """_choose_chunk_size never exceeds n_samples."""
    from amica.solver import _choose_chunk_size

    cs = _choose_chunk_size(n_samples=100, n_components=4, n_mix_comps=3)
    assert cs <= 100


def test_auto_chunk_size_small_budget():
    """_choose_chunk_size returns small chunk when available RAM is tiny."""
    psutil = pytest.importorskip("psutil")
    from unittest.mock import MagicMock, patch

    from amica.solver import _choose_chunk_size

    mock_vmem = MagicMock()
    mock_vmem.available = 5 * 1024 * 1024  # 5 MiB
    with patch.object(psutil, "virtual_memory", return_value=mock_vmem):
        cs = _choose_chunk_size(
            n_samples=500_000, n_components=64, n_mix_comps=3, memory_fraction=1.0
        )
    assert cs < 500_000


def test_chunk_size_auto_config_accepted():
    """AmicaConfig accepts chunk_size='auto' without error."""
    from amica.config import AmicaConfig

    cfg = AmicaConfig(chunk_size="auto")
    assert cfg.chunk_size == "auto"


def test_chunk_size_auto_config_rejects_bad_string():
    """AmicaConfig rejects unknown string for chunk_size."""
    from amica.config import AmicaConfig

    with pytest.raises(ValueError, match="chunk_size"):
        AmicaConfig(chunk_size="bad")


def test_chunk_size_auto_fit_completes(tiny_data):
    """chunk_size='auto' fit runs without error."""
    from amica import Amica, AmicaConfig

    res = Amica(AmicaConfig(chunk_size="auto", max_iter=3), random_state=42).fit(tiny_data)
    assert res.n_iter == 3


def test_chunk_size_auto_matches_fullbatch(tiny_data):
    """chunk_size='auto' on data that fits in RAM gives same W as full-batch."""
    from amica import Amica, AmicaConfig

    kw = {"max_iter": 10, "dtype": "float64"}
    res_full = Amica(AmicaConfig(**kw, chunk_size=None), random_state=42).fit(tiny_data)
    res_auto = Amica(AmicaConfig(**kw, chunk_size="auto"), random_state=42).fit(tiny_data)
    np.testing.assert_allclose(
        np.asarray(res_auto.unmixing_matrix_white_),
        np.asarray(res_full.unmixing_matrix_white_),
        atol=1e-10,
    )


def test_chunk_size_default_is_auto():
    """The CPU default was flipped to 'auto' (cache-sized chunking)."""
    from amica import AmicaConfig

    assert AmicaConfig().chunk_size == "auto"


def test_auto_actually_chunks_and_matches_fullbatch():
    """On data large enough to trigger cache-sized chunking, 'auto' must reproduce the
    full-batch fixed point to ~float64 reassociation noise (algebraic identity, not
    byte-identical). Guards the default flip in config.chunk_size."""
    from amica import Amica, AmicaConfig
    from amica.solver import _choose_chunk_size

    n_ch, T = 16, 12000
    rng = np.random.RandomState(0)
    x = rng.randn(n_ch, n_ch) @ rng.laplace(size=(n_ch, T))

    # Sanity: auto genuinely chunks here (chunk < T), otherwise the test is vacuous.
    assert _choose_chunk_size(T, n_ch, 3, dtype=np.dtype(np.float64)) < T

    kw = {"max_iter": 8, "dtype": "float64", "num_mix_comps": 3}
    res_full = Amica(AmicaConfig(**kw, chunk_size=None), random_state=42).fit(x)
    res_auto = Amica(AmicaConfig(**kw, chunk_size="auto"), random_state=42).fit(x)

    ll_full = np.asarray(res_full.log_likelihood)
    ll_auto = np.asarray(res_auto.log_likelihood)
    max_rel = np.max(np.abs(ll_full - ll_auto)) / max(np.max(np.abs(ll_full)), 1.0)
    assert max_rel < 1e-6, f"LL trajectory diverged: rel {max_rel:.2e}"
    np.testing.assert_allclose(
        np.asarray(res_auto.unmixing_matrix_white_),
        np.asarray(res_full.unmixing_matrix_white_),
        atol=1e-8,
    )


def test_chunk_size_auto_takes_chunked_path_when_forced():
    """chunk_size='auto' uses chunked E-step when psutil reports tiny RAM."""
    psutil = pytest.importorskip("psutil")
    from unittest.mock import MagicMock, patch

    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(0)
    srcs = rng.laplace(size=(4, 2000))
    x = rng.randn(4, 4) @ srcs

    mock_vmem = MagicMock()
    mock_vmem.available = 1 * 1024 * 1024  # 1 MiB — forces tiny chunk
    with patch.object(psutil, "virtual_memory", return_value=mock_vmem):
        res = Amica(
            AmicaConfig(chunk_size="auto", max_iter=5, dtype="float64"), random_state=42
        ).fit(x)
    assert res.n_iter == 5
    assert np.all(np.isfinite(np.asarray(res.unmixing_matrix_white_)))


def test_float32_dtype(tiny_data):
    """Test float32 precision mode."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(dtype="float32", max_iter=2)
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    # Just ensure it ran, exact dtype might be float64 in fallback mode
    assert res.unmixing_matrix_white_.shape == (4, 4)


@pytest.mark.slow
def test_float32_matches_float64_parity():
    """float32 is a *fast* mode, not the reference: it must recover the same
    spatial filters as float64.

    This locks the local-GPU usability claim (Stage 3A: ``--dtype float32``
    halves memory and is faster on consumer GPUs without changing the
    solution). Uses a well-determined synthetic Laplacian mixture and the same
    worst-case matched-cosine gate as the JAX-vs-NumPy backend parity test.
    Slow because it runs two full 300-iteration fits.
    """
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(3)
    n_src, n_samp = 12, 8000
    S = rng.laplace(size=(n_src, n_samp))
    A_true = rng.randn(n_src, n_src)
    data = (A_true @ S).astype(np.float64)

    def _fit(dt):
        cfg = AmicaConfig(max_iter=300, num_mix_comps=3, do_newton=True, dtype=dt)
        return np.asarray(Amica(cfg, random_state=3).fit(data).unmixing_matrix_white_, dtype=float)

    W64 = _fit("float64")
    W32 = _fit("float32")

    min_sim = _align_and_compare(W64, W32)
    assert min_sim > 0.99, (
        f"float32 vs float64 worst matched component cosine: {min_sim:.4f} < 0.99"
    )


def test_pcakeep_reduces_components(tiny_data):
    """Test pcakeep reduces the component count."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(pcakeep=2, max_iter=2)
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.unmixing_matrix_white_.shape == (2, 2)
    assert res.mean_.shape == (4,)
    assert res.whitener_.shape == (2, 4)


def test_fit_transform_and_inverse(tiny_data):
    """Test fit_transform output matches fit().transform() and inverse_transform works."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(max_iter=2)
    solver1 = Amica(config, random_state=42)
    sources1 = solver1.fit_transform(tiny_data)

    solver2 = Amica(config, random_state=42)
    solver2.fit(tiny_data)
    sources2 = solver2.transform(tiny_data)

    np.testing.assert_allclose(sources1, sources2)

    reconstructed = solver2.inverse_transform(sources2)
    assert reconstructed.shape == tiny_data.shape

    # Test unfitted error
    unfitted = Amica(config)
    with pytest.raises(RuntimeError):
        unfitted.transform(tiny_data)
    with pytest.raises(RuntimeError):
        unfitted.inverse_transform(tiny_data)
    with pytest.raises(RuntimeError):
        unfitted.save("dummy")


def test_multimodel_runs(tiny_data):
    """Multi-model (num_models > 1) now runs and returns per-model results.

    (Previously asserted NotImplementedError; multi-model is implemented as of
    the multimodel-amica work — see tests/test_multimodel.py for the full
    parity + recovery suite.)
    """
    from amica import Amica, AmicaConfig

    config = AmicaConfig(num_models=2, max_iter=3)
    solver = Amica(config, random_state=42)
    result = solver.fit(tiny_data)

    n_comp = result.unmixing_matrix_white_.shape[-1]
    # per-model leading axis of 2
    assert result.unmixing_matrix_white_.shape == (2, n_comp, n_comp)
    assert result.gm_.shape == (2,)
    assert np.isclose(result.gm_.sum(), 1.0, atol=1e-6)
    # model-posterior time-course present and normalized
    assert result.model_posteriors_ is not None
    assert result.model_posteriors_.shape == (2, tiny_data.shape[1])
    assert np.allclose(result.model_posteriors_.sum(axis=0), 1.0, atol=1e-6)


def test_checkpoint_save_load(tiny_data, tmp_path):
    """Test writing checkpoints and reloading."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(
        outdir=tmp_path / "amica_out",
        max_iter=3,
        writestep=1,  # Write every iteration
    )
    solver = Amica(config, random_state=42)
    res_orig = solver.fit(tiny_data)

    # Load from checkpoint
    solver_loaded = Amica.load(tmp_path / "amica_out")
    res_loaded = solver_loaded.result_

    # Compare
    np.testing.assert_allclose(res_loaded.unmixing_matrix_white_, res_orig.unmixing_matrix_white_)

    # Test load missing dir
    with pytest.raises(FileNotFoundError):
        Amica.load(tmp_path / "nonexistent")

    # Test load missing W
    (tmp_path / "amica_out" / "W").unlink()
    with pytest.raises(FileNotFoundError):
        Amica.load(tmp_path / "amica_out")


def test_checkpoint_load_partial(tiny_data, tmp_path):
    """Test loading a directory where some optional files are missing."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(max_iter=2)
    solver = Amica(config, random_state=42)
    solver.fit(tiny_data)
    solver.save(tmp_path / "amica_out2")

    # Delete optional files
    for fname in ["S", "A", "mean", "c", "alpha", "mu", "rho", "sbeta", "gm", "LL"]:
        if (tmp_path / "amica_out2" / fname).exists():
            (tmp_path / "amica_out2" / fname).unlink()

    solver_loaded = Amica.load(tmp_path / "amica_out2")
    res = solver_loaded.result_
    # Should fallback to defaults
    assert res.alpha_.shape == (1, 4)
    assert res.whitener_.shape == (4, 4)
    assert res.mean_.shape == (4,)
    assert res.log_likelihood.size == 0


def test_rejection_path(tiny_data):
    """Test outlier rejection path."""
    from amica import Amica, AmicaConfig

    config = AmicaConfig(
        do_reject=True,
        rejstart=1,
        numrej=1,
        rejint=1,
        max_iter=2,
    )
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.n_iter == 2
    # Rejection ran (rejstart=1) → the mask is exposed over the input samples.
    assert res.sample_mask_ is not None
    assert res.sample_mask_.shape == (tiny_data.shape[1],)
    assert res.n_rejected_ == int((~res.sample_mask_).sum()) >= 0


def test_rejection_behavioral():
    """Behavioral checks for sample rejection on contaminated data.

    Assertions:
    - Output matrix shapes are correct (rejection is internal; outputs are full-rank).
    - W @ A ≈ I in whitened space (inverse relationship preserved after rejection).
    - Final LL is higher with rejection than without on spike-contaminated data.
    """
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(7)
    n_ch, n_samp = 6, 2000

    # Clean Laplacian sources
    S = rng.laplace(size=(n_ch, n_samp))
    A_true = rng.randn(n_ch, n_ch)
    data = A_true @ S

    # Inject large-amplitude spikes into 5% of samples
    n_spike = n_samp // 20
    spike_idx = rng.choice(n_samp, size=n_spike, replace=False)
    data_contaminated = data.copy()
    data_contaminated[:, spike_idx] *= 50.0

    shared_cfg = {
        "max_iter": 60,
        "num_mix_comps": 2,
        "do_newton": False,
        "rejstart": 10,
        "rejint": 5,
        "numrej": 3,
        "rejsig": 3.0,
    }

    solver_rej = Amica(AmicaConfig(**shared_cfg, do_reject=True), random_state=42)
    res_rej = solver_rej.fit(data_contaminated)

    solver_no = Amica(AmicaConfig(**shared_cfg, do_reject=False), random_state=42)
    res_no = solver_no.fit(data_contaminated)

    # Shape: rejection is internal; outputs always (n_ch, n_ch)
    assert res_rej.unmixing_matrix_white_.shape == (n_ch, n_ch)
    assert res_rej.mixing_matrix_white_.shape == (n_ch, n_ch)

    # W @ A ≈ I in whitened space
    WA = res_rej.unmixing_matrix_white_ @ res_rej.mixing_matrix_white_
    np.testing.assert_allclose(WA, np.eye(n_ch), atol=1e-10, err_msg="W @ A != I after rejection")

    # Rejection must improve final LL on contaminated data
    ll_rej = res_rej.log_likelihood[-1]
    ll_no = res_no.log_likelihood[-1]
    assert ll_rej > ll_no, (
        f"Rejection did not improve LL: with_reject={ll_rej:.4f}, no_reject={ll_no:.4f}"
    )

    # The rejected-sample mask is exposed and flags the planted spikes.
    mask = res_rej.sample_mask_
    assert mask is not None and mask.shape == (n_samp,) and mask.dtype == bool
    rejected = ~mask
    spike_recall = rejected[spike_idx].mean()  # planted spikes that got flagged
    assert spike_recall > 0.5, f"only {spike_recall:.0%} of planted spikes flagged"
    assert rejected.mean() < 0.2, f"rejected too much clean data: {rejected.mean():.0%}"
    assert res_rej.n_rejected_ == int(rejected.sum())
    # do_reject=False leaves the mask unset (no rejection ran).
    assert res_no.sample_mask_ is None and res_no.n_rejected_ == 0


def test_rejection_threshold_rule():
    """_reject_threshold: one-sided lower cut at mean - rejsig*std over the current
    good set, and monotone (a rejected sample is never re-accepted)."""
    from amica.solver import _reject_threshold

    lls = np.zeros(100)
    lls[50] = -10.0  # one clear low outlier
    mask0 = np.ones(100, dtype=bool)

    mask1, n1 = _reject_threshold(lls, mask0, rejsig=3.0)
    assert n1 == 1 and not mask1[50] and mask1.sum() == 99
    # Re-running over the new good set never re-accepts the rejected sample.
    mask2, _ = _reject_threshold(lls, mask1, rejsig=3.0)
    assert not mask2[50]
    assert mask2.sum() <= mask1.sum()  # monotone non-increasing good count


def test_rejection_no_reacceptance():
    """A sample rejected early is never re-accepted across rounds (Fortran-faithful)."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(3)
    data = rng.laplace(size=(5, 1500))
    data[:, 100] *= 80.0  # one extreme spike → rejected early
    cfg = AmicaConfig(
        max_iter=40, do_newton=False, do_reject=True, rejstart=5, rejint=5, numrej=3, rejsig=3.0
    )
    res = Amica(cfg, random_state=0).fit(data)
    assert res.sample_mask_ is not None
    assert not res.sample_mask_[100]  # the spike is rejected and stays rejected


def test_reject_weight_anchor():
    """All-ones weight ≡ unweighted: a do_reject fit that rejects nothing (rejsig
    huge) reproduces a no-rejection fit, and do_reject=False leaves the mask unset."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(1)
    data = rng.laplace(size=(4, 1000))

    res_no = Amica(AmicaConfig(max_iter=20, do_newton=False), random_state=0).fit(data)
    assert res_no.sample_mask_ is None and res_no.n_rejected_ == 0

    # rejsig huge → threshold far below every LL → nothing rejected → weight all 1s.
    cfg_rej = AmicaConfig(
        max_iter=20, do_newton=False, do_reject=True, rejstart=5, rejint=5, numrej=3, rejsig=1e6
    )
    res_rej = Amica(cfg_rej, random_state=0).fit(data)
    assert res_rej.sample_mask_ is not None and res_rej.n_rejected_ == 0
    assert res_rej.sample_mask_.all()
    # The all-ones weight path matches the unweighted path (modulo XLA reassociation).
    np.testing.assert_allclose(
        res_rej.unmixing_matrix_white_,
        res_no.unmixing_matrix_white_,
        rtol=1e-5,
        atol=1e-7,
    )


def test_rejection_chunked_matches_fused():
    """Rejection mask + W agree between the full-batch (fused) and chunked paths."""
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(5)
    data = rng.laplace(size=(5, 1200))
    spikes = [50, 600, 900]
    data[:, spikes] *= 60.0
    common = {
        "max_iter": 40,
        "do_newton": False,
        "do_reject": True,
        "rejstart": 8,
        "rejint": 6,
        "numrej": 2,
        "rejsig": 3.0,
    }

    res_full = Amica(AmicaConfig(**common, chunk_size=None), random_state=0).fit(data)
    res_chunk = Amica(AmicaConfig(**common, chunk_size=137), random_state=0).fit(data)

    # Both paths reject the spikes.
    for s in spikes:
        assert not res_full.sample_mask_[s] and not res_chunk.sample_mask_[s]
    # Masks agree on the vast majority (borderline clean samples may differ by a few
    # due to fused-vs-chunked float rounding in the per-sample LL).
    agree = (res_full.sample_mask_ == res_chunk.sample_mask_).mean()
    assert agree > 0.99, f"chunked/fused masks agree on only {agree:.3%}"
    np.testing.assert_allclose(
        res_full.unmixing_matrix_white_,
        res_chunk.unmixing_matrix_white_,
        rtol=1e-3,
        atol=1e-5,
    )


def test_amica_wrapper(tiny_data):
    """Test the picard-compatible amica() wrapper function."""
    from amica.solver import amica

    # tiny_data is (n_channels, n_samples) — same as picard convention
    X = tiny_data

    K, W, Y, n_iter = amica(X, max_iter=2, return_n_iter=True, random_state=42)
    assert K is None
    assert W.shape == (4, 4)
    assert Y.shape == X.shape
    assert n_iter == 2

    K2, W2, _Y2 = amica(X, max_iter=1, return_n_iter=False)
    assert K2 is None
    assert W2.shape == (4, 4)


def test_picard_api_parity(tiny_data):
    """amica() return signature matches picard() exactly.

    Verifies that the same call MNE makes to picard() works identically
    with amica() — same X shape, same keyword args, same unpack pattern.
    Does not check numerical agreement (different algorithms).
    """
    picard_mod = pytest.importorskip("picard")
    from amica.solver import amica

    # MNE passes data[:, sel].T → (n_components, n_samples)
    X = tiny_data  # (4, 200)

    # Exact MNE calling pattern: _, W, _, n_iter = func(X, whiten=False, return_n_iter=True, ...)
    K_p, W_p, Y_p, n_iter_p = picard_mod.picard(
        X, whiten=False, return_n_iter=True, random_state=42
    )
    K_a, W_a, Y_a, n_iter_a = amica(X, whiten=False, return_n_iter=True, random_state=42)

    # K: picard returns None when whiten=False
    assert K_p is None
    assert K_a is None

    # W and Y shapes must match
    assert W_a.shape == W_p.shape
    assert Y_a.shape == Y_p.shape

    # n_iter: both ints > 0
    assert isinstance(n_iter_a, int) and n_iter_a > 0
    assert isinstance(n_iter_p, int) and n_iter_p > 0

    # MNE unpack pattern works without error
    _, W_mne, _, _ = amica(X, whiten=False, return_n_iter=True, random_state=42)
    assert W_mne.shape == (X.shape[0], X.shape[0])


def test_result_to_mne(tiny_data, monkeypatch):
    """Test AmicaResult.to_mne(info) method."""
    import sys
    from unittest.mock import MagicMock

    from amica import Amica, AmicaConfig

    # Mock MNE so we don't need real raw
    mne_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "mne", mne_mock)

    class MockICA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mne_prep_mock = MagicMock()
    mne_prep_mock.ICA = MockICA
    monkeypatch.setitem(sys.modules, "mne.preprocessing", mne_prep_mock)

    config = AmicaConfig(max_iter=2)
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)

    mock_info = {"ch_names": ["EEG1", "EEG2", "EEG3", "EEG4"]}
    ica = res.to_mne(mock_info)

    assert ica.method == "amica"
    assert ica.n_components_ == 4


def test_checkpoint_resume_bit_exact(tmp_path):
    """50 iters + save + resume 50 iters must be bit-exact to a single 100-iter run.

    Requires lratefact=rholratefact=1.0 so lrate is constant (not affected by LL
    history which differs between the two runs), do_newton=False (no Newton lrate
    ramp), and do_mean=False (c=0 throughout; c is not restored by init_params).
    """
    from amica import Amica, AmicaConfig

    rng = np.random.RandomState(0)
    n_ch, n_samp = 4, 2000
    S = rng.laplace(size=(n_ch, n_samp))
    A_true = rng.randn(n_ch, n_ch)
    data = (A_true @ S).astype(np.float64)

    shared = {
        "num_mix_comps": 2,
        "do_newton": False,
        "do_mean": False,
        "lratefact": 1.0,
        "rholratefact": 1.0,
    }

    # Run A: 100 iters straight through
    res_full = Amica(AmicaConfig(max_iter=100, **shared), random_state=42).fit(data)
    W_100 = res_full.unmixing_matrix_white_

    # Run B: 50 iters → save checkpoint
    ckpt = tmp_path / "ckpt"
    res_50 = Amica(
        AmicaConfig(max_iter=50, outdir=ckpt, writestep=50, **shared), random_state=42
    ).fit(data)

    # Resume from checkpoint for 50 more iters
    init_params = {
        "alpha": res_50.alpha_,
        "mu": res_50.mu_,
        "sbeta": res_50.sbeta_,
        "rho": res_50.rho_,
    }
    res_resumed = Amica(AmicaConfig(max_iter=50, **shared), random_state=42).fit(
        data,
        init_weights=res_50.unmixing_matrix_white_,
        init_params=init_params,
    )
    W_50_50 = res_resumed.unmixing_matrix_white_

    # JAX JIT or OS-level BLAS (macOS Accelerate) may reorder FP ops
    # causing small differences. atol=1e-6 is tight enough to catch
    # algorithmic bugs without failing due to cross-platform math.
    np.testing.assert_allclose(
        W_100,
        W_50_50,
        atol=1e-6,
        err_msg="50+50 checkpoint resume diverged from single 100-iter run",
    )


def test_solver_init_paths(tiny_data):
    """Test solver initialization paths with fixed init and initial params."""
    from amica import Amica, AmicaConfig

    # Test fix_init
    config1 = AmicaConfig(fix_init=True, max_iter=1)
    solver1 = Amica(config1, random_state=42)
    solver1.fit(tiny_data)

    # Test init_weights and init_params
    W_init = np.eye(4)
    params = {
        "alpha": np.ones((1, 4)),
        "mu": np.zeros((1, 4)),
        "beta": np.ones((1, 4)),
        "rho": np.ones((1, 4)),
    }
    config2 = AmicaConfig(max_iter=1)
    solver2 = Amica(config2, random_state=42)
    solver2.fit(tiny_data, init_weights=W_init, init_params=params)


# ---------------------------------------------------------------------------
# Backend parity: JAX-CPU vs NumPy-CPU
# ---------------------------------------------------------------------------

_BACKEND_PARITY_SCRIPT = """
import os, sys, numpy as np
backend = sys.argv[1]   # "jax" or "numpy"
data_path = sys.argv[2]
out_path = sys.argv[3]

if backend == "numpy":
    os.environ["AMICA_NO_JAX"] = "1"
else:
    os.environ.pop("AMICA_NO_JAX", None)
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

from amica import Amica, AmicaConfig

data = np.load(data_path)
config = AmicaConfig(max_iter=100, num_mix_comps=2, do_newton=False)
solver = Amica(config, random_state=42)
result = solver.fit(data)
np.save(out_path, result.unmixing_matrix_white_)
"""


def _run_backend(backend, data_path, out_path):
    env = os.environ.copy()
    env.pop("AMICA_NO_JAX", None)
    subprocess.run(
        [sys.executable, "-c", _BACKEND_PARITY_SCRIPT, backend, str(data_path), str(out_path)],
        check=True,
        env=env,
        capture_output=True,
    )


def _align_and_compare(W_ref, W_test):
    """Match rows by max-abs correlation; return max off-component residual."""
    n = W_ref.shape[0]
    # Normalise rows to unit norm for correlation
    W_r = W_ref / (np.linalg.norm(W_ref, axis=1, keepdims=True) + 1e-12)
    W_t = W_test / (np.linalg.norm(W_test, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(W_r @ W_t.T)  # (n, n) absolute cosine similarities
    used = set()
    max_sim = []
    for i in range(n):
        row = corr[i].copy()
        for j in used:
            row[j] = -1
        best = int(np.argmax(row))
        max_sim.append(corr[i, best])
        used.add(best)
    return np.min(max_sim)  # worst matched component similarity


@pytest.mark.slow
def test_backend_parity_jax_vs_numpy(tmp_path):
    """JAX-CPU and NumPy backends must converge to equivalent unmixing matrices.

    Uses synthetic Laplacian data where the mixing solution is well-determined.
    Checks that every matched component pair has cosine similarity > 0.99.
    Marks slow because it spawns two subprocesses running 100 iterations each.
    """
    rng = np.random.RandomState(0)
    n_src, n_samp = 4, 3000
    S = rng.laplace(size=(n_src, n_samp))
    A_true = rng.randn(n_src, n_src)
    data = (A_true @ S).astype(np.float64)

    data_path = tmp_path / "data.npy"
    np.save(data_path, data)

    jax_out = tmp_path / "W_jax.npy"
    numpy_out = tmp_path / "W_numpy.npy"

    _run_backend("jax", data_path, jax_out)
    _run_backend("numpy", data_path, numpy_out)

    W_jax = np.load(jax_out)
    W_numpy = np.load(numpy_out)

    min_sim = _align_and_compare(W_jax, W_numpy)
    assert min_sim > 0.99, (
        f"Worst component cosine similarity between JAX-CPU and NumPy: {min_sim:.4f} < 0.99"
    )
