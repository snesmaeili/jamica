"""Real MNE-Python integration tests.

Requires MNE-Python installed; all tests skip automatically if absent.
Companion to test_mne_integration.py (mocked unit tests for internal helpers).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mne():
    """Skip entire module if MNE-Python is not installed."""
    try:
        import mne as _mne

        return _mne
    except ImportError:
        pytest.skip("MNE-Python not installed")


def _make_raw(mne, n_ch=4, n_samp=1000, seed=42):
    rng = np.random.RandomState(seed)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_ch)],
        sfreq=256,
        ch_types="eeg",
    )
    return mne.io.RawArray(rng.randn(n_ch, n_samp) * 1e-6, info)


def test_fit_ica_on_raw(mne):
    """fit_ica produces a working MNE ICA object; get_sources() works."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=8, n_samp=2000)
    ica = fit_ica(
        raw,
        n_components=4,
        max_iter=20,
        num_mix=2,
        random_state=42,
        fit_params={"do_newton": False},
    )

    assert ica.n_components_ == 4
    assert ica.method == "amica"
    assert ica.unmixing_matrix_ is not None
    assert ica.get_sources(raw).get_data().shape[0] == 4


def test_fit_ica_omits_bad_annotations(mne):
    """fit_ica leaves BAD-annotated samples out of the fit, like ICA.fit."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=6, n_samp=3000)
    raw.set_annotations(mne.Annotations(onset=4.0, duration=2.0, description="BAD_segment"))
    good = mne.io.RawArray(raw.get_data(reject_by_annotation="omit"), raw.info)
    params = {
        "n_components": 3,
        "max_iter": 20,
        "num_mix": 2,
        "random_state": 42,
        "fit_params": {"do_newton": False},
    }

    ica = fit_ica(raw, **params)
    assert ica.n_samples_ == len(good.times) < len(raw.times)
    np.testing.assert_allclose(ica.unmixing_matrix_, fit_ica(good, **params).unmixing_matrix_)
    assert fit_ica(raw, reject_by_annotation=False, **params).n_samples_ == len(raw.times)


def test_direct_vs_shim_sources_correlated(mne):
    """Two identical fit_ica calls produce highly correlated sources."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=6, n_samp=3000)
    params = {
        "n_components": 3,
        "max_iter": 30,
        "num_mix": 2,
        "random_state": 42,
        "fit_params": {"do_newton": False},
    }
    ica_a = fit_ica(raw.copy(), **params)
    ica_b = fit_ica(raw.copy(), **params)

    src_a = ica_a.get_sources(raw).get_data()
    src_b = ica_b.get_sources(raw).get_data()

    corr = np.abs(np.corrcoef(src_a, src_b)[:3, 3:])
    max_corrs = np.max(corr, axis=1)
    assert np.all(max_corrs > 0.9), f"Source correlation too low: {max_corrs}"


def test_multi_model_now_supported(mne):
    """fit_ica() with num_models > 1 is now supported (returns a multi-model fit);
    the detailed exposure is checked in test_fit_ica_multimodel_mne_exposure."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=6, n_samp=3000)
    ica = fit_ica(
        raw, n_components=4, max_iter=20, fit_params={"num_models": 2, "do_newton": False}
    )
    assert np.asarray(ica.amica_result_.unmixing_matrix_white_).shape[0] == 2


def test_amica_result_attached(mne):
    """fit_ica() attaches amica_result_ to the ICA object."""
    from jamica import fit_ica

    raw = _make_raw(mne)
    ica = fit_ica(raw, n_components=2, max_iter=10, fit_params={"do_newton": False})
    assert hasattr(ica, "amica_result_")
    assert ica.amica_result_ is not None


def test_fit_ica_sample_rejection(mne):
    """fit_ica(fit_params={do_reject:True}) runs AMICA likelihood-based sample
    rejection and exposes the mask on ica.amica_result_; planted spikes are flagged
    (the mask indexes the fit-input samples)."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=6, n_samp=3000)
    data = raw.get_data()
    rng = np.random.RandomState(11)
    spike_idx = rng.choice(raw.n_times, size=raw.n_times // 20, replace=False)
    data[:, spike_idx] *= 40.0
    raw_c = mne.io.RawArray(data, raw.info)

    ica = fit_ica(
        raw_c,
        n_components=4,
        max_iter=40,
        fit_params={
            "do_reject": True,
            "rejstart": 8,
            "rejint": 5,
            "numrej": 3,
            "rejsig": 3.0,
            "do_newton": False,
        },
    )
    mask = ica.amica_result_.sample_mask_
    assert mask is not None and mask.shape == (raw.n_times,) and mask.dtype == bool
    assert ica.amica_result_.n_rejected_ == int((~mask).sum()) > 0
    assert (~mask)[spike_idx].mean() > 0.5  # most planted spikes flagged


def test_fit_ica_multimodel_rejection(mne):
    """fit_ica(num_models>1, do_reject) runs M>1 likelihood rejection through MNE:
    one global mask is exposed on amica_result_ and the primary ICA still works."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=6, n_samp=3000)
    data = raw.get_data()
    rng = np.random.RandomState(13)
    spike_idx = rng.choice(raw.n_times, size=raw.n_times // 25, replace=False)
    data[:, spike_idx] *= 40.0
    raw_c = mne.io.RawArray(data, raw.info)

    ica = fit_ica(
        raw_c,
        n_components=4,
        max_iter=40,
        fit_params={
            "num_models": 2,
            "do_reject": True,
            "rejstart": 8,
            "rejint": 5,
            "numrej": 2,
            "rejsig": 3.0,
            "do_newton": False,
        },
    )
    mask = ica.amica_result_.sample_mask_
    assert mask is not None and mask.shape == (raw.n_times,)
    assert ica.amica_result_.n_rejected_ > 0
    assert (~mask)[spike_idx].mean() > 0.5  # spikes flagged via the mixture-LL threshold
    assert np.asarray(ica.amica_result_.unmixing_matrix_white_).shape == (2, 4, 4)
    assert ica.get_sources(raw_c).get_data().shape[0] == 4  # primary model ICA works


def test_apply_preserves_shape(mne):
    """ica.apply() preserves data shape; full-rank round-trip is machine-precise."""
    from jamica import fit_ica

    rng = np.random.RandomState(42)
    n_ch, n_samp = 4, 1000
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_ch)],
        sfreq=256,
        ch_types="eeg",
    )
    data = rng.randn(n_ch, n_samp) * 1e-6
    raw = mne.io.RawArray(data, info)

    ica = fit_ica(raw, n_components=n_ch, max_iter=10, fit_params={"do_newton": False})
    assert ica.apply(raw.copy()).get_data().shape == data.shape

    recon = ica.apply(raw.copy(), exclude=[]).get_data()
    fro_norm_rel = np.linalg.norm(data - recon, "fro") / np.linalg.norm(data, "fro")
    assert fro_norm_rel < 1e-12, f"MNE apply() round-trip Frobenius residual: {fro_norm_rel:.2e}"


def test_apply_after_channel_reorder(mne):
    """apply() matches components to channels by name, so a reordered Raw still
    reconstructs to machine precision (channel-reordering conformance)."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=5, n_samp=1500)
    ica = fit_ica(raw, n_components=5, max_iter=10, fit_params={"do_newton": False})

    reordered = raw.copy().reorder_channels(list(reversed(raw.ch_names)))
    out = ica.apply(reordered, exclude=[]).copy().reorder_channels(raw.ch_names)
    fro_norm_rel = np.linalg.norm(raw.get_data() - out.get_data(), "fro") / np.linalg.norm(
        raw.get_data(), "fro"
    )
    assert fro_norm_rel < 1e-9, f"reordered apply round-trip residual: {fro_norm_rel:.2e}"


def test_fit_rank_deficient_input(mne):
    """A rank-deficient Raw (one duplicated channel) fits at the reduced rank
    without producing non-finite sources (rank-deficiency conformance)."""
    from jamica import fit_ica

    rng = np.random.RandomState(0)
    base = rng.randn(4, 2000) * 1e-6
    data = np.vstack([base, base[0:1]])  # 5 channels, true rank 4
    info = mne.create_info([f"EEG{i:03d}" for i in range(5)], sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    ica = fit_ica(raw, n_components=4, max_iter=15, fit_params={"do_newton": False})
    assert ica.n_components_ == 4
    src = ica.get_sources(raw).get_data()
    assert src.shape[0] == 4
    assert np.all(np.isfinite(src))


@pytest.mark.filterwarnings("ignore")
def test_iclabel_interop(mne):
    """The fitted ICA object is accepted by mne-icalabel and returns one label
    per component (ICLabel interoperability conformance; skips if not installed).

    mne-icalabel emits a RuntimeWarning that ICLabel was tuned for extended
    infomax rather than AMICA -- a documented caveat, not an interop failure --
    so warnings are ignored here; the test asserts only that it runs and returns
    one label per component."""
    try:
        from mne_icalabel import label_components
    except ImportError:
        pytest.skip("mne-icalabel not installed")
    from jamica import fit_ica

    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
    rng = np.random.RandomState(7)
    info = mne.create_info(names, sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(rng.randn(len(names), 4000) * 1e-6, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.filter(1.0, 100.0, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    ica = fit_ica(raw, n_components=7, max_iter=20, fit_params={"do_newton": False})
    result = label_components(raw, ica, method="iclabel")
    assert len(result["labels"]) == ica.n_components_


def test_fit_ica_on_epochs(mne):
    """fit_ica accepts an Epochs object and decomposes it (Epochs conformance)."""
    from jamica import fit_ica

    raw = _make_raw(mne, n_ch=6, n_samp=6000)
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=0.99, baseline=None, preload=True)

    ica = fit_ica(epochs, n_components=4, max_iter=15, fit_params={"do_newton": False})
    assert ica.n_components_ == 4
    src = ica.get_sources(epochs).get_data()
    assert src.shape[1] == 4 and np.all(np.isfinite(src))


def test_fit_ica_multimodel_mne_exposure(mne):
    """fit_ica(num_models>1) returns the primary (highest-weight) model's ICA with
    the full multi-model AmicaResult attached, and get_model_ica materialises any
    model so standard MNE ops keep working (multi-model MNE conformance)."""
    from jamica import fit_ica, get_model_ica

    raw = _make_raw(mne, n_ch=8, n_samp=4000)
    ica = fit_ica(
        raw, n_components=6, max_iter=40, fit_params={"num_models": 2, "do_newton": False}
    )

    # Full multi-model result is attached.
    W = np.asarray(ica.amica_result_.unmixing_matrix_white_)
    assert W.shape == (2, 6, 6)
    assert np.asarray(ica.amica_result_.model_posteriors_).shape == (2, raw.n_times)

    # Primary ICA = highest-gm model; standard MNE ops work.
    assert ica.unmixing_matrix_.shape == (6, 6)
    assert ica._amica_model_index == int(np.argmax(np.asarray(ica.amica_result_.gm_)))
    assert ica.get_sources(raw).get_data().shape[0] == 6
    assert ica.apply(raw.copy(), exclude=[]).get_data().shape == (8, raw.n_times)

    # Any model is retrievable and distinct.
    other = get_model_ica(ica, 1 - ica._amica_model_index)
    assert other._amica_model_index != ica._amica_model_index
    assert not np.allclose(other.unmixing_matrix_, ica.unmixing_matrix_)
    assert np.all(np.isfinite(other.get_sources(raw).get_data()))


def _make_avg_ref_raw(mne, n_ch=12, n_samp=4000, seed=7):
    """Average-referenced EEG: rank is exactly n_ch - 1."""
    rng = np.random.RandomState(seed)
    sources = rng.laplace(size=(n_ch, n_samp))
    data = (rng.randn(n_ch, n_ch) @ sources) * 1e-6
    info = mne.create_info([f"EEG{i:03d}" for i in range(n_ch)], sfreq=250, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def test_fit_ica_default_rank_deficient_reconstructs(mne):
    """fit_ica() with the *default* n_components on average-referenced EEG must
    return an ICA whose apply() reconstructs the data.

    Regression test. Previously the default kept all n_ch components on rank
    n_ch-1 data; the trailing PCA direction (std ~1e-16) inflated one column of
    W by ~1e16, np.linalg.pinv then discarded every legitimate singular value,
    and apply() returned near-zero data with no warning.
    """
    from jamica import fit_ica

    raw = _make_avg_ref_raw(mne)
    n_ch = len(raw.ch_names)

    with pytest.warns(RuntimeWarning, match="Estimated data rank"):
        ica = fit_ica(raw, max_iter=15, random_state=0, fit_params={"do_newton": False})

    # the degenerate direction is dropped, not fitted
    assert ica.n_components_ == n_ch - 1

    # pinv must retain every singular value of the unmixing matrix
    sv = np.linalg.svd(ica.unmixing_matrix_, compute_uv=False)
    cutoff = np.finfo(ica.unmixing_matrix_.dtype).eps * max(ica.unmixing_matrix_.shape) * sv.max()
    assert np.all(sv > cutoff), f"pinv would discard {(sv <= cutoff).sum()} of {sv.size} components"

    # apply() with nothing excluded is the identity, to numerical precision
    out = ica.apply(raw.copy(), exclude=[], verbose="ERROR")
    x_in, x_out = raw.get_data(), out.get_data()
    rel = np.linalg.norm(x_out - x_in) / np.linalg.norm(x_in)
    assert rel < 1e-9, f"reconstruction residual {rel:.3e} (data destroyed?)"


def test_fit_ica_apply_with_nonempty_exclude(mne):
    """apply() with a non-empty exclude removes one component's contribution and
    leaves the rest of the recording intact.

    Regression test: every other apply() call in this suite passes exclude=[],
    so the component-removal path — the operation practitioners actually use —
    was never exercised.
    """
    from jamica import fit_ica

    raw = _make_avg_ref_raw(mne)
    with pytest.warns(RuntimeWarning, match="Estimated data rank"):
        ica = fit_ica(raw, max_iter=15, random_state=0, fit_params={"do_newton": False})

    out = ica.apply(raw.copy(), exclude=[0], verbose="ERROR")
    x_in, x_out = raw.get_data(), out.get_data()

    assert np.all(np.isfinite(x_out))
    # something was removed ...
    assert not np.allclose(x_out, x_in)
    # ... but the recording survived: amplitude stays the same order of magnitude
    ratio = x_out.std() / x_in.std()
    assert 0.3 < ratio < 1.05, f"amplitude ratio {ratio:.3f} after removing 1 component"


def test_fit_ica_explicit_n_components_above_rank_raises(mne):
    """An explicit n_components larger than the data rank is an error, not something
    to silently reinterpret. Regression test for the review finding that the first
    version of the rank guard overrode explicit user requests."""
    from jamica import fit_ica

    raw = _make_avg_ref_raw(mne)
    n_ch = len(raw.ch_names)
    with pytest.raises(ValueError, match="exceeds the estimated rank"):
        fit_ica(raw, n_components=n_ch, max_iter=5, fit_params={"do_newton": False})

    # at or below the rank it proceeds without complaint
    ica = fit_ica(raw, n_components=n_ch - 1, max_iter=5, fit_params={"do_newton": False})
    assert ica.n_components_ == n_ch - 1


def test_rank_guard_keeps_genuine_low_variance_component(mne):
    """A real but very weak independent mode must survive the rank guard.

    Two orthogonal spatial modes whose standard deviations differ by 1e-9 — far below
    a sqrt(eps) threshold but far above the SVD rank tolerance. Both are genuine, so
    both must be fitted; only the average-reference null space may be dropped.
    """
    from jamica import fit_ica

    rng = np.random.RandomState(3)
    n_samp = 4000
    u1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    u2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    z1 = rng.laplace(size=n_samp)
    z1 -= z1.mean()
    z2 = rng.laplace(size=n_samp) * 1e-9
    z2 -= z2.mean()
    data = (np.outer(u1, z1) + np.outer(u2, z2)) * 1e-6  # rank 2 of 3, sums to zero

    info = mne.create_info(["EEG000", "EEG001", "EEG002"], sfreq=250, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    with pytest.warns(RuntimeWarning, match="Estimated data rank"):
        ica = fit_ica(raw, max_iter=5, random_state=0, fit_params={"do_newton": False})

    assert ica.n_components_ == 2, (
        f"kept {ica.n_components_} components; the 1e-9 mode is genuine and must not be dropped"
    )


@pytest.mark.parametrize("bad", [0, -1, 1])
def test_fit_ica_rejects_out_of_range_n_components(mne, bad):
    """n_components below 2 must fail with one clear message.

    Regression test: 0 reached the rank check and raised IndexError on an empty
    array, -1 was silently reinterpreted by negative-slice semantics into
    n_channels-1 components, and 1 was reported as "estimated data rank is 1"
    even on full-rank data, misattributing an explicit request to data quality.
    """
    from jamica import fit_ica

    rng = np.random.RandomState(0)
    data = (rng.randn(6, 6) @ rng.laplace(size=(6, 3000))) * 1e-6
    info = mne.create_info([f"EEG{i:03d}" for i in range(6)], sfreq=250, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    with pytest.raises(ValueError, match="at least 2 components"):
        fit_ica(raw, n_components=bad, max_iter=3, fit_params={"do_newton": False})
