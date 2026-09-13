"""Real-MNE tests for the multi-model ``AmicaICA`` parent object.

These use genuine ``mne.io.RawArray`` / ``mne.EpochsArray`` /
``mne.preprocessing.ICA`` objects rather than mocks: the point of the
models-as-views design is that each child really is an ordinary MNE ``ICA``,
which only a real object can demonstrate.
"""

from __future__ import annotations

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from jamica import AmicaICA  # noqa: E402

# The fit runs in float64 and the child transform is a re-association of the
# same products (W/s) @ (P @ (x - mu)) rather than W @ ((P @ x) / s), so the
# agreement is limited by float64 accumulation over the channel axis only.
ATOL_SOURCES = 1e-8
RTOL_SOURCES = 1e-6


def _make_raw(n_ch=6, tseg=1500, seed=0, offset=6.0, average_ref=False):
    """Two-regime EEG-like Raw: distinct mixing AND distinct centre per regime.

    The offset is what forces non-zero model centres ``c_h``; a zero-centre
    fixture would not exercise the ``pca_mean_`` fold at all.
    """
    rng = np.random.default_rng(seed)
    u = np.zeros(n_ch)
    u[0] = offset
    a1 = rng.standard_normal((n_ch, n_ch))
    a2 = rng.standard_normal((n_ch, n_ch))
    x1 = a1 @ rng.laplace(size=(n_ch, tseg)) + u[:, None]
    x2 = a2 @ rng.laplace(size=(n_ch, tseg)) - u[:, None]
    data = np.concatenate([x1, x2], axis=1) * 1e-6  # volts
    if average_ref:
        data = data - data.mean(axis=0, keepdims=True)  # rank n_ch - 1
    info = mne.create_info([f"EEG{i:03d}" for i in range(n_ch)], sfreq=100.0, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def _make_epochs(n_ch=6, n_epochs=8, n_times=200, seed=1):
    raw = _make_raw(n_ch=n_ch, tseg=(n_epochs * n_times) // 2, seed=seed)
    events = np.column_stack(
        [
            np.arange(0, n_epochs * n_times, n_times),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    ).astype(int)
    return mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=(n_times - 1) / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose="ERROR",
    )


def _native_sources(amica, model_idx):
    """AMICA's own transform for model h: ``W_h @ (x_white - c_h)``.

    Mirrors ``multimodel.compute_model_sources`` / the E-step definition at
    ``multimodel.py`` (``y_all = W @ (data_white - c[:, None])``).
    """
    prep = amica._prep
    w_h = amica._W_all[model_idx]
    c_h = amica._c_all[model_idx]
    return w_h @ (prep.data_for_amica - c_h[:, None])


@pytest.fixture(scope="module")
def fitted_mm():
    """A fitted 2-model AmicaICA plus the Raw it was fitted on."""
    raw = _make_raw()
    amica = AmicaICA(n_models=2, n_components=4, max_iter=80, random_state=0).fit(raw)
    return amica, raw


# ---------------------------------------------------------------------------
# 1 / 2. source equivalence against jamica's native transform
# ---------------------------------------------------------------------------


def test_single_model_source_equivalence():
    """M=1: the child's get_sources matches AMICA's own transform."""
    raw = _make_raw(seed=3)
    amica = AmicaICA(n_models=1, n_components=4, max_iter=60, random_state=0).fit(raw)

    native = _native_sources(amica, 0)
    got = amica.models_[0].get_sources(raw).get_data()
    assert np.allclose(got, native, rtol=RTOL_SOURCES, atol=ATOL_SOURCES)


def test_multimodel_source_equivalence_every_model(fitted_mm):
    """M>1: every child matches its own model, with a genuinely non-zero centre."""
    amica, raw = fitted_mm

    c = np.asarray(amica._c_all)
    assert np.max(np.abs(c)) > 1e-6, f"fixture failed to produce non-zero centres: {c}"

    for h in range(amica.n_models_):
        native = _native_sources(amica, h)
        got = amica.models_[h].get_sources(raw).get_data()
        assert np.allclose(got, native, rtol=RTOL_SOURCES, atol=ATOL_SOURCES), (
            f"model {h} sources disagree (max |diff| "
            f"{np.max(np.abs(got - native)):.3e}, |c_h|max {np.max(np.abs(c[h])):.3e})"
        )


def test_center_fold_actually_matters(fitted_mm):
    """Dropping the centre fold must change the sources.

    Without this, a bug that ignored ``c_h`` entirely would still pass the
    equivalence tests if the fixture happened to have small centres.
    """
    amica, raw = fitted_mm
    h = int(np.argmax(np.abs(amica._c_all).max(axis=1)))

    child = amica.models_[h]
    with_fold = child.get_sources(raw).get_data()

    import copy

    naive = copy.copy(child)
    naive.pca_mean_ = np.array(amica._prep.pca_mean, copy=True)  # unfolded
    without_fold = naive.get_sources(raw).get_data()

    assert not np.allclose(with_fold, without_fold, atol=1e-10)


# ---------------------------------------------------------------------------
# 3 / 4. reconstruction and residual PCA subspace
# ---------------------------------------------------------------------------


def test_apply_without_exclusions_reconstructs(fitted_mm):
    """A child with no exclusions must round-trip the data."""
    amica, raw = fitted_mm
    child = amica.models_[0]
    assert child.exclude == []

    cleaned = child.apply(raw.copy(), verbose="ERROR")
    assert np.allclose(cleaned.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12)


def test_residual_pca_subspace_preserved():
    """Components beyond n_components_ are PCA residual and must be untouched.

    MNE keeps the full ``pca_components_`` and restores the directions outside
    the ICA subspace. Cleaning with no exclusions must leave that residual at
    its original amplitude.
    """
    raw = _make_raw(n_ch=6, seed=5)
    amica = AmicaICA(n_models=2, n_components=3, max_iter=60, random_state=0).fit(raw)
    child = amica.models_[0]
    assert child.n_components_ == 3
    assert child.pca_components_.shape == (6, 6)

    prep = amica._prep
    resid = prep.pca_components[3:] @ (raw.get_data() / prep.pre_whitener - prep.pca_mean[:, None])

    cleaned = child.apply(raw.copy(), verbose="ERROR")
    resid_after = prep.pca_components[3:] @ (
        cleaned.get_data() / prep.pre_whitener - prep.pca_mean[:, None]
    )
    assert np.allclose(resid_after, resid, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# 5 / 6 / 8. child independence, caching, refit
# ---------------------------------------------------------------------------


def test_children_have_independent_exclude_and_labels(fitted_mm):
    amica, _raw = fitted_mm
    a, b = amica.models_[0], amica.models_[1]

    a.exclude = [0, 2]
    a.labels_["eog"] = [0]

    assert b.exclude == []
    assert b.labels_ == {}
    assert a.pca_mean_ is not b.pca_mean_

    a.exclude = []
    a.labels_ = {}


def test_repeated_access_returns_same_cached_child(fitted_mm):
    """Interactive review mutates ICA.exclude; that state must not vanish."""
    amica, _raw = fitted_mm
    first = amica.models_[1]
    first.exclude = [1]

    second = amica.models_[1]
    assert second is first
    assert second.exclude == [1]

    first.exclude = []


def test_refit_rebuilds_child_cache():
    raw = _make_raw(seed=7)
    amica = AmicaICA(n_models=2, n_components=4, max_iter=40, random_state=0)
    amica.fit(raw)
    before = amica.models_[0]
    before.exclude = [0]

    amica.fit(raw)
    after = amica.models_[0]
    assert after is not before
    assert after.exclude == [], "stale review state survived a refit"


def test_models_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = AmicaICA(n_models=2).models_


# ---------------------------------------------------------------------------
# 7. parent apply semantics
# ---------------------------------------------------------------------------


def test_parent_apply_raises_for_multimodel(fitted_mm):
    amica, raw = fitted_mm
    with pytest.raises(ValueError, match="model_idx"):
        amica.apply(raw.copy())


def test_parent_apply_with_model_idx_matches_child(fitted_mm):
    amica, raw = fitted_mm
    amica.models_[1].exclude = [0]
    try:
        via_parent = amica.apply(raw.copy(), model_idx=1, verbose="ERROR").get_data()
        via_child = amica.models_[1].apply(raw.copy(), verbose="ERROR").get_data()
        assert np.allclose(via_parent, via_child)
    finally:
        amica.models_[1].exclude = []


def test_parent_apply_delegates_for_single_model():
    raw = _make_raw(seed=11)
    amica = AmicaICA(n_models=1, n_components=4, max_iter=40, random_state=0).fit(raw)
    cleaned = amica.apply(raw.copy(), verbose="ERROR")
    assert np.allclose(cleaned.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12)


def test_parent_apply_rejects_bad_model_idx(fitted_mm):
    amica, raw = fitted_mm
    with pytest.raises(IndexError):
        amica.apply(raw.copy(), model_idx=99)


# ---------------------------------------------------------------------------
# 9 / 10. posterior shapes and weights
# ---------------------------------------------------------------------------


def test_raw_posterior_shape_and_normalisation(fitted_mm):
    amica, raw = fitted_mm
    post = amica.model_posteriors_
    assert post.shape == (2, len(raw.times))
    assert np.allclose(post.sum(axis=0), 1.0, atol=1e-8)


def test_model_weights_are_priors_not_posteriors(fitted_mm):
    amica, _raw = fitted_mm
    w = amica.model_weights_
    assert w.shape == (2,)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)


def test_epochs_posterior_shape():
    epochs = _make_epochs()
    amica = AmicaICA(n_models=2, n_components=4, max_iter=40, random_state=0).fit(epochs)
    post = amica.model_posteriors_
    assert post.shape == (2, len(epochs), len(epochs.times))
    assert np.allclose(post.sum(axis=0), 1.0, atol=1e-8)


def test_get_model_probabilities_reproduces_fit_posteriors(fitted_mm):
    """On the fit data, recomputed probabilities must match the stored ones.

    This validates the parameter conventions used when calling back into the
    solver (notably that ``sbeta_`` is forwarded as ``beta``, and that the
    shared ``log_det_sphere`` cancels in the softmax over models).
    """
    amica, raw = fitted_mm
    recomputed = amica.get_model_probabilities(raw)
    assert recomputed.shape == amica.model_posteriors_.shape
    assert np.allclose(recomputed, amica.model_posteriors_, atol=1e-6)


def test_get_model_probabilities_on_new_data(fitted_mm):
    amica, _raw = fitted_mm
    other = _make_raw(seed=99)
    post = amica.get_model_probabilities(other)
    assert post.shape == (2, len(other.times))
    assert np.allclose(post.sum(axis=0), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 11. rank-deficient data
# ---------------------------------------------------------------------------


def test_rank_deficient_average_referenced_data():
    """Average referencing drops rank by one; the fit must respect that."""
    raw = _make_raw(n_ch=6, seed=13, average_ref=True)
    amica = AmicaICA(n_models=2, n_components=4, max_iter=60, random_state=0).fit(raw)

    assert amica.n_components_ == 4
    for h in range(2):
        native = _native_sources(amica, h)
        got = amica.models_[h].get_sources(raw).get_data()
        assert np.allclose(got, native, rtol=RTOL_SOURCES, atol=ATOL_SOURCES)

    cleaned = amica.models_[0].apply(raw.copy(), verbose="ERROR")
    assert np.allclose(cleaned.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12)


def test_requesting_more_components_than_rank_raises():
    raw = _make_raw(n_ch=6, seed=17, average_ref=True)
    with pytest.raises(ValueError, match="rank"):
        AmicaICA(n_models=2, n_components=6, max_iter=10, random_state=0).fit(raw)


# ---------------------------------------------------------------------------
# construction contract
# ---------------------------------------------------------------------------


def test_children_are_real_mne_ica_objects(fitted_mm):
    amica, _raw = fitted_mm
    for child in amica.models_:
        assert isinstance(child, mne.preprocessing.ICA)
        assert child.method == "amica"
    assert not isinstance(amica, mne.preprocessing.ICA)


def test_child_model_index_metadata(fitted_mm):
    amica, _raw = fitted_mm
    assert [c._amica_model_index for c in amica.models_] == [0, 1]


def test_invalid_n_models_rejected():
    with pytest.raises(ValueError, match="n_models"):
        AmicaICA(n_models=0)


# ---------------------------------------------------------------------------
# Posterior timeline semantics: decimation, rejection, epochs
#
# model_posteriors_ is always on the ORIGINAL input grid. Decimation changes
# which samples drove the optimisation, not the reported shape.
# ---------------------------------------------------------------------------


def test_raw_decim_keeps_full_posterior_grid():
    """decim must not shorten model_posteriors_."""
    raw = _make_raw(seed=21)
    amica = AmicaICA(n_models=2, n_components=4, max_iter=40, random_state=0, decim=2).fit(raw)

    assert amica.model_posteriors_.shape == (2, len(raw.times))
    assert np.isfinite(amica.model_posteriors_).all(), "decim must not introduce NaN"
    assert np.allclose(amica.model_posteriors_.sum(axis=0), 1.0, atol=1e-8)


def test_raw_decim_mask_is_all_true():
    """Anti-aliased decimation filters every sample into the retained ones.

    ``scipy.signal.decimate`` applies an FIR filter before downsampling, so no
    input sample is truly absent from the optimisation; marking the dropped
    grid points False would misdescribe the fit.
    """
    raw = _make_raw(seed=22)
    amica = AmicaICA(n_models=2, n_components=4, max_iter=40, random_state=0, decim=2).fit(raw)

    assert amica.fit_sample_mask_.shape == (len(raw.times),)
    assert amica.fit_sample_mask_.all()


def test_raw_amplitude_rejection_masks_and_nans_excluded_samples():
    """Rejected epochs are False in the mask and NaN in the posteriors."""
    raw = _make_raw(n_ch=6, tseg=800, seed=23)

    # Make one second grossly bad so reject drops exactly that epoch.
    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])
    data[:, 3 * sfreq : 4 * sfreq] *= 500.0
    raw._data = data

    amica = AmicaICA(
        n_models=2,
        n_components=4,
        max_iter=40,
        random_state=0,
        reject={"eeg": 5e-4},
    ).fit(raw)

    mask = amica.fit_sample_mask_
    post = amica.model_posteriors_

    assert mask.shape == (len(raw.times),)
    assert post.shape == (2, len(raw.times))
    assert not mask.all(), "the bad second should have been rejected"

    # Excluded samples: NaN posteriors. Included samples: finite and normalised.
    assert np.isnan(post[:, ~mask]).all()
    assert np.isfinite(post[:, mask]).all()
    assert np.allclose(post[:, mask].sum(axis=0), 1.0, atol=1e-8)

    # The rejected region must overlap the second we corrupted.
    assert not mask[3 * sfreq : 4 * sfreq].all()


def test_raw_bad_annotations_masked_and_excluded_from_fit():
    """BAD-annotated samples are False in the mask and do not enter the fit."""
    raw = _make_raw(n_ch=6, tseg=800, seed=23)
    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])
    data[:, 3 * sfreq : 4 * sfreq] *= 500.0
    raw._data = data
    raw.set_annotations(mne.Annotations(onset=3.0, duration=1.0, description="BAD_segment"))
    kwargs = {"n_models": 2, "n_components": 4, "max_iter": 40, "random_state": 0}

    amica = AmicaICA(**kwargs).fit(raw)
    mask = amica.fit_sample_mask_
    assert not mask[3 * sfreq : 4 * sfreq].any()
    assert mask.sum() == amica._prep.n_samples == len(raw.times) - sfreq

    good = mne.io.RawArray(raw.get_data(reject_by_annotation="omit"), raw.info, verbose="ERROR")
    np.testing.assert_allclose(amica._W_all, AmicaICA(**kwargs).fit(good)._W_all)

    kept = AmicaICA(**kwargs, reject_by_annotation=False).fit(raw)
    assert kept.fit_sample_mask_.all()
    assert kept._prep.n_samples == len(raw.times)


def test_epochs_decim_keeps_public_epoch_shape():
    """Epochs + decim must still report (M, n_epochs, n_times)."""
    epochs = _make_epochs(seed=24)
    amica = AmicaICA(n_models=2, n_components=4, max_iter=40, random_state=0, decim=2).fit(epochs)

    assert amica.model_posteriors_.shape == (2, len(epochs), len(epochs.times))
    assert amica.fit_sample_mask_.shape == (len(epochs), len(epochs.times))
    assert amica.fit_sample_mask_.all()
    assert np.allclose(amica.model_posteriors_.sum(axis=0), 1.0, atol=1e-8)


def test_posteriors_match_recomputation_on_fit_data(fitted_mm):
    """The stored posteriors are exactly what get_model_probabilities returns."""
    amica, raw = fitted_mm
    assert np.allclose(amica.model_posteriors_, amica.get_model_probabilities(raw), atol=1e-12)


# ---------------------------------------------------------------------------
# Shared vs model-specific child state
# ---------------------------------------------------------------------------


def test_child_state_isolation_shared_vs_model_specific(fitted_mm):
    """Model-specific arrays are per-child; preprocessing arrays are shared.

    Mutating one child's model-specific state must not reach a sibling. The
    shared PCA arrays are identical objects by design -- they describe the one
    preprocessing pipeline, not anything per-model.
    """
    amica, _raw = fitted_mm
    a, b = amica.models_[0], amica.models_[1]

    # model-specific: distinct objects
    assert a.pca_mean_ is not b.pca_mean_
    assert a.unmixing_matrix_ is not b.unmixing_matrix_
    assert a.mixing_matrix_ is not b.mixing_matrix_
    assert a.exclude is not b.exclude
    assert a.labels_ is not b.labels_

    # shared preprocessing: same object, and describing the shared PCA
    assert a.pca_components_ is b.pca_components_
    assert a.pre_whitener_ is b.pre_whitener_
    assert a.pca_explained_variance_ is b.pca_explained_variance_

    # mutating one child's model-specific state leaves the sibling untouched
    b_mean_before = b.pca_mean_.copy()
    b_unmix_before = b.unmixing_matrix_.copy()
    a_mean, a_unmix = a.pca_mean_, a.unmixing_matrix_
    try:
        a.pca_mean_ = a_mean + 1.0
        a.unmixing_matrix_ = a_unmix * 2.0
        a.exclude = [0]
        a.labels_["eog"] = [1]

        assert np.array_equal(b.pca_mean_, b_mean_before)
        assert np.array_equal(b.unmixing_matrix_, b_unmix_before)
        assert b.exclude == []
        assert b.labels_ == {}
    finally:
        # the fixture is module-scoped; leave it as found
        a.pca_mean_, a.unmixing_matrix_ = a_mean, a_unmix
        a.exclude, a.labels_ = [], {}


# ---------------------------------------------------------------------------
# Persistence: HDF5 container + per-model FIF export
# ---------------------------------------------------------------------------

h5io = pytest.importorskip("h5io")

from jamica import read_amica_ica  # noqa: E402


def test_save_load_roundtrip(fitted_mm, tmp_path):
    """Everything that defines the fit survives a save/load cycle."""
    amica, _raw = fitted_mm
    fname = tmp_path / "fit.h5"
    amica.save(fname)
    back = read_amica_ica(fname)

    assert back.n_models_ == amica.n_models_
    assert back.n_components_ == amica.n_components_
    assert back.n_iter_ == amica.n_iter_
    assert back.ch_names == amica.ch_names
    assert np.allclose(back.model_weights_, amica.model_weights_)
    assert np.allclose(back.model_posteriors_, amica.model_posteriors_, equal_nan=True)
    assert np.array_equal(back.fit_sample_mask_, amica.fit_sample_mask_)
    assert np.allclose(back._W_all, amica._W_all)
    assert np.allclose(back._c_all, amica._c_all)


def test_loaded_children_produce_identical_sources(fitted_mm, tmp_path):
    """A restored model must transform data exactly as the original did."""
    amica, raw = fitted_mm
    fname = tmp_path / "fit.h5"
    amica.save(fname)
    back = read_amica_ica(fname)

    for h in range(amica.n_models_):
        want = amica.models_[h].get_sources(raw).get_data()
        got = back.models_[h].get_sources(raw).get_data()
        assert np.allclose(got, want, rtol=RTOL_SOURCES, atol=ATOL_SOURCES)
        # the per-model centre fold must survive too
        assert np.allclose(back.models_[h].pca_mean_, amica.models_[h].pca_mean_)


def test_save_load_preserves_review_state(tmp_path):
    """exclude / labels_ set during review are part of the fit, not scratch."""
    raw = _make_raw(seed=31)
    amica = AmicaICA(n_models=2, n_components=4, max_iter=40, random_state=0).fit(raw)
    amica.models_[0].exclude = [1, 3]
    amica.models_[1].labels_["eog"] = [0]

    fname = tmp_path / "fit.h5"
    amica.save(fname)
    back = read_amica_ica(fname)

    assert back.models_[0].exclude == [1, 3]
    assert back.models_[1].labels_ == {"eog": [0]}
    assert back.models_[1].exclude == []


def test_save_refuses_to_clobber(fitted_mm, tmp_path):
    amica, _raw = fitted_mm
    fname = tmp_path / "fit.h5"
    amica.save(fname)
    with pytest.raises(FileExistsError):
        amica.save(fname)
    amica.save(fname, overwrite=True)


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        AmicaICA(n_models=2).save(tmp_path / "x.h5")


def test_export_model_fifs_readable_by_plain_mne(fitted_mm, tmp_path):
    """The exported files must be ordinary MNE ICA objects.

    Read back through ``mne.preprocessing.read_ica`` only -- nothing from
    jamica is involved -- so a decomposition never becomes readable solely
    through this package.
    """
    amica, raw = fitted_mm
    written = amica.export_model_fifs(tmp_path / "sub-01-ica.fif")

    assert len(written) == amica.n_models_
    assert [p.name for p in written] == [
        f"sub-01-model-{h}-ica.fif" for h in range(amica.n_models_)
    ]

    for h, path in enumerate(written):
        loaded = mne.preprocessing.read_ica(path, verbose="ERROR")
        assert isinstance(loaded, mne.preprocessing.ICA)
        assert loaded.n_components_ == amica.n_components_
        want = amica.models_[h].get_sources(raw).get_data()
        got = loaded.get_sources(raw).get_data()
        assert np.allclose(got, want, rtol=1e-6, atol=1e-10)


def test_export_model_fifs_requires_ica_suffix(fitted_mm, tmp_path):
    """MNE refuses to read a file that is not named -ica.fif, so fail early."""
    amica, _raw = fitted_mm
    with pytest.raises(ValueError, match=r"-ica\.fif"):
        amica.export_model_fifs(tmp_path / "wrong-name.fif")


def test_rejects_unknown_file_format(fitted_mm, tmp_path):
    """A file from a future format version must fail loudly, not half-load."""
    amica, _raw = fitted_mm
    fname = tmp_path / "fit.h5"
    amica.save(fname)

    state = amica._as_state()
    state["jamica_format"] = 99
    bogus = tmp_path / "future.h5"
    h5io.write_hdf5(bogus, state, overwrite=True, title="jamica")

    with pytest.raises(ValueError, match="format"):
        read_amica_ica(bogus)
