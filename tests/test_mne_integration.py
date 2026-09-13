"""Tests for mne_integration module using mocked MNE and AMICA objects."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jamica import mne_integration
from jamica.solver import AmicaResult


class MockInfo(dict):
    """Mock MNE Info object."""

    def __init__(self, n_channels=4):
        super().__init__()
        self["ch_names"] = [f"EEG{i}" for i in range(n_channels)]
        self["sfreq"] = 256.0
        self["bads"] = []
        self["chs"] = [{"kind": 2} for _ in range(n_channels)]  # 2 = FIFFV_EEG_CH


class MockRaw:
    """Mock MNE Raw object."""

    def __init__(self, data=None):
        if data is None:
            data = np.random.randn(4, 100)
        self.data = data
        self.info = MockInfo(data.shape[0])

    def get_data(self, picks=None, reject_by_annotation=None):
        if picks is not None:
            return self.data[picks]
        return self.data


class MockEpochs:
    """Mock MNE Epochs object."""

    def __init__(self, data=None):
        if data is None:
            # (n_epochs, n_channels, n_samples)
            data = np.random.randn(3, 4, 100)
        self.data = data
        self.info = MockInfo(data.shape[1])

    def get_data(self, picks=None):
        if picks is not None:
            return self.data[:, picks, :]
        return self.data


@pytest.fixture(autouse=True)
def mock_mne_modules(monkeypatch):
    """Mock all MNE modules to avoid real imports and scipy errors."""
    mne_mock = MagicMock()
    mne_mock.channel_type.return_value = "eeg"
    mne_mock.pick_types.return_value = [0, 1, 2, 3]
    mne_mock.pick_info.return_value = MockInfo()

    mne_io_mock = MagicMock()
    mne_io_mock.BaseRaw = MockRaw

    mne_epochs_mock = MagicMock()
    mne_epochs_mock.BaseEpochs = MockEpochs

    class MockICA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.exclude = []

    mne_prep_mock = MagicMock()
    mne_prep_mock.ICA = MockICA

    monkeypatch.setitem(sys.modules, "mne", mne_mock)
    monkeypatch.setitem(sys.modules, "mne.io", mne_io_mock)
    monkeypatch.setitem(sys.modules, "mne.epochs", mne_epochs_mock)
    monkeypatch.setitem(sys.modules, "mne.preprocessing", mne_prep_mock)
    return mne_mock


def test_extract_data():
    raw = MockRaw()
    data = mne_integration._extract_data(raw, [0, 2])
    assert data.shape == (2, 100)

    epochs = MockEpochs()
    data = mne_integration._extract_data(epochs, [0, 2])
    # epochs concatenates along time: 3 epochs * 100 samples
    assert data.shape == (2, 300)

    with pytest.raises(TypeError, match="inst must be Raw or Epochs"):
        mne_integration._extract_data("not an mne object", [0])


def test_compute_pre_whitener():
    data = np.random.randn(4, 100)
    info = MockInfo()
    picks = [0, 1, 2, 3]

    pw = mne_integration._compute_pre_whitener(data, info, picks)
    assert pw.shape == (4, 1)
    assert np.all(pw > 0)


def test_compute_pca():
    data = np.random.randn(4, 100)
    pca_comp, pca_mean, ev = mne_integration._compute_pca(data, 2)
    assert pca_comp.shape == (4, 4)
    assert pca_mean.shape == (4,)
    assert ev.shape == (4,)


@patch("jamica.Amica")
def test_fit_ica_basic(AmicaMock, mock_mne_modules):
    """Test fit_ica normal execution path."""
    raw = MockRaw()

    # Mock Amica solver
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10

    mock_solver_instance = MagicMock()
    mock_solver_instance.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver_instance

    # Run fit_ica
    ica = mne_integration.fit_ica(raw, n_components=4, max_iter=10)

    # Check attributes were attached
    assert ica.method == "amica"
    assert hasattr(ica, "amica_result_")
    assert ica.amica_result_ is mock_result
    assert ica.n_components_ == 4
    assert ica.unmixing_matrix_.shape == (4, 4)
    assert ica.mixing_matrix_.shape == (4, 4)


@patch("jamica.Amica")
def test_fit_ica_stores_unmixer_for_unwhitened_pca(AmicaMock, mock_mne_modules):
    """Regression: ica.unmixing_matrix_ must operate on unwhitened X_pca.

    MNE's _transform computes sources = unmixing_matrix_ @ pca_components_ @ centered_data
    so the stored unmixing_matrix_ acts on unwhitened PCA-projected data, not on
    whitened data. A previous bug also divided by sqrt(pca_explained_variance_),
    which mis-scaled sources by ~sqrt(eigvals) per column and inflated log2|det W|
    by tens of bits on EEG-like spectra. This test pins the correct convention:
    when AMICA returns the identity for unit-variance-normalised input, the stored
    unmixer must equal diag(1/comp_stds), so the resulting sources have unit variance.
    """
    rng = np.random.default_rng(0)
    # Construct data with well-separated PCA eigenvalues so the bug would be visible.
    n_ch, n_samp = 4, 4000
    sources = rng.standard_normal((n_ch, n_samp))
    # Per-row scaling -> known eigenvalues 16, 4, 1, 0.25 (each row already orthogonal)
    sources *= np.array([4.0, 2.0, 1.0, 0.5])[:, None]
    raw = MockRaw(data=sources)

    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(n_ch)  # AMICA finds no rotation
    mock_result.n_iter = 1
    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver

    ica = mne_integration.fit_ica(raw, n_components=n_ch, max_iter=1)

    # Replicate MNE's _transform: sources = unmixing_matrix_ @ pca_components_ @ centered_data
    data = raw.get_data() / ica.pre_whitener_
    data_centered = data - ica.pca_mean_[:, None]
    X_pca = ica.pca_components_[:n_ch] @ data_centered
    Y = ica.unmixing_matrix_ @ X_pca

    # With W_white = I and the correct convention, sources are pca_data / comp_stds,
    # which is by construction unit variance per row.
    src_std = Y.std(axis=1)
    assert np.allclose(src_std, 1.0, atol=0.05), (
        f"Sources should have unit variance after fit_ica with W_white=I, got std={src_std}"
    )

    # log2|det W| with the bug would equal 2*sum(log2(eigvals)) too negative
    # (off by exactly that factor). With the fix, log2|det| should be tractable.
    _, logdet = np.linalg.slogdet(ica.unmixing_matrix_)
    log2_abs_det = logdet / np.log(2.0)
    # diag(1/comp_stds) -> det = prod(1/comp_stds); |log2 det| <= ~5 for eigvals 16..0.25.
    assert abs(log2_abs_det) < 10.0, (
        f"log2|det unmixing| = {log2_abs_det:.2f} suggests double /sqrt(eigvals) bug returned"
    )


def test_mne_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "mne.preprocessing", None)
    with pytest.raises(ImportError, match="MNE-Python is required"):
        mne_integration.fit_ica(MockRaw())


@patch("jamica.Amica")
def test_fit_ica_picks_string(AmicaMock, mock_mne_modules):
    raw = MockRaw(data=np.random.randn(4, 1000))
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10

    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver

    ica_str = mne_integration.fit_ica(raw, n_components=4, decim=2, picks="eeg")
    assert hasattr(ica_str, "amica_result_")


@patch("jamica.Amica")
def test_fit_ica_picks_list(AmicaMock, mock_mne_modules):
    raw = MockRaw(data=np.random.randn(4, 1000))
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(2)
    mock_result.n_iter = 10

    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver

    # n_components must not exceed the number of *picked* channels. Both
    # sklearn's PCA and mne.preprocessing.ICA raise here rather than silently
    # clipping, and so do we.
    with pytest.raises(ValueError, match="exceeds the number of selected channels"):
        mne_integration.fit_ica(raw, n_components=4, picks=[0, 1])

    ica_list = mne_integration.fit_ica(raw, n_components=2, picks=[0, 1])
    assert hasattr(ica_list, "amica_result_")


@patch("jamica.Amica")
def test_fit_ica_raw_with_reject(AmicaMock, mock_mne_modules):
    """Test lines 218-233: Raw with reject parameters uses Epochs."""
    raw = MockRaw(data=np.random.randn(4, 1000))

    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10

    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver

    # Needs to mock make_fixed_length_events and Epochs constructor returns
    import sys

    sys.modules["mne"].make_fixed_length_events.return_value = np.zeros((3, 3))
    mock_epochs_instance = MagicMock()
    mock_epochs_instance.get_data.return_value = np.random.randn(3, 4, 100)
    sys.modules["mne"].Epochs.return_value = mock_epochs_instance

    ica = mne_integration.fit_ica(
        raw,
        n_components=None,  # line 241
        reject={"eeg": 100},
        fit_params={"do_mean": True},  # line 278
    )
    assert ica.n_components_ == 4


@patch("jamica.Amica")
def test_fit_ica_names_exception(AmicaMock, mock_mne_modules, monkeypatch):
    """Test line 327-328: Exception when assigning _ica_names."""
    raw = MockRaw()

    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10

    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver

    class MockICABadNames:
        def __init__(self, **kwargs):
            self.exclude = []

        @property
        def _ica_names(self):
            return []

        @_ica_names.setter
        def _ica_names(self, val):
            raise AttributeError("read-only")

    mne_prep_mock = MagicMock()
    mne_prep_mock.ICA = MockICABadNames
    monkeypatch.setitem(sys.modules, "mne.preprocessing", mne_prep_mock)

    ica = mne_integration.fit_ica(raw, n_components=4)
    assert hasattr(ica, "amica_result_")
