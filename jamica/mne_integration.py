"""MNE-Python integration for AMICA.

Provides helper functions to use AMICA with MNE-Python's ICA workflow.

Usage
-----
>>> from jamica import fit_ica
>>> ica = fit_ica(raw, n_components=20, max_iter=2000)
>>> ica.plot_components()
>>> ica.apply(raw)
"""

from __future__ import annotations

import contextlib
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def _extract_data(inst, picks):
    """Extract (n_channels, n_samples) data array from Raw or Epochs.

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        The MNE data object to extract data from.
    picks : array-like of int
        Indices of the channels to extract.

    Returns
    -------
    data : np.ndarray, shape (n_channels, n_samples)
        The extracted continuous data. Epochs are concatenated along the time axis.
    """
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw

    if isinstance(inst, BaseRaw):
        return inst.get_data(picks)
    if isinstance(inst, BaseEpochs):
        # Concatenate all epochs along time axis
        return np.concatenate(inst.get_data()[:, picks, :], axis=-1)
    raise TypeError(f"inst must be Raw or Epochs, got {type(inst)}")


def _compute_pre_whitener(data, info, picks):
    """Compute MNE-style pre-whitener: per-channel-type std normalization.

    This replicates what MNE does in ICA._pre_whiten when noise_cov=None:
    divide each channel by the std of channels of that type.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        The data to compute the pre-whitener for.
    info : mne.Info
        The measurement info from the MNE object.
    picks : array-like of int
        Indices of the channels corresponding to the rows of `data`.

    Returns
    -------
    pre_whitener : np.ndarray, shape (n_channels, 1)
        The pre-whitening multiplier for each channel.
    """
    from mne import channel_type

    ch_types = [channel_type(info, idx) for idx in picks]
    unique_types = set(ch_types)

    pre_whitener = np.ones((len(picks), 1), dtype=np.float64)
    for ch_type in unique_types:
        mask = np.array([t == ch_type for t in ch_types])
        if mask.sum() > 0:
            std = np.std(data[mask])
            if std > 0:
                pre_whitener[mask, 0] = std

    return pre_whitener


def _compute_pca(data, n_components):
    """Compute PCA on pre-whitened, centered data.

    Computes a full SVD on the data and returns the principal components,
    mean, and explained variance for the full set of components (not truncated).

    Parameters
    ----------
    data : np.ndarray, shape (n_features, n_samples)
        The input data (channels by time).
    n_components : int | None
        Target number of components. Included for API compatibility.

    Returns
    -------
    pca_components : np.ndarray, shape (n_features, n_features)
        The principal components (V^T from SVD).
    pca_mean : np.ndarray, shape (n_features,)
        The mean of the data across samples.
    pca_explained_variance : np.ndarray, shape (n_features,)
        The variance explained by each component.
    """
    import scipy.linalg

    # data is (n_features, n_samples)
    data = data.T  # (n_samples, n_features)
    n_samples = data.shape[0]

    pca_mean = np.mean(data, axis=0)
    data_centered = data - pca_mean

    # Compute full SVD. For PCA we need U, S, Vh
    _U, S, Vh = scipy.linalg.svd(data_centered, full_matrices=False)

    pca_components = Vh  # (n_features, n_features)
    pca_explained_variance = (S**2) / (n_samples - 1)

    return pca_components, pca_mean, pca_explained_variance


def fit_ica(
    inst,
    n_components: int | None = None,
    max_iter: int = 2000,
    num_mix: int = 3,
    random_state: int | None = None,
    picks=None,
    reject=None,
    flat=None,
    decim=None,
    fit_params: dict | None = None,
    reject_by_annotation: bool = True,
    verbose=None,
):
    """Fit ICA using AMICA on MNE Raw or Epochs data.

    This function replicates MNE's whitening/PCA pipeline, then runs
    AMICA for the unmixing step. The result is a standard MNE ICA object
    that works with all MNE ICA methods (plot_components, apply, etc.).

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        MNE data object.
    n_components : int | None
        Number of PCA components to fit. ``None`` (default) uses the estimated
        numerical rank of the data, which is one less than the channel count for
        average-referenced EEG and lower still after channel interpolation; a
        ``RuntimeWarning`` reports the value chosen. An explicit value is never
        silently reduced: exceeding the number of selected channels, or exceeding
        the estimated rank, raises ``ValueError``.
    max_iter : int
        Maximum AMICA iterations. Default 2000.
    num_mix : int
        Number of generalized Gaussian mixture components. Default 3.
    random_state : int | None
        Random seed.
    picks : str | array-like | None
        Channels to use for ICA.
    reject : dict | None
        MNE-style **epoch amplitude** rejection (e.g. ``dict(eeg=100e-6)``): whole
        fixed-length epochs exceeding the threshold are dropped *before* fitting.
        Distinct from AMICA's likelihood-based sample rejection (see Notes).
    flat : dict | None
        Flat channel rejection parameters.
    decim : int | None
        Decimation factor.
    fit_params : dict | None
        Additional parameters forwarded to :class:`~jamica.config.AmicaConfig`,
        e.g. ``dict(do_reject=True, rejsig=3.0)`` to enable AMICA's per-sample
        likelihood rejection (see Notes).
    reject_by_annotation : bool
        Whether to omit bad segments from the data before fitting. If ``True``
        (default), annotated segments whose description begins with ``'bad'`` are
        omitted. If ``False``, no rejection based on annotations is performed.

        Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
    verbose : bool | None
        Verbosity.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object with AMICA decomposition. The full result is attached as
        ``ica.amica_result_``.

    Notes
    -----
    **Sample rejection** — two independent mechanisms, not to be confused:

    - ``reject`` / ``flat`` drop bad *epochs* by amplitude *before* the
      decomposition (standard MNE preprocessing).
    - AMICA's own ``do_reject`` drops individual outlier *samples* by their model
      log-likelihood *during* EM, following the Fortran AMICA 1.7 procedure (works for
      ``num_models`` = 1 and > 1; multi-model uses one global mask on the mixture LL).
      Enable it via ``fit_params``::

          ica = fit_ica(raw, n_components=20, fit_params=dict(do_reject=True, rejsig=3.0))
          mask = ica.amica_result_.sample_mask_  # bool, True = kept
          n_dropped = ica.amica_result_.n_rejected_

      ``sample_mask_`` indexes the samples passed to the fit (after any ``decim`` /
      epoch ``reject`` / ``reject_by_annotation``), not ``raw.times``.

    Examples
    --------
    >>> from jamica import fit_ica
    >>> ica = fit_ica(raw, n_components=20)
    >>> ica.plot_sources(raw)
    >>> ica.apply(raw)
    """
    try:
        from mne.preprocessing import ICA
    except ImportError as err:
        raise ImportError(
            "MNE-Python is required for fit_ica(). Install with: pip install mne"
        ) from err

    from jamica import Amica, AmicaConfig

    # Multi-model (num_models>1) is supported: fit_ica returns the highest-weight
    # model's decomposition as the primary mne.preprocessing.ICA, with the full
    # multi-model AmicaResult attached (ica.amica_result_) and any model
    # retrievable via jamica.get_model_ica(ica, h).
    _fit_params = fit_params or {}

    # ================================================================
    # Direct MNE ICA construction (no throwaway Infomax)
    # ================================================================
    prep = _prepare_mne_input(
        inst,
        n_components=n_components,
        picks=picks,
        reject=reject,
        flat=flat,
        decim=decim,
        reject_by_annotation=reject_by_annotation,
        verbose=verbose,
    )
    from mne.io import BaseRaw as _BaseRaw

    data_for_amica = prep.data_for_amica
    pre_whitener = prep.pre_whitener
    pca_components = prep.pca_components
    pca_mean = prep.pca_mean
    pca_explained_variance = prep.pca_explained_variance
    comp_stds = prep.comp_stds
    n_comp = prep.n_comp
    picks_idx = prep.picks_idx
    n_samples = prep.n_samples
    import mne

    # Step 4: Run AMICA
    cfg_kwargs = {
        "max_iter": max_iter,
        "num_mix_comps": num_mix,
        "do_sphere": False,
        "do_mean": False,
    }
    if fit_params:
        cfg_kwargs.update(fit_params)
    config = AmicaConfig(**cfg_kwargs)  # type: ignore[arg-type]

    solver = Amica(config, random_state=random_state)
    result = solver.fit(data_for_amica)

    W_white = np.asarray(result.unmixing_matrix_white_)
    if W_white.ndim == 3:
        # Multi-model: the primary decomposition is the highest-weight model.
        _primary = int(np.argmax(np.asarray(result.gm_)))
        W = W_white[_primary]
    else:
        _primary = 0
        W = W_white  # (n_comp, n_comp), operates on data_for_amica

    # Step 5: undo per-component unit-variance normalisation applied at line ~269.
    # AMICA was fed pca_data / comp_stds; recovering the unmixer for unwhitened
    # pca_data requires dividing each column j of W by comp_stds[j].
    # MNE's _transform computes sources = unmixing_matrix_ @ pca_components_ @ centered_data
    # i.e. unmixing_matrix_ must operate on unwhitened X_pca = pca_components_ @ centered_data
    # (the sqrt(eigvals) factor is *not* applied here — MNE bakes it into
    # unmixing_matrix_ at fit time only for backends that receive whitened input,
    # which AMICA does not).
    W_corrected = W / comp_stds.squeeze()[np.newaxis, :]

    # Step 6: Construct MNE ICA object with all required attributes
    # MNE validates method at __init__ — use 'infomax' placeholder, override below
    ica = ICA(n_components=n_comp, method="infomax", max_iter=max_iter)

    # Channel info
    ica.info = mne.pick_info(inst.info, picks_idx)
    ica.ch_names = [inst.info["ch_names"][i] for i in picks_idx]

    # Pre-whitening
    ica.pre_whitener_ = pre_whitener

    # PCA
    ica.pca_components_ = pca_components
    ica.pca_mean_ = pca_mean
    ica.pca_explained_variance_ = pca_explained_variance

    # ICA decomposition (W_corrected operates on unwhitened X_pca — same convention
    # as MNE's stored unmixing_matrix_ for picard/fastica/infomax post-fit).
    ica.n_components_ = n_comp
    ica.unmixing_matrix_ = W_corrected
    ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

    # Metadata
    ica.n_iter_ = result.n_iter
    ica.n_samples_ = n_samples
    ica.current_fit = "raw" if isinstance(inst, _BaseRaw) else "epochs"
    ica.method = "amica"
    ica.labels_ = {}
    ica.exclude = []
    ica.reject_ = reject
    ica.drop_inds_ = np.array([], dtype=int)

    # Internal naming
    with contextlib.suppress(Exception):
        ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_comp)]

    # Attach full AMICA result for viz module
    ica.amica_result_ = result

    # Bookkeeping so any model of a multi-model fit can be materialised later
    # (get_model_ica): the per-component normalisation and the primary index.
    ica._amica_comp_stds = comp_stds
    ica._amica_model_index = _primary

    return ica


def get_model_ica(ica, model):
    """Return an ``mne.preprocessing.ICA`` for one model of a multi-model AMICA fit.

    ``fit_ica(num_models>1)`` returns the highest-weight model as the primary ICA.
    This materialises any model's decomposition, sharing the same pre-whitening and
    PCA so ``apply``/``get_sources``/``plot_*`` work unchanged.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        An ICA returned by :func:`fit_ica` for a multi-model fit (has
        ``amica_result_`` with a 3-D ``unmixing_matrix_white_``).
    model : int
        Model index in ``[0, num_models)``.

    Returns
    -------
    mne.preprocessing.ICA
        A copy of ``ica`` whose unmixing/mixing matrices are model ``model``'s.
    """
    import copy

    result = getattr(ica, "amica_result_", None)
    W_white = None if result is None else np.asarray(result.unmixing_matrix_white_)
    if W_white is None or W_white.ndim != 3:
        raise ValueError("get_model_ica() requires an ICA returned by fit_ica(num_models>1).")
    n_models = W_white.shape[0]
    if not 0 <= int(model) < n_models:
        raise IndexError(f"model {model} out of range [0, {n_models}).")

    comp_stds = np.asarray(ica._amica_comp_stds).squeeze()[np.newaxis, :]
    W_corrected = W_white[int(model)] / comp_stds

    out = copy.copy(ica)  # shares pca/pre_whitener/info/amica_result_ (read-only)
    out.unmixing_matrix_ = W_corrected
    out.mixing_matrix_ = np.linalg.pinv(W_corrected)
    out.exclude = []
    out.labels_ = {}
    out._amica_model_index = int(model)
    return out


class _MnePrep:
    """Everything MNE-side needed to build ICA objects from an AMICA fit.

    Holds the result of replicating MNE's pre-whitening + PCA pipeline, so that
    :func:`fit_ica` and :class:`~jamica.AmicaICA` share one preprocessing path
    rather than two that can drift apart.
    """

    __slots__ = (
        "comp_stds",
        "data_for_amica",
        "decim",
        "fit_sample_mask",
        "n_comp",
        "n_samples",
        "pca_components",
        "pca_explained_variance",
        "pca_mean",
        "picks_idx",
        "pre_whitener",
    )

    def __init__(
        self,
        data_for_amica,
        pre_whitener,
        pca_components,
        pca_mean,
        pca_explained_variance,
        comp_stds,
        n_comp,
        picks_idx,
        n_samples,
        fit_sample_mask=None,
        decim=None,
    ):
        self.data_for_amica = data_for_amica
        self.pre_whitener = pre_whitener
        self.pca_components = pca_components
        self.pca_mean = pca_mean
        self.pca_explained_variance = pca_explained_variance
        self.comp_stds = comp_stds
        self.n_comp = n_comp
        self.picks_idx = picks_idx
        self.n_samples = n_samples
        self.fit_sample_mask = fit_sample_mask
        self.decim = decim

    def project(self, data):
        """Apply the fitted pre-whitening/PCA/scaling to new sensor-space data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            Sensor-space data for the same channels, in the same order, as the fit.

        Returns
        -------
        np.ndarray, shape (n_comp, n_samples)
            Data in the space AMICA was fitted in.
        """
        data_pre = np.asarray(data, dtype=np.float64) / self.pre_whitener
        centered = data_pre - self.pca_mean[:, None]
        return (self.pca_components[: self.n_comp] @ centered) / self.comp_stds


def _prepare_mne_input(
    inst,
    n_components=None,
    picks=None,
    reject=None,
    flat=None,
    decim=None,
    reject_by_annotation=True,
    verbose=None,
):
    """Replicate MNE's pre-whiten + PCA pipeline ahead of an AMICA fit.

    Extracted verbatim from :func:`fit_ica` so the multi-model parent object
    uses exactly the same preprocessing rather than a second implementation.

    Returns
    -------
    _MnePrep
        The projected data plus every array needed to populate an
        ``mne.preprocessing.ICA``.
    """
    import mne

    # Resolve picks using public MNE API
    if picks is None:
        # Default: all data channels (eeg, meg, etc.) excluding bads — same as MNE ICA
        picks_idx = mne.pick_types(inst.info, meg=True, eeg=True, ref_meg=False, exclude="bads")
    elif isinstance(picks, str):
        picks_idx = mne.pick_types(inst.info, **{picks: True}, exclude="bads")
    else:
        picks_idx = np.asarray(picks, dtype=int)

    # Extract data, applying reject/flat if provided for Raw
    from mne.io import BaseRaw as _BaseRaw

    if isinstance(inst, _BaseRaw) and (reject is not None or flat is not None):
        # Create fixed-length epochs to apply rejection (matches MNE ICA behavior)
        events = mne.make_fixed_length_events(inst, duration=1.0)
        epochs = mne.Epochs(
            inst,
            events,
            tmin=0,
            tmax=1.0 - 1.0 / inst.info["sfreq"],
            picks=picks_idx,
            reject=reject,
            flat=flat,
            baseline=None,
            reject_by_annotation=reject_by_annotation,
            preload=True,
            verbose=verbose,
        )
        raw_data = np.concatenate(epochs.get_data(), axis=-1)
        # picks_idx already applied inside Epochs
        raw_data = raw_data.reshape(len(picks_idx), -1) if raw_data.ndim == 3 else raw_data

        # Record which points of the ORIGINAL timeline entered the fit. Epoching
        # excludes two disjoint sets: epochs dropped by reject/flat, and any tail
        # shorter than one epoch that make_fixed_length_events never covered.
        # epochs.selection indexes surviving events, so the mapping is exact.
        n_orig = getattr(inst, "n_times", None)
        if n_orig is None:
            times = getattr(inst, "times", None)
            n_orig = len(times) if times is not None else None
        if n_orig is None:
            # No timeline exposed (e.g. a lightweight test double). Claim no
            # exclusions rather than guess at a grid we cannot see.
            fit_sample_mask = np.ones(raw_data.shape[1], dtype=bool)
        else:
            fit_sample_mask = np.zeros(int(n_orig), dtype=bool)
            epoch_len = len(epochs.times)
            first_samp = int(getattr(inst, "first_samp", 0) or 0)
            for onset in events[np.asarray(epochs.selection, dtype=int), 0]:
                start = int(onset) - first_samp
                fit_sample_mask[start : start + epoch_len] = True
    elif isinstance(inst, _BaseRaw) and reject_by_annotation:
        raw_data = inst.get_data(picks_idx, reject_by_annotation="omit")
        fit_sample_mask = ~np.isnan(inst.get_data(picks_idx[:1], reject_by_annotation="NaN")[0])
    else:
        raw_data = _extract_data(inst, picks_idx)
        fit_sample_mask = np.ones(raw_data.shape[1], dtype=bool)

    n_channels, n_samples = raw_data.shape

    # Resolve n_components
    if n_components is None:
        n_comp = n_channels
    else:
        if n_components > n_channels:
            raise ValueError(
                f"n_components={n_components} exceeds the number of selected channels "
                f"({n_channels}). Pass n_components<={n_channels}."
            )
        if n_components < 2:
            # Validate the lower bound here so 0 and negative values fail with a clear
            # message rather than an IndexError on an empty array or a negative slice,
            # and so n_components=1 is not misreported as a rank-1 dataset.
            raise ValueError(
                f"n_components={n_components} is invalid; ICA needs at least 2 components."
            )
        n_comp = n_components

    # Decimation
    if decim is not None and decim > 1:
        import scipy.signal

        logger.info("Decimating data by factor %d using FIR anti-aliasing filter.", decim)
        raw_data = scipy.signal.decimate(raw_data, decim, axis=-1, ftype="fir")
        n_samples = raw_data.shape[1]

    # Step 1: Pre-whiten (per-channel-type std normalization)
    pre_whitener = _compute_pre_whitener(raw_data, inst.info, picks_idx)
    data_pre = raw_data / pre_whitener

    # Step 2: PCA
    pca_components, pca_mean, pca_explained_variance = _compute_pca(data_pre, n_comp)

    # Step 3: Project to PCA space (truncated to n_components)
    data_centered = data_pre - pca_mean[:, None]
    pca_data = pca_components[:n_comp] @ data_centered
    # pca_data shape: (n_comp, n_samples)

    # Normalize to unit variance per component to stabilize AMICA's gradient.
    comp_stds = np.std(pca_data, axis=1, keepdims=True)

    # Drop rank-deficient PCA directions before the division below. Average-referenced
    # EEG has rank n_channels - 1, so its trailing direction carries only numerical
    # noise (std ~1e-16). Dividing by that std inflates the corresponding column of W
    # by ~1e16; np.linalg.pinv then uses a cutoff of rcond * sigma_max that exceeds
    # every legitimate singular value, the pseudo-inverse collapses, and ICA.apply()
    # returns near-zero data. Because SVD orders components by decreasing variance,
    # the degenerate directions are always trailing, so truncating n_comp removes
    # exactly those. MNE keeps the full pca_components_ and restores the dropped
    # directions as PCA residual at their true (negligible) amplitude.
    # Use the standard SVD numerical-rank tolerance (the rule behind
    # np.linalg.matrix_rank): sigma_max * max(matrix shape) * eps. A looser
    # threshold would discard genuine low-variance components; a tighter one would
    # leave the ill-conditioned column in place.
    stds_flat = comp_stds.ravel()
    rank_tol = stds_flat[0] * max(data_centered.shape) * np.finfo(pca_data.dtype).eps
    n_keep = int(np.count_nonzero(stds_flat > rank_tol))  # SVD order makes this a prefix

    if n_keep < 2:
        raise ValueError(
            f"Estimated data rank is {n_keep}; ICA needs at least 2 components. The "
            "input appears constant or degenerate after pre-whitening."
        )
    if n_keep < n_comp:
        if n_components is not None:
            # An explicit request we cannot honour is an error, not something to
            # silently reinterpret.
            raise ValueError(
                f"n_components={n_components} exceeds the estimated rank of the data "
                f"({n_keep} from {n_channels} selected channels). Average referencing "
                "reduces rank by one. Pass n_components<=" + f"{n_keep}."
            )
        warnings.warn(
            f"Estimated data rank is {n_keep} from {n_channels} selected channels; "
            f"fitting {n_keep} components instead of {n_comp}. Rank deficiency is "
            "expected after average referencing or channel interpolation. Pass "
            f"n_components={n_keep} to select this explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )
        n_comp = n_keep
        pca_data = pca_data[:n_comp]
        comp_stds = comp_stds[:n_comp]

    data_for_amica = pca_data / comp_stds

    return _MnePrep(
        data_for_amica=data_for_amica,
        pre_whitener=pre_whitener,
        pca_components=pca_components,
        pca_mean=pca_mean,
        pca_explained_variance=pca_explained_variance,
        comp_stds=comp_stds,
        n_comp=n_comp,
        picks_idx=picks_idx,
        n_samples=n_samples,
        fit_sample_mask=fit_sample_mask,
        decim=decim,
    )
