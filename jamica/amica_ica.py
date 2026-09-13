"""Multi-model AMICA as a parent object owning per-model MNE ICA views.

``mne.preprocessing.ICA`` holds one unmixing matrix, so it cannot represent a
mixture of ``M`` models natively. Rather than change what an ``ICA`` means,
:class:`AmicaICA` keeps the complete AMICA fit as the authoritative object and
exposes each constituent model as an ordinary ``ICA``.

The children need not teach MNE what a mixture is: each has exactly one
unmixing matrix, one PCA state, its own ``exclude`` and its own ``labels_``.
The mixture -- the model priors and the posterior time course -- stays one
level above them, on the parent.
"""

from __future__ import annotations

import contextlib

import numpy as np

from .mne_integration import _prepare_mne_input

__all__ = ["AmicaICA", "read_amica_ica"]


def _fold_center_into_pca_mean(pca_mean, pca_components, comp_stds, c_h, n_comp):
    """Express an AMICA model centre as an offset to MNE's ``pca_mean_``.

    AMICA's sources for model ``h`` are ``W_h @ (x_white - c_h)``, where
    ``x_white`` is what the fit was given: the PCA projection scaled to unit
    variance per component. MNE instead computes
    ``unmixing_matrix_ @ pca_components_[:n] @ (x_prewhitened - pca_mean_)``.

    Writing ``P`` for ``pca_components[:n_comp]`` and ``D`` for
    ``diag(comp_stds)``, matching the two expressions requires
    ``D^-1 P delta = c_h``, i.e. ``P delta = D c_h``. ``P`` has orthonormal
    rows, so ``delta = P.T @ (comp_stds * c_h)``.

    Returns
    -------
    np.ndarray, shape (n_channels,)
        The per-model ``pca_mean_``. Equals ``pca_mean`` when ``c_h`` is zero.
    """
    scaled = np.asarray(comp_stds).ravel()[:n_comp] * np.asarray(c_h).ravel()[:n_comp]
    return np.asarray(pca_mean) + pca_components[:n_comp].T @ scaled


class AmicaICA:
    """Multi-model AMICA fit exposing one ``mne.preprocessing.ICA`` per model.

    Parameters
    ----------
    n_models : int
        Number of AMICA models. ``1`` reduces to an ordinary single-model fit.
    n_components : int | None
        Number of PCA components. ``None`` uses the estimated numerical rank.
    max_iter : int
        Maximum AMICA iterations.
    num_mix : int
        Generalized-Gaussian mixture components per source.
    random_state : int | None
        Seed for the AMICA fit.
    picks : str | array-like | None
        Channels to use, following :func:`~jamica.fit_ica`.
    reject : dict | None
        MNE-style epoch amplitude rejection applied before fitting.
    flat : dict | None
        Flat-channel rejection applied before fitting.
    decim : int | None
        Decimation factor applied before fitting. See Notes on posteriors.
    fit_params : dict | None
        Extra keywords forwarded to :class:`~jamica.config.AmicaConfig`.
    reject_by_annotation : bool
        Whether to omit bad segments from the data before fitting. If ``True``
        (default), annotated segments whose description begins with ``'bad'`` are
        omitted. If ``False``, no rejection based on annotations is performed.

        Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
    verbose : bool | None
        Verbosity.

    Attributes
    ----------
    models_ : list of mne.preprocessing.ICA
        One child per AMICA model, in model order. Built once at fit time and
        cached, so ``exclude`` / ``labels_`` set on a child by interactive
        review survive repeated access.
    n_models_ : int
        Number of fitted models.
    model_weights_ : np.ndarray, shape (n_models,)
        Model priors (AMICA's ``gm``).
    model_posteriors_ : np.ndarray
        ``p(h | x_t)`` on the ORIGINAL input sampling grid: ``(n_models,
        n_times)`` for Raw and ``(n_models, n_epochs, n_times)`` for Epochs,
        regardless of any decimation used while fitting. Points excluded from
        the fit by epoch or annotation rejection are ``NaN``.
    fit_sample_mask_ : np.ndarray of bool
        Which points of that same grid entered the optimisation --
        ``(n_times,)`` for Raw, ``(n_epochs, n_times)`` for Epochs. See
        :meth:`_build_fit_sample_mask` for how decimation is treated.
    amica_result_ : jamica.AmicaResult
        The complete fit, including the density parameters per model.

    Notes
    -----
    ``model_weights_`` and ``model_posteriors_`` are different quantities: the
    former is a single global prior per model, the latter a per-sample
    responsibility that says which model is active when.

    ``model_posteriors_`` is always on the input grid. Because a genuine
    new-data evaluation exists, the fitted model is simply applied to every
    sample afterwards; nothing is interpolated or resampled from the fit-time
    array. Decimation therefore changes which samples drove the optimisation,
    not the shape of the reported posteriors.
    """

    def __init__(
        self,
        n_models: int = 1,
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
        if int(n_models) < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}.")
        self.n_models = int(n_models)
        self.n_components = n_components
        self.max_iter = max_iter
        self.num_mix = num_mix
        self.random_state = random_state
        self.picks = picks
        self.reject = reject
        self.flat = flat
        self.decim = decim
        self.fit_params = fit_params
        self.reject_by_annotation = reject_by_annotation
        self.verbose = verbose
        self._models = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, inst):
        """Fit AMICA on Raw or Epochs.

        Parameters
        ----------
        inst : mne.io.Raw | mne.Epochs
            Data to decompose.

        Returns
        -------
        AmicaICA
            The fitted instance.
        """
        import mne
        from mne.io import BaseRaw

        from .config import AmicaConfig
        from .solver import Amica

        prep = _prepare_mne_input(
            inst,
            n_components=self.n_components,
            picks=self.picks,
            reject=self.reject,
            flat=self.flat,
            decim=self.decim,
            reject_by_annotation=self.reject_by_annotation,
            verbose=self.verbose,
        )

        cfg_kwargs = {
            "max_iter": self.max_iter,
            "num_mix_comps": self.num_mix,
            "num_models": self.n_models,
            "do_sphere": False,
            "do_mean": False,
        }
        if self.fit_params:
            cfg_kwargs.update(self.fit_params)
        config = AmicaConfig(**cfg_kwargs)  # type: ignore[arg-type]

        result = Amica(config, random_state=self.random_state).fit(prep.data_for_amica)

        self._prep = prep
        self._inst_kind = "raw" if isinstance(inst, BaseRaw) else "epochs"
        self.amica_result_ = result
        self.n_components_ = prep.n_comp
        self.n_iter_ = result.n_iter
        self.ch_names = [inst.info["ch_names"][i] for i in prep.picks_idx]

        w_all = np.asarray(result.unmixing_matrix_white_)
        if w_all.ndim == 2:  # single model: give it a leading model axis
            w_all = w_all[None]
        self.n_models_ = w_all.shape[0]

        c_all = np.asarray(result.c_)
        if c_all.ndim == 1:
            c_all = c_all[None]
        self._W_all = w_all
        self._c_all = c_all

        gm = np.asarray(result.gm_).ravel()
        self.model_weights_ = gm if gm.size == self.n_models_ else np.ones(self.n_models_)

        # Build every child eagerly and cache: interactive MNE review mutates
        # ICA.exclude, so handing back a fresh object per access would silently
        # discard the user's work.
        picked_info = mne.pick_info(inst.info, prep.picks_idx)
        self._models = [self._build_child(picked_info, h, prep) for h in range(self.n_models_)]

        # Posteriors are reported on the ORIGINAL sampling grid, not the grid the
        # optimiser happened to see. They are re-evaluated from the fitted model
        # rather than stretched from the fit-time array, so no value is invented.
        # Computed after the children exist, since the evaluation path is shared
        # with get_model_probabilities().
        self.fit_sample_mask_ = self._build_fit_sample_mask(inst, prep)
        self.model_posteriors_ = self._posteriors_on_input_grid(inst)
        return self

    # ------------------------------------------------------------------
    # children
    # ------------------------------------------------------------------
    def _build_child(self, picked_info, model_index, prep):
        """Materialise model ``model_index`` as an ordinary ``ICA``.

        Constructed by assignment rather than by calling ``ICA.fit()``: MNE's
        fit sorts components by explained variance afterwards, which would
        break the correspondence between an AMICA component and its density
        parameters (``alpha`` / ``mu`` / ``beta`` / ``rho``) in
        ``amica_result_``.
        """
        from mne.preprocessing import ICA

        n_comp = prep.n_comp
        w_h = self._W_all[model_index]
        c_h = self._c_all[model_index] if self._c_all.shape[0] > model_index else None

        # MNE validates ``method`` at __init__; set a known value and override.
        ica = ICA(n_components=n_comp, method="infomax", max_iter=self.max_iter)

        ica.info = picked_info.copy()
        ica.ch_names = list(self.ch_names)
        ica.pre_whitener_ = prep.pre_whitener
        ica.pca_components_ = prep.pca_components
        ica.pca_explained_variance_ = prep.pca_explained_variance

        # Per-model centre, folded in. Its own array per child, never shared.
        if c_h is None or not np.any(c_h):
            ica.pca_mean_ = np.array(prep.pca_mean, copy=True)
        else:
            ica.pca_mean_ = _fold_center_into_pca_mean(
                prep.pca_mean, prep.pca_components, prep.comp_stds, c_h, n_comp
            )

        ica.n_components_ = n_comp
        ica.unmixing_matrix_ = w_h / prep.comp_stds.squeeze()[np.newaxis, :]
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

        ica.n_iter_ = self.n_iter_
        ica.n_samples_ = prep.n_samples
        ica.current_fit = self._inst_kind
        ica.method = "amica"
        ica.labels_ = {}
        ica.exclude = []
        ica.reject_ = self.reject
        ica.drop_inds_ = np.array([], dtype=int)
        with contextlib.suppress(Exception):
            ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_comp)]

        # Identify which AMICA model this view came from, and keep the full fit
        # reachable from any child.
        ica._amica_model_index = int(model_index)
        ica.amica_result_ = self.amica_result_
        ica._amica_comp_stds = prep.comp_stds
        return ica

    @property
    def models_(self):
        """List of per-model ``mne.preprocessing.ICA`` views (cached)."""
        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")
        return self._models

    # ------------------------------------------------------------------
    # posteriors
    # ------------------------------------------------------------------
    def _build_fit_sample_mask(self, inst, prep):
        """Which points of the input timeline actually entered the optimisation.

        ``True`` where the sample contributed. Epoch rejection (``reject`` /
        ``flat``) sets ``False`` on the dropped epochs and on any trailing
        segment shorter than one epoch. ``reject_by_annotation`` sets ``False``
        on bad-annotated spans.

        Decimation does **not** clear the mask. ``scipy.signal.decimate`` applies
        an anti-aliasing FIR filter before downsampling, so each retained point
        is a weighted combination of its neighbours and every input sample
        contributes to the fit. Marking the non-retained grid points ``False``
        would misdescribe that.
        """
        mask = prep.fit_sample_mask
        if self._inst_kind == "raw":
            out = np.ones(len(inst.times), dtype=bool)
            if mask is not None and mask.shape[0] == out.shape[0]:
                out = np.asarray(mask, dtype=bool).copy()
            return out
        n_epochs, n_times = len(inst), len(inst.times)
        return np.ones((n_epochs, n_times), dtype=bool)

    def _posteriors_on_input_grid(self, inst):
        """``p(h|x_t)`` for every point of the input timeline.

        Samples excluded from the fit by epoch or annotation rejection are returned as ``NaN``
        rather than as a value the fit never saw.
        """
        post = self.get_model_probabilities(inst)
        mask = self.fit_sample_mask_
        if mask is not None and not mask.all():
            post = np.array(post, dtype=float, copy=True)
            if post.ndim == 2:
                post[:, ~mask] = np.nan
            else:
                post[:, ~mask] = np.nan
        return post

    def get_model_probabilities(self, inst):
        """Model posteriors ``p(h | x_t)`` evaluated on new data.

        Recomputed from the fitted parameters, not resampled from the fit-time
        posterior array.

        Parameters
        ----------
        inst : mne.io.Raw | mne.Epochs
            Data with the same channels as the fit.

        Returns
        -------
        np.ndarray
            ``(n_models, n_times)`` for Raw, ``(n_models, n_epochs, n_times)``
            for Epochs.
        """
        from mne.io import BaseRaw

        from .mne_integration import _extract_data
        from .multimodel import compute_model_posteriors

        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")

        prep = self._prep
        data_white = prep.project(_extract_data(inst, prep.picks_idx))

        res = self.amica_result_
        # ``sbeta_`` stores verbatim the array the solver passes as ``beta`` to
        # compute_model_posteriors, so it is forwarded as-is -- the name is
        # Fortran lineage, not an inversion.
        alpha = np.asarray(res.alpha_)
        mu = np.asarray(res.mu_)
        beta = np.asarray(res.sbeta_)
        rho = np.asarray(res.rho_)
        # Density parameters lose the model axis for a single-model fit.
        if alpha.ndim == 2:
            alpha, mu, beta, rho = alpha[None], mu[None], beta[None], rho[None]

        # log_det_sphere is identical across models, so it cancels in the
        # softmax over h and can be left at zero here.
        post = np.asarray(
            compute_model_posteriors(
                data_white,
                self._W_all,
                self._c_all,
                alpha,
                mu,
                beta,
                rho,
                np.asarray(self.model_weights_),
                log_det_sphere=0.0,
            )
        )

        if not isinstance(inst, BaseRaw):
            n_epochs, n_times = len(inst), len(inst.times)
            if post.shape[1] == n_epochs * n_times:
                return post.reshape(post.shape[0], n_epochs, n_times)
        return post

    # ------------------------------------------------------------------
    # application
    # ------------------------------------------------------------------
    def apply(self, inst, model_idx=None, **kwargs):
        """Remove excluded components using one model's decomposition.

        Parameters
        ----------
        inst : mne.io.Raw | mne.Epochs
            Data to clean, modified in place by MNE.
        model_idx : int | None
            Which model to apply. Required when ``n_models_ > 1``.
        **kwargs
            Passed through to :meth:`mne.preprocessing.ICA.apply`.

        Returns
        -------
        inst
            The cleaned instance.

        Raises
        ------
        ValueError
            When ``n_models_ > 1`` and no model was named. A mixture has no
            single reconstruction until a combination rule is chosen, and
            silently taking the highest-weight model would hide that choice.
        """
        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")

        if model_idx is None:
            if self.n_models_ == 1:
                return self._models[0].apply(inst, **kwargs)
            raise ValueError(
                f"This is a {self.n_models_}-model AMICA fit, and no strategy for "
                "reconstructing data from the whole mixture has been selected, so "
                "apply() is ambiguous. Name the model explicitly:\n"
                "    amica.apply(inst, model_idx=h)\n"
                "    amica.models_[h].apply(inst)\n"
                "Model priors are in .model_weights_ and the per-sample "
                "responsibilities in .model_posteriors_."
            )

        idx = int(model_idx)
        if not 0 <= idx < self.n_models_:
            raise IndexError(f"model_idx {idx} out of range [0, {self.n_models_}).")
        return self._models[idx].apply(inst, **kwargs)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, fname, overwrite=False):
        """Write the whole fit -- every model plus the mixture -- to HDF5.

        A FIF file stores one unmixing matrix, so it cannot hold a mixture on
        its own. HDF5 is used here for the same reason
        :class:`~mne.preprocessing.EOGRegression` uses it, and through the same
        MNE helper. Pair this with :meth:`export_model_fifs` when the
        individual models should also be readable without jamica installed.

        Parameters
        ----------
        fname : path-like
            Destination, conventionally ending in ``.h5``.
        overwrite : bool
            Overwrite an existing file.
        """
        from mne.utils import _check_fname, _import_h5io_funcs, _validate_type

        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")
        _, write_hdf5 = _import_h5io_funcs()
        _validate_type(fname, "path-like", "fname")
        fname = _check_fname(fname, overwrite=overwrite, name="fname")
        write_hdf5(fname, self._as_state(), overwrite=overwrite, title="jamica")

    def _as_state(self):
        """Everything needed to rebuild this object, as plain containers."""
        import dataclasses

        prep = self._prep
        return {
            "jamica_format": 1,
            "params": {
                "n_models": self.n_models,
                "n_components": self.n_components,
                "max_iter": self.max_iter,
                "num_mix": self.num_mix,
                "random_state": self.random_state,
                "reject": self.reject,
                "flat": self.flat,
                "decim": self.decim,
                "fit_params": self.fit_params,
                "reject_by_annotation": self.reject_by_annotation,
            },
            "fitted": {
                "n_models_": int(self.n_models_),
                "n_components_": int(self.n_components_),
                "n_iter_": int(self.n_iter_),
                "ch_names": list(self.ch_names),
                "inst_kind": self._inst_kind,
                "model_weights_": np.asarray(self.model_weights_),
                "model_posteriors_": np.asarray(self.model_posteriors_),
                "fit_sample_mask_": np.asarray(self.fit_sample_mask_),
                "W_all": np.asarray(self._W_all),
                "c_all": np.asarray(self._c_all),
            },
            # The picked Info, taken from a child so it matches exactly what
            # the children were built with.
            "info": self._models[0].info.copy(),
            "prep": {
                "pre_whitener": np.asarray(prep.pre_whitener),
                "pca_components": np.asarray(prep.pca_components),
                "pca_mean": np.asarray(prep.pca_mean),
                "pca_explained_variance": np.asarray(prep.pca_explained_variance),
                "comp_stds": np.asarray(prep.comp_stds),
                "n_comp": int(prep.n_comp),
                "picks_idx": np.asarray(prep.picks_idx),
                "n_samples": int(prep.n_samples),
                "fit_sample_mask": (
                    None if prep.fit_sample_mask is None else np.asarray(prep.fit_sample_mask)
                ),
                "decim": prep.decim,
            },
            "result": dataclasses.asdict(self.amica_result_),
            "exclude": [list(m.exclude) for m in self._models],
            "labels": [dict(m.labels_) for m in self._models],
        }

    def export_model_fifs(self, fname, overwrite=False):
        """Write each model as an ordinary ``-ica.fif``.

        These are plain MNE ICA files: they open with
        :func:`mne.preprocessing.read_ica` on a machine that has never
        installed jamica, so a decomposition never becomes readable only
        through this package. What they cannot carry is the mixture itself,
        the priors and the posterior time course, which is what :meth:`save`
        is for.

        Parameters
        ----------
        fname : path-like
            Template ending in ``-ica.fif``. The model index is inserted before
            the suffix, so ``sub-01-ica.fif`` yields ``sub-01-model-0-ica.fif``
            and so on.
        overwrite : bool
            Overwrite existing files.

        Returns
        -------
        list of pathlib.Path
            The files written, in model order.
        """
        import pathlib

        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")
        fname = pathlib.Path(fname)
        name = fname.name
        for suffix in ("-ica.fif.gz", "-ica.fif"):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                break
        else:
            raise ValueError(f"fname must end in '-ica.fif' for MNE to read it back, got {name!r}.")

        written = []
        for h, model in enumerate(self._models):
            out = fname.with_name(f"{stem}-model-{h}{suffix}")
            model.save(out, overwrite=overwrite)
            written.append(out)
        return written

    def __repr__(self):
        if self._models is None:
            return f"<AmicaICA (unfitted, n_models={self.n_models})>"
        return (
            f"<AmicaICA | {self.n_models_} models, {self.n_components_} components, "
            f"{self.n_iter_} iterations>"
        )


def read_amica_ica(fname):
    """Read an :class:`AmicaICA` written by :meth:`AmicaICA.save`.

    Parameters
    ----------
    fname : path-like
        The ``.h5`` file to read.

    Returns
    -------
    AmicaICA
        The restored fit, with its per-model :class:`~mne.preprocessing.ICA`
        views rebuilt and each child's ``exclude`` / ``labels_`` preserved.
    """
    import mne
    from mne.utils import _check_fname, _import_h5io_funcs, _validate_type

    from .mne_integration import _MnePrep
    from .solver import AmicaResult

    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, "path-like", "fname")
    fname = _check_fname(fname, overwrite="read", must_exist=True, name="fname")
    state = read_hdf5(fname, title="jamica")

    fmt = state.get("jamica_format")
    if fmt != 1:
        raise ValueError(f"unsupported jamica file format {fmt!r}; expected 1.")

    out = AmicaICA(**state["params"])
    fitted = state["fitted"]
    prep_d = state["prep"]

    # h5io hands an Info back as a plain dict, and MNE's own helpers reject
    # that, so put it back into a real Info before anything is built from it.
    info = state["info"]
    if not isinstance(info, mne.Info):
        info = mne.Info(**info)

    out._prep = _MnePrep(
        data_for_amica=None,  # fit data is not stored; not needed to rebuild views
        pre_whitener=prep_d["pre_whitener"],
        pca_components=prep_d["pca_components"],
        pca_mean=prep_d["pca_mean"],
        pca_explained_variance=prep_d["pca_explained_variance"],
        comp_stds=prep_d["comp_stds"],
        n_comp=int(prep_d["n_comp"]),
        picks_idx=prep_d["picks_idx"],
        n_samples=int(prep_d["n_samples"]),
        fit_sample_mask=prep_d["fit_sample_mask"],
        decim=prep_d["decim"],
    )
    out.amica_result_ = AmicaResult(**state["result"])
    out._inst_kind = fitted["inst_kind"]
    out.n_models_ = int(fitted["n_models_"])
    out.n_components_ = int(fitted["n_components_"])
    out.n_iter_ = int(fitted["n_iter_"])
    out.ch_names = list(fitted["ch_names"])
    out.model_weights_ = fitted["model_weights_"]
    out.model_posteriors_ = fitted["model_posteriors_"]
    out.fit_sample_mask_ = fitted["fit_sample_mask_"]
    out._W_all = fitted["W_all"]
    out._c_all = fitted["c_all"]

    out._models = [out._build_child(info, h, out._prep) for h in range(out.n_models_)]
    for model, exclude, labels in zip(out._models, state["exclude"], state["labels"], strict=False):
        model.exclude = list(exclude)
        model.labels_ = dict(labels)
    return out
