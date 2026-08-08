"""AMICA configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class AmicaConfig:
    """Configuration for AMICA algorithm.

    Parameters
    ----------
    num_models : int
        Number of ICA models to learn simultaneously. Default is 1.
    num_mix_comps : int
        Number of Gaussian mixture components per source. Default is 3.
    dtype : str
        Data type to use for computation: "float32" or "float64". Default is "float64".
    pcakeep : int, optional
        Number of PCA components to keep. If None, uses data rank.
    max_iter : int
        Maximum number of iterations. Default is 2000.
    lrate : float
        Initial learning rate. Default is 0.1.
    minlrate : float
        Minimum learning rate before stopping. Default is 1e-8.
    lratefact : float
        Factor to decrease learning rate on likelihood decrease. Default is 0.5.
    rholrate : float
        Learning rate for shape parameter rho. Default is 0.05.
    rholratefact : float
        Factor to decrease rho learning rate on likelihood decrease. Default is 0.5.
    minrho : float
        Minimum value for shape parameter (1.0 = Laplacian). Default is 1.0.
    maxrho : float
        Maximum value for shape parameter (2.0 = Gaussian). Default is 2.0.
    do_newton : bool
        Whether to use Newton updates for faster convergence. Default is True.
    newt_start : int
        Iteration to start Newton updates. Default is 50.
    newt_ramp : int
        Number of iterations to ramp up Newton. Default is 10.
    newtrate : float
        Newton learning rate multiplier. Default is 1.0.
    do_mean : bool
        Whether to remove data mean. Default is True.
    do_sphere : bool
        Whether to whiten/sphere data. Default is True.
    sphere_type : str
        Type of sphering: "pca" or "zca". Default is "zca".
    do_pca : bool
        Whether to apply PCA dimensionality reduction. Default is True.
    do_approx_sphere : bool
        Use approximate sphering. Default is True.
    mineig : float
        Minimum eigenvalue threshold for PCA. Default is 1e-12.
    do_reject : bool
        Whether to reject outlier samples. Default is False.
    rejsig : float
        Rejection threshold in standard deviations. Default is 3.0.
    rejstart : int
        Iteration to start rejection. Default is 2 (Klug et al. 2024).
    rejint : int
        Interval between rejection passes. Default is 3 (Klug et al. 2024).
    numrej : int
        Number of rejection passes per interval. Default is 5.
    min_dll : float
        Minimum log-likelihood change for convergence. Default is 1e-9.
    max_decs : int
        Number of LL decreases before reducing max learning rates. Default is 3.
    max_incs : int
        Number of small LL increases before stopping. Default is 10.
    use_min_dll : bool
        Whether to use min_dll for convergence. Default is True.
    invsigmax : float
        Maximum inverse sigma for numerical stability. Default is 100.0.
    invsigmin : float
        Minimum inverse sigma for numerical stability. Default is 0.0.
    doscaling : bool
        Whether to rescale A/mu/sbeta each iteration. Default is True.
    writestep : int
        Interval for writing intermediate results. Default is 100.
    outdir : Path, optional
        Output directory for results.
    fix_init : bool
        Use identity matrix initialization instead of random. Default is False.
    update_alpha : bool
        Whether to update mixture weights. Default is True.
    update_mu : bool
        Whether to update location parameters. Default is True.
    update_beta : bool
        Whether to update scale parameters. Default is True.
    update_rho : bool
        Whether to update shape parameters. Default is True.
    """

    # Model structure
    num_models: int = 1
    num_mix_comps: int = 3
    pcakeep: int | None = None
    dtype: str = "float64"  # "float32" or "float64"

    # Iteration control
    max_iter: int = 2000

    # Learning rates
    lrate: float = 0.01  # 0.1 crashes on 60+ component real EEG (both Fortran and Python)
    minlrate: float = 1e-8
    lratefact: float = 0.5
    rholrate: float = 0.05
    rholratefact: float = 0.5

    # Shape parameter bounds
    minrho: float = 1.0
    maxrho: float = 2.0
    rho0: float = 1.5

    # Newton method
    do_newton: bool = True
    newt_start: int = 50
    newt_ramp: int = 10
    newtrate: float = 1.0

    # Preprocessing
    do_mean: bool = True
    do_sphere: bool = True
    sphere_type: str = "zca"
    do_pca: bool = True
    do_approx_sphere: bool = True
    mineig: float = 1e-12

    # Likelihood-based sample rejection (Fortran 1.7 do_reject). Each round drops
    # samples whose per-sample total LL < mean - rejsig*std over the current good set
    # (monotone, never re-accepted; rejected samples are zero-weighted, not removed).
    # Works for M=1 and M>1 — for M>1 a single GLOBAL mask thresholds the mixture LL
    # logsumexp_h[log gm_h + log p(x|h)]. rejsig/rejstart/rejint match Fortran; numrej
    # follows Klug et al. 2024 (Fortran's maxrej default is 1).
    do_reject: bool = False
    rejsig: float = 3.0
    rejstart: int = 2
    rejint: int = 3
    numrej: int = 5

    # Convergence
    min_dll: float = 1e-9
    use_min_dll: bool = True
    max_decs: int = 3  # Match Fortran amicadefs.param (was 5; see AMICA_AUDIT.md F1/F4)
    max_incs: int = 10

    # Numerical stability
    invsigmax: float = 100.0
    invsigmin: float = 1e-8  # Match Fortran amicadefs.param (was 0.0; see AMICA_AUDIT.md F2)

    # Rescaling
    doscaling: bool = True

    # Output
    writestep: int = 100
    outdir: Path | None = None

    # Initialization
    fix_init: bool = False

    # Update flags
    update_alpha: bool = True
    update_mu: bool = True
    update_beta: bool = True
    update_rho: bool = True

    # Time-axis chunking for the E-step accumulator.
    #   "auto"   — (default) pick a chunk size automatically. On CPU this targets a
    #              cache-resident hot buffer (~2x faster than full-batch and far lower peak
    #              RAM; see benchmarks/profile_cpu.py). On GPU it sizes against the VRAM
    #              budget and returns full-batch whenever the data fits, so GPU behaviour is
    #              unchanged except for added OOM safety.
    #   None     — force full-batch (one E-step pass over all samples). Use this to
    #              reproduce the exact pre-0.x full-batch reduction order / trace.
    #   int >= 1 — explicit chunk size in samples
    # Note: "auto"/int reassociate the time-axis reductions vs full-batch — an algebraic
    # identity (same fixed point, ~1e-14 LL agreement), not a byte-identical trace.
    chunk_size: int | Literal["auto"] | None = "auto"

    # Full-batch E-step implementation (advanced; most users should leave "auto"):
    #   "auto"    — use the fused single-pass step (one responsibility pass per
    #               iteration). Default, fastest.
    #   "fused"   — force the fused single-graph step.
    #   "classic" — force the original recompute-3x step. Kept as the parity
    #               oracle / for exact reproduction of pre-fusion results.
    # Only affects the full-batch path; the chunked path is always fused.
    estep: Literal["auto", "fused", "classic"] = "auto"

    def __post_init__(self):
        """Validate configuration."""
        if self.num_models < 1:
            raise ValueError("num_models must be >= 1")
        if self.num_mix_comps < 1:
            raise ValueError("num_mix_comps must be >= 1")
        if self.lrate <= 0:
            raise ValueError("lrate must be > 0")
        if self.minrho < 1.0:
            raise ValueError("minrho must be >= 1.0")
        if self.maxrho > 2.0:
            raise ValueError("maxrho must be <= 2.0")
        if self.minrho > self.maxrho:
            raise ValueError("minrho must be <= maxrho")
        if self.max_decs < 0:
            raise ValueError("max_decs must be >= 0")
        if self.max_incs < 0:
            raise ValueError("max_incs must be >= 0")
        if (
            self.chunk_size is not None
            and self.chunk_size != "auto"
            and (not isinstance(self.chunk_size, int) or self.chunk_size < 1)
        ):
            raise ValueError("chunk_size must be an int >= 1, 'auto', or None")
        if self.estep not in ("auto", "fused", "classic"):
            raise ValueError("estep must be 'auto', 'fused', or 'classic'")
        if self.outdir is not None:
            self.outdir = Path(self.outdir)
