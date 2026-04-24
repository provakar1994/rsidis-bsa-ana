"""
YAML configuration loading and validation.

One YAML file drives one complete analysis (one setting, one target).
All tunable parameters live here — nothing is hardcoded in the analysis code.

Usage
-----
    from rsidis_ssa.config_loader import load_config
    cfg = load_config("config/my_setting.yaml")
    setting = cfg.to_setting()
    rootfile_path = cfg.rootfiles.get_path(run_number)

Adding a histogram
------------------
Add one block under 'histograms:' in the YAML — no code changes needed:

    - name:   my_new_var
      branch: P_some_branch
      bins:   50
      xmin:   0.0
      xmax:   1.0
      xlabel: "My label"

Computed quantities (not a direct ROOT branch) use the __computed__ prefix:

    - name:   zhad
      branch: __computed__zhad
      ...

Unknown YAML keys raise a ValidationError immediately at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Filename of the run-period constants file, looked up as a sibling of the
# analysis YAML (i.e. in the same directory as the config file).
RUN_CONSTANTS_FILENAME = "run_constants.yaml"

from rsidis_ssa.runlist import SIDIS_RUN_TYPES, Setting, is_cryo


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class SettingConfig(BaseModel):
    """Kinematic setting — mirrors runlist.Setting for unambiguous conversion."""
    model_config = ConfigDict(extra="forbid")

    ebeam:    float
    x:        float
    Q2:       float
    z:        float
    thpq:     float
    run_type: Literal["PI-SIDIS", "PI+SIDIS"]

    def to_setting(self) -> Setting:
        return Setting(
            ebeam=self.ebeam,
            x=self.x,
            Q2=self.Q2,
            z=self.z,
            thpq=self.thpq,
            run_type=self.run_type,
        )


class RootfilesConfig(BaseModel):
    """Location and naming convention of the ROOT replay files."""
    model_config = ConfigDict(extra="forbid")

    directory: str
    pattern:   str    # must contain {run}, e.g. skimmed_coin_replay_{run}_-1.root
    treename:  str = "T"

    @field_validator("pattern")
    @classmethod
    def pattern_must_contain_run_placeholder(cls, v: str) -> str:
        if "{run}" not in v:
            raise ValueError(
                f"rootfiles.pattern must contain '{{run}}' placeholder, got: {v!r}"
            )
        return v

    def get_path(self, run: int, config_dir: Optional[Path] = None) -> Path:
        """
        Return the full path to the ROOT file for *run*.

        Relative *directory* paths are resolved against *config_dir* (the
        directory containing the analysis YAML).  If config_dir is None the
        current working directory is used.
        """
        d = Path(self.directory)
        if not d.is_absolute():
            base = config_dir if config_dir is not None else Path.cwd()
            d = (base / d).resolve()
        return d / self.pattern.format(run=run)


class RunlistConfig(BaseModel):
    """Path to the run-table CSV."""
    model_config = ConfigDict(extra="forbid")

    csv: str


class NormalizationConfig(BaseModel):
    """Controls which columns are used for luminosity normalization."""
    model_config = ConfigDict(extra="forbid")

    charge_column: Literal[
        "BCM2_Q", "BCM1_Q", "BCM4A_Q", "BCM4B_Q", "BCM4C_Q"
    ] = "BCM2_Q"

    weight_scheme: Literal["eff_corrected_counts", "eff_corrected_charge"] = "eff_corrected_counts"
    """
    ``eff_corrected_counts`` (default):
        Fill each event with ``eff_scale_r = ps × boil / (h_e × p_h × lt)``.
        Divide histograms by ``Q_tot = Σ Q_r`` after the loop.
        Bin variance: ``Σ_r eff_scale_r² × n_rk / Q_tot²``.

    ``eff_corrected_charge``:
        Fill each event with weight 1 (raw counts).
        Divide histograms by ``Q_eff_tot = Σ_r Q_r × h_e_r × p_h_r × lt_r / (ps_r × boil_r)``.
        Bin variance: ``Σ_r n_rk / Q_eff_tot²``  — pure Poisson, minimum variance.
        Equivalent to ``eff_corrected_counts`` when efficiencies are uniform across runs.
    """


# Coincidence-time center: either the string 'auto' (use ctmean from CSV)
# or a fixed float in nanoseconds.
_CtimeCenter = Annotated[
    Union[Literal["auto"], float],
    Field(union_mode="left_to_right"),
]


class CutsConfig(BaseModel):
    """
    All event-selection cuts.  Nothing is hardcoded in the filling code.

    *_lo / *_hi        — acceptance range bounds in %
    *_min              — minimum value for PID variables
    ctime_real_center  — 'auto' uses the per-run ctmean from the CSV;
                         a float fixes the center for all runs.
    ctime_real_nsigma  — half-width = nsigma × ctsigma  (auto mode only).
                         Set to null to always use ctime_real_window_fallback,
                         even when a valid ctsigma is available in the CSV.
    ctime_real_window_fallback — fixed half-width [ns] used when ctsigma is
                         NaN/missing, or when ctime_real_nsigma is null.

    Random sideband — discrete windows around individual beam-bunch peaks:
    ctime_random_n_skip     — number of beam-bunch peaks to skip on each side
                              of the real peak before starting the random
                              windows.  Must be ≥ 1 (the immediately adjacent
                              peaks are always excluded).
    ctime_random_n_peaks_lo — number of random peaks to use on the low-ctime
                              (negative-offset) side of the real peak.  0 means
                              no peaks on this side (one-sided sideband).
    ctime_random_n_peaks_hi — number of random peaks to use on the high-ctime
                              (positive-offset) side.  0 means no peaks on this
                              side.  lo + hi must be ≥ 1.

    The k-th random peak center on the lo side:
        center_k = ctmean − (n_skip + k) × beam_bunch_ns   for k = 1 … n_peaks_lo

    Each random window uses the same half-width as the real peak.
    win_scale = 1 / (n_peaks_lo + n_peaks_hi)  (area ratio, constant per run).
    """
    model_config = ConfigDict(extra="forbid")

    # HMS electron PID
    hsdelta_lo:    float
    hsdelta_hi:    float
    hcer_npe_min:  float
    hsshsum_min:   float

    # SHMS pion PID
    psdelta_lo:    float
    psdelta_hi:    float
    paero_npe_min: float
    phgc_npe_min:  float
    psshsum_max:   float   # SHMS calorimeter E/p upper cut (pion rejection)

    # Coincidence time — real peak
    ctime_real_center:          _CtimeCenter
    ctime_real_nsigma:          Optional[float]   # null → always use ctime_real_window_fallback
    ctime_real_window_fallback: float

    # Coincidence time — random sideband (discrete peaks, relative to real peak)
    ctime_random_n_skip:     int   # beam-bunch peaks to skip each side of real peak (≥ 1)
    ctime_random_n_peaks_lo: int   # random peaks to use on the low-ctime side (≥ 0; total lo+hi ≥ 1)
    ctime_random_n_peaks_hi: int   # random peaks to use on the high-ctime side (≥ 0; total lo+hi ≥ 1)

    @model_validator(mode="after")
    def hsdelta_lo_lt_hi(self) -> "CutsConfig":
        if self.hsdelta_lo >= self.hsdelta_hi:
            raise ValueError(
                f"hsdelta_lo ({self.hsdelta_lo}) must be < hsdelta_hi ({self.hsdelta_hi})"
            )
        return self

    @model_validator(mode="after")
    def psdelta_lo_lt_hi(self) -> "CutsConfig":
        if self.psdelta_lo >= self.psdelta_hi:
            raise ValueError(
                f"psdelta_lo ({self.psdelta_lo}) must be < psdelta_hi ({self.psdelta_hi})"
            )
        return self

    @field_validator(
        "hcer_npe_min", "hsshsum_min", "paero_npe_min", "phgc_npe_min",
        "ctime_real_window_fallback",
        mode="after",
    )
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Value must be > 0, got {v}")
        return v

    @field_validator("ctime_random_n_skip", mode="after")
    @classmethod
    def n_skip_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"ctime_random_n_skip must be ≥ 1, got {v}")
        return v

    @field_validator("ctime_random_n_peaks_lo", "ctime_random_n_peaks_hi", mode="after")
    @classmethod
    def n_peaks_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Value must be ≥ 0, got {v}")
        return v

    @model_validator(mode="after")
    def total_random_peaks_must_be_positive(self) -> "CutsConfig":
        total = self.ctime_random_n_peaks_lo + self.ctime_random_n_peaks_hi
        if total < 1:
            raise ValueError(
                f"ctime_random_n_peaks_lo + ctime_random_n_peaks_hi must be ≥ 1, "
                f"got {self.ctime_random_n_peaks_lo} + {self.ctime_random_n_peaks_hi} = {total}"
            )
        return self

    @field_validator("ctime_real_nsigma", mode="after")
    @classmethod
    def nsigma_positive_if_set(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"ctime_real_nsigma must be > 0 when set, got {v}")
        return v

    @property
    def real_window_ns(self) -> Optional[float]:
        """
        Fixed real-coincidence half-window in ns.
        None when ctime_real_center == 'auto' (window is computed per-run
        from nsigma × ctsigma at fill time).
        """
        if self.ctime_real_center == "auto":
            return None
        return self.ctime_real_window_fallback


class HistogramConfig(BaseModel):
    """Definition of one 1-D histogram."""
    model_config = ConfigDict(extra="forbid")

    name:         str
    branch:       str
    bins:         int   = Field(gt=0)
    xmin:         float
    xmax:         float
    xlabel:       str   = ""
    helicity_cut: Optional[Literal["positive", "negative"]] = None

    @field_validator("xmax")
    @classmethod
    def xmax_gt_xmin(cls, v: float, info) -> float:
        xmin = info.data.get("xmin")
        if xmin is not None and v <= xmin:
            raise ValueError(f"xmax ({v}) must be > xmin ({xmin})")
        return v

    @property
    def is_computed(self) -> bool:
        """True when the quantity must be calculated, not read directly from the tree."""
        return self.branch.startswith("__computed__")

    @property
    def computed_name(self) -> Optional[str]:
        """
        The name of the computed quantity (e.g. 'zhad'), or None for direct branches.
        Used by the filling code to dispatch to the right computation function.
        """
        if self.is_computed:
            return self.branch[len("__computed__"):]
        return None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class AnalysisConfig(BaseModel):
    """
    Complete configuration for one SSA analysis job.

    Loaded from YAML via load_config().  Unknown keys raise immediately.
    """
    model_config = ConfigDict(extra="forbid")

    setting:              SettingConfig
    target:               str
    run_period:           str
    do_eplus_subtraction: bool = True
    do_dummy_subtraction: bool = False
    runs_include:         list[int] | None = None   # None = all runs
    runs_exclude:         list[int] = Field(default_factory=list)
    rootfiles:            RootfilesConfig
    runlist:              RunlistConfig
    normalization:        NormalizationConfig = Field(default_factory=NormalizationConfig)
    cuts:                 CutsConfig
    histograms:           list[HistogramConfig] = Field(min_length=1)

    # ---- cross-field validators ----

    @model_validator(mode="after")
    def dummy_subtraction_requires_cryo_target(self) -> "AnalysisConfig":
        if self.do_dummy_subtraction and not is_cryo(self.target):
            raise ValueError(
                f"do_dummy_subtraction=true but target='{self.target}' is not a cryo "
                f"target (LH2 or LD2).  Set do_dummy_subtraction: false for solid targets."
            )
        return self

    @model_validator(mode="after")
    def no_duplicate_histogram_names(self) -> "AnalysisConfig":
        names = [h.name for h in self.histograms]
        seen, dupes = set(), []
        for n in names:
            if n in seen:
                dupes.append(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"Duplicate histogram name(s): {dupes}.  "
                "Each histogram must have a unique 'name'."
            )
        return self

    @model_validator(mode="after")
    def runs_include_exclude_consistent(self) -> "AnalysisConfig":
        if self.runs_include is not None and self.runs_exclude:
            overlap = set(self.runs_include) & set(self.runs_exclude)
            if overlap:
                import logging
                logging.getLogger(__name__).warning(
                    "runs_exclude supersedes runs_include for run(s): %s",
                    sorted(overlap),
                )
        return self

    # ---- convenience helpers ----

    @property
    def apply_boil_corr(self) -> bool:
        """Derived from target: True for LH2/LD2, False for everything else."""
        return is_cryo(self.target)

    def to_setting(self) -> Setting:
        """Convert the setting section to a runlist.Setting for run selection."""
        return self.setting.to_setting()

    def resolve_runlist_path(self, config_dir: Optional[Path] = None) -> Path:
        """
        Resolve runlist.csv to an absolute Path.

        Relative paths are resolved against *config_dir* (the directory
        that contains the YAML file).  If config_dir is None the current
        working directory is used.
        """
        p = Path(self.runlist.csv)
        if p.is_absolute():
            return p
        base = config_dir if config_dir is not None else Path.cwd()
        return (base / p).resolve()

    def resolve_run_constants_path(self, config_dir: Optional[Path] = None) -> Path:
        """
        Return the path to run_constants.yaml, which must sit alongside the
        analysis YAML in the same directory.
        """
        base = config_dir if config_dir is not None else Path.cwd()
        return (base / RUN_CONSTANTS_FILENAME).resolve()

    def dummy_scale(self, config_dir: Optional[Path] = None) -> float:
        """
        Return the dummy-to-cryo-wall thickness ratio for this target and run
        period, loaded from run_constants.yaml.

        The subtraction scale applied to the dummy histogram is 1 / ratio, so
        that only the aluminium cell-wall fraction is removed:

            h_cryo_wall = h_dummy / ratio
            h_final     = h_data  − h_cryo_wall

        Raises FileNotFoundError if run_constants.yaml is not found.
        Raises KeyError if the run_period or target is not listed.
        """
        constants_path = self.resolve_run_constants_path(config_dir)
        if not constants_path.exists():
            raise FileNotFoundError(
                f"run_constants.yaml not found: {constants_path}\n"
                f"Expected alongside the analysis config in {constants_path.parent}"
            )
        with constants_path.open() as fh:
            data = yaml.safe_load(fh)

        if self.run_period not in data:
            raise KeyError(
                f"run_period '{self.run_period}' not found in {constants_path}. "
                f"Available periods: {list(data.keys())}"
            )
        scales = data[self.run_period].get("dummy_scale", {})
        if self.target not in scales:
            raise KeyError(
                f"No dummy_scale for target '{self.target}' in period "
                f"'{self.run_period}' of {constants_path}. "
                f"Available targets: {list(scales.keys())}"
            )
        return float(scales[self.target])

    def beam_bunch_ns(self, config_dir: Optional[Path] = None) -> float:
        """
        Return the beam bunch spacing [ns] for this run period, loaded from
        run_constants.yaml.

        The random sideband half-window is:
            random_half_win = ctime_random_wscale × (beam_bunch_ns / 2)

        Raises FileNotFoundError if run_constants.yaml is not found.
        Raises KeyError if the run_period or beam_bunch_ns key is missing.
        """
        constants_path = self.resolve_run_constants_path(config_dir)
        if not constants_path.exists():
            raise FileNotFoundError(
                f"run_constants.yaml not found: {constants_path}\n"
                f"Expected alongside the analysis config in {constants_path.parent}"
            )
        with constants_path.open() as fh:
            data = yaml.safe_load(fh)
        if self.run_period not in data:
            raise KeyError(
                f"run_period '{self.run_period}' not found in {constants_path}."
            )
        period = data[self.run_period]
        if "beam_bunch_ns" not in period:
            raise KeyError(
                f"'beam_bunch_ns' not found in period '{self.run_period}' of "
                f"{constants_path}. Add it as a top-level key under the period."
            )
        return float(period["beam_bunch_ns"])

    def beam_polarization(self, config_dir: Optional[Path] = None) -> float:
        """
        Return the beam polarization magnitude |P_e| for this run period,
        loaded from run_constants.yaml.

        Used to convert the raw beam-helicity asymmetry to the physics asymmetry:
            A_phys = A_raw / beam_polarization

        Raises FileNotFoundError if run_constants.yaml is not found.
        Raises KeyError if the run_period or beam_polarization key is missing.
        """
        constants_path = self.resolve_run_constants_path(config_dir)
        if not constants_path.exists():
            raise FileNotFoundError(
                f"run_constants.yaml not found: {constants_path}\n"
                f"Expected alongside the analysis config in {constants_path.parent}"
            )
        with constants_path.open() as fh:
            data = yaml.safe_load(fh)
        if self.run_period not in data:
            raise KeyError(
                f"run_period '{self.run_period}' not found in {constants_path}."
            )
        period = data[self.run_period]
        if "beam_polarization" not in period:
            raise KeyError(
                f"'beam_polarization' not found in period '{self.run_period}' of "
                f"{constants_path}. Add it as a top-level key under the period."
            )
        val = float(period["beam_polarization"])
        if not (0.0 < val <= 1.0):
            raise ValueError(
                f"beam_polarization must be in (0, 1], got {val} "
                f"(period '{self.run_period}' of {constants_path})"
            )
        return val

    def histogram_names(self) -> list[str]:
        return [h.name for h in self.histograms]

    def get_histogram(self, name: str) -> HistogramConfig:
        for h in self.histograms:
            if h.name == name:
                return h
        raise KeyError(f"No histogram named {name!r}")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path) -> AnalysisConfig:
    """
    Load and validate an analysis config from a YAML file.

    Parameters
    ----------
    yaml_path : path-like
        Path to the YAML config file.

    Returns
    -------
    AnalysisConfig

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If validation fails (wrong types, unknown keys, logical errors).
        The message includes the file path and Pydantic's full error detail.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open() as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file {yaml_path} did not parse to a YAML mapping (got {type(raw).__name__})"
        )

    try:
        return AnalysisConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(
            f"Invalid config file {yaml_path}:\n{exc}"
        ) from exc
