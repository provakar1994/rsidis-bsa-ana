# RSIDIS SSA Analysis Package

Single-spin asymmetry analysis for the Hall C RSIDIS experiment (HMS + SHMS).  
One YAML config file drives one complete analysis (one kinematic setting, one target).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Package Layout](#package-layout)
3. [Module Dependency Tree](#module-dependency-tree)
4. [Workflow: Step by Step](#workflow-step-by-step)
   - [Step 1 — Configuration](#step-1--configuration)
   - [Step 2 — Run Selection](#step-2--run-selection)
   - [Step 3 — Per-Run Normalization Weights](#step-3--per-run-normalization-weights)
   - [Step 4 — ROOT File Reading](#step-4--root-file-reading)
   - [Step 5 — Event Selection Cuts](#step-5--event-selection-cuts)
   - [Step 6 — Histogram Filling](#step-6--histogram-filling)
   - [Step 7 — Run Combination](#step-7--run-combination)
   - [Step 8 — Background Subtraction](#step-8--background-subtraction)
   - [Step 9 — Asymmetry Calculation](#step-9--asymmetry-calculation)
5. [Diagnostic PDF Pages](#diagnostic-pdf-pages)
6. [Configuration Reference](#configuration-reference)
7. [Run Constants Reference](#run-constants-reference)
8. [Branch Name Reference](#branch-name-reference)

---

## Quick Start

```bash
python analysis.py config/example_C_z05_thpq2.yaml
python analysis.py config/example_LH2_z05_thpq2.yaml
```

Each command produces a multi-page PDF (`<config-stem>_diagnostics.pdf`) with
coincidence-time cuts, per-run yield diagnostics, per-histogram subtraction overlays,
and a statistics table.

---

## Package Layout

```
ssa/
├── analysis.py                  # Top-level driver — generates the diagnostic PDF
├── config/
│   ├── example_C_z05_thpq2.yaml
│   ├── example_LH2_z05_thpq2.yaml
│   └── run_constants.yaml       # Run-period constants (dummy scales, beam_bunch_ns, …)
├── data/
│   ├── rsidis_bigtable_pass0p1.csv   # Master runlist
│   └── rootfiles_pass0p1/            # Symlinked or local ROOT skim files
└── rsidis_ssa/
    ├── config_loader.py         # YAML → validated Pydantic models
    ├── runlist.py               # Run selection from the CSV runlist
    ├── normalization.py         # Per-run weight calculation
    ├── reader.py                # ROOT file I/O (uproot)
    ├── cuts.py                  # Event selection masks
    ├── histograms.py            # Histogram registry creation and filling
    ├── pipeline.py              # Orchestrates all of the above
    ├── backgrounds.py           # Random / e⁺ / dummy subtraction
    └── asymmetry.py             # Per-bin A_LU and sin(φ) amplitude fit

tests/
    test_config_loader.py
    test_runlist.py
    test_normalization.py
    test_reader.py
    test_cuts.py
    test_histograms.py
    test_backgrounds.py
    test_asymmetry.py
```

---

## Module Dependency Tree

```
analysis.py
└── rsidis_ssa.pipeline          ← single public entry point
    ├── rsidis_ssa.config_loader ← loads & validates the YAML
    │   └── rsidis_ssa.runlist   (Setting, is_cryo, SIDIS_RUN_TYPES)
    ├── rsidis_ssa.runlist       ← selects runs from the CSV
    ├── rsidis_ssa.normalization ← builds per-run weight tables
    ├── rsidis_ssa.reader        ← reads branches from ROOT files
    ├── rsidis_ssa.cuts          ← builds PID + ctime masks
    └── rsidis_ssa.histograms    ← creates and fills histogram registries
        └── rsidis_ssa.cuts      (effective_helicity, compute_*)

analysis.py
└── rsidis_ssa.backgrounds       ← subtract_randoms / subtract_eplus / subtract_dummy
    └── (operates on bh.Histogram objects returned by pipeline)
```

`pipeline.py` is the only module that calls all others; `backgrounds.py` is pure
arithmetic on boost_histogram objects and has no upstream dependencies within the
package.

---

## Workflow: Step by Step

### Step 1 — Configuration

`config_loader.load_config(yaml_path)` reads the YAML and returns a fully validated
`AnalysisConfig` object.  All tunable parameters live in the YAML — nothing is
hardcoded in the analysis code.

**Key config sections:**

| Section | Purpose |
|---------|---------|
| `setting` | Kinematic setting used to select runs from the runlist |
| `target` | Target material (C, LH2, LD2, …) |
| `run_period` | Key into `run_constants.yaml` for period-specific constants |
| `do_eplus_subtraction` | Whether to subtract e⁺ charge-symmetric background |
| `do_dummy_subtraction` | Whether to subtract dummy-target background (LH2/LD2 only) |
| `rootfiles` | Directory, filename pattern (`{run}` placeholder), tree name |
| `runlist` | Path to the master CSV (resolved relative to the config file) |
| `cuts` | All PID and coincidence-time cut thresholds |
| `histograms` | List of 1-D histogram definitions |

Relative paths (`rootfiles.directory`, `runlist.csv`) are resolved relative to the
directory that contains the YAML file — not the working directory.

---

### Step 2 — Run Selection

`runlist.select_runs(df, setting, target, tol=0.02)` filters the master CSV to the
rows matching the requested setting.  Floating-point kinematic columns are matched
within a tolerance of ±0.02:

```
|ebeam − setting.ebeam| < 0.02  AND
|x    − setting.x|    < 0.02  AND
|Q2   − setting.Q2|   < 0.02  AND
|z    − setting.z|    < 0.02  AND
|thpq − setting.thpq| < 0.02  AND
run_type == setting.run_type (exact)
target   == target            (exact)
```

The matched rows are then split into three sub-sets:

| Sub-set | Filter | Physics |
|---------|--------|---------|
| **signal** | `hms_p < 0` | e⁻ beam → π production |
| **e⁺ background** | `hms_p > 0` | e⁺ beam → charge-symmetric background |
| **dummy** | `target == "Dummy"` + same kinematics | Aluminium cell walls |

---

### Step 3 — Per-Run Normalization Weights

`normalization.build_weight_table(df_runs, apply_boil_corr)` computes one weight
per run.  The weight converts raw event counts into yield per unit luminosity so
that runs with different beam charges, efficiencies, and live-times can be summed
correctly.

**Formula:**

```
w_r = (ps_factor_r × boil_corr_r) / (BCM2_Q_r × h_esing_Eff_r × p_hadron_Eff_r × comp_livetime_r)
```

| Symbol | Column | Units | Typical value |
|--------|--------|-------|---------------|
| `BCM2_Q` | `BCM2_Q` | mC | varies |
| `h_esing_Eff` | `h_esing_Eff` | fraction | ~0.9997 |
| `p_hadron_Eff` | `p_hadron_Eff` | fraction | ~0.93 |
| `comp_livetime` | `comp_livetime` | fraction | ~1.0 |
| `ps_factor` | derived from `ps5`/`ps6` | dimensionless | 1.0 (unprescaled) |
| `boil_corr` | `boil_corr` | fraction ≥ 1 | 1.0 for solid targets |

**`boil_corr`** appears in the numerator because beam heating reduces target density.
A run with lower density contains fewer target nucleons per unit charge, so each
recorded event represents proportionally more cross-section:
`Y_nominal = Y_measured × boil_corr`.  Applied only for LH2/LD2.

**`ps_factor`** is the coincidence trigger prescale.  `ps6` takes precedence over
`ps5`; exactly one must be positive.  In the current dataset the coincidence trigger
is always unprescaled (`ps_factor = 1.0`).

A run is **excluded** from the weight table (and from analysis) if any normalization
column is NaN, zero, or negative.  Excluded runs are recorded in
`WeightTableResult.excluded` with a human-readable reason.

---

### Step 4 — ROOT File Reading

`reader.required_branches(histo_cfgs, cuts_cfg)` computes the minimal set of
branches that must be read from the ROOT tree.  It always includes the PID and
coincidence-time branches, adds histogram fill branches, and adds `BRANCH_HELICITY`
only when at least one histogram has a `helicity_cut`.

`reader.read_branches(root_path, treename, branch_names)` opens the ROOT file with
uproot, retrieves the requested tree, and calls `tree.arrays(..., library="np")`
returning a `dict[str, np.ndarray]`.

- **Missing file** → `FileNotFoundError` (caught by the pipeline, run is skipped)
- **Missing tree or branch** → `KeyError` (fatal, indicates a config mismatch)

If **all** signal runs are missing, the pipeline raises `FileNotFoundError`
immediately rather than producing empty histograms.

---

### Step 5 — Event Selection Cuts

All cuts are applied as numpy boolean masks (no loops; vectorised over all events
in a run at once).

**PID mask** (`cuts.pid_mask`) — HMS electron + SHMS pion identification:

```python
(H_gtr_dp              >= hsdelta_lo)  &  (H_gtr_dp         <= hsdelta_hi)
& (H_cer_npeSum        >= hcer_npe_min)
& (H_cal_etottracknorm >= hsshsum_min)
& (P_gtr_dp            >= psdelta_lo)  &  (P_gtr_dp         <= psdelta_hi)
& (P_aero_npeSum       >= paero_npe_min)
& (P_hgcer_npeSum      >= phgc_npe_min)
& (P_cal_etottracknorm <= psshsum_max)
```

**Real coincidence-time mask** (`cuts.build_real_mask`) — PID AND:

```
|CTime_ePiCoinTime_ROC2 − center| ≤ half_win
```

where `center` and `half_win` are determined per run:

| `ctime_real_center` | center | half_win |
|---------------------|--------|----------|
| `"auto"` | `ctmean` from runlist CSV | `ctime_real_nsigma × ctsigma` (or fallback if NaN) |
| fixed float | that value | `ctime_real_window_fallback` |

**Random sideband mask** (`cuts.build_random_mask`) — PID AND union of discrete
windows around individual beam-bunch peaks:

```
center_k = ctmean − (n_skip + k) × beam_bunch_ns   for k = 1 … n_peaks_lo   (lo side)
center_k = ctmean + (n_skip + k) × beam_bunch_ns   for k = 1 … n_peaks_hi   (hi side)

event passes if |ctime − center_k| ≤ real_half_win  for any k
```

`n_skip ≥ 1` ensures the immediately adjacent beam-bunch peaks are always excluded.
Each random window uses the same half-width as the real peak, so statistics per
window are directly comparable.  Setting `n_peaks_lo = 0` or `n_peaks_hi = 0`
produces a one-sided sideband; `lo + hi` must be ≥ 1.

`beam_bunch_ns` (the beam bunch spacing in ns) is loaded once per run period from
`run_constants.yaml`.

**Effective helicity** (for helicity-split histograms):

```
effective_hel = T_helicity_hel × ihwp_sign
    ihwp_sign = +1  if IHWP == "OUT"
    ihwp_sign = −1  if IHWP == "IN"
```

---

### Step 6 — Histogram Filling

`histograms.build_histogram_registry(histo_cfgs)` creates one
`bh.Histogram(bh.axis.Regular(bins, xmin, xmax), storage=bh.storage.Weight())`
per histogram.  The `Weight` storage accumulates both the sum of weights
(`h.values()`) and the sum of weights-squared (`h.variances()`), which enables
correct statistical uncertainty propagation through all subsequent arithmetic.

`histograms.fill_run(arrays, mask, weight, registry, histo_cfgs, ihwp)` fills every
histogram for one run:

1. If any histogram has a `helicity_cut`, compute `effective_helicity` once.
2. For each histogram, compute the final mask:
   - No cut: `mask` (PID + ctime)
   - `helicity_cut: positive`: `mask & (eff_hel > 0.5)`
   - `helicity_cut: negative`: `mask & (eff_hel < -0.5)`
3. Resolve fill values:
   - Direct branch: `arrays[branch][final_mask]`
   - `__computed__zhad`: `sqrt(P_gtr_p² + m_π²) / H_kin_primary_nu`
   - `__computed__Pt`: `P_gtr_p × sin(P_kin_secondary_th_xq)`
4. `registry[name].fill(values, weight=weight)`

---

### Step 7 — Run Combination

Runs are combined by simply accumulating into the same `bh.Histogram` object.
boost_histogram's `fill()` is additive: calling it N times for N runs produces the
same result as filling all events at once.  No explicit re-weighting after filling
is needed because the per-event weight `w_r` is baked in at fill time.

Two parallel registries are filled per run type:

| Registry | Mask | Weight per event |
|----------|------|-----------------|
| `real` | PID ∩ real ctime window | `w_r` |
| `random` | PID ∩ union of random-peak windows | `w_r × win_scale` |

**`win_scale`** normalises the random sum to one real-window equivalent:

```
win_scale = 1 / (n_peaks_lo + n_peaks_hi)
```

This is a constant (independent of run or window width), because every random window
has the same half-width as the real peak.  The simple subtraction `h_real − h_random`
is then already correctly normalised without any additional scale factor.

For example, with 3 lo + 3 hi random peaks: `win_scale = 1/6 ≈ 0.167`.

---

### Step 8 — Background Subtraction

All subtractions are performed on the combined (all-runs) histograms.  Each step
returns a **new** histogram; inputs are never modified.  The `Weight` storage
ensures variance propagates correctly through every arithmetic operation.

#### 8a — Random-sideband subtraction

```
h_sig = h_real + h_random × (−1.0)
```

The scale factor is 1.0 here because `win_scale` was already absorbed into
the random fill weight (Step 7).

`SubtractionResult.pulls[i] = random_scaled[i] / sqrt(variance_before[i])`
measures how many statistical sigmas the subtracted background is in each bin.
`max_abs_pull` flags any bin where the background dominates the statistical error.

#### 8b — e⁺ charge-symmetric background subtraction

```
h_ep_sub_ran = subtract_randoms(h_ep_real, h_ep_random, scale=1.0)
h_final      = h_sig + h_ep_sub_ran × (−1.0)
```

No additional scale factor: both signal and e⁺ histograms are already normalised
per unit charge by their weight tables.

#### 8c — Dummy-target subtraction (LH2/LD2 only)

```
dummy_scale   = 1 / (dummy_thickness / cryo_wall_thickness)
h_dum_sub_ran = subtract_randoms(h_dum_real, h_dum_random, scale=1.0)
h_final       = h_prev + h_dum_sub_ran × (−dummy_scale)
```

`dummy_scale` accounts for the fact that the dummy target is thicker than the actual
aluminium walls of the cryo cell.  Only the cell-wall fraction is removed.  The
thickness ratios are stored in `config/run_constants.yaml` (see below).

For LH2: `dummy_scale = 1/7.2323 ≈ 0.138`, removing ~14% of the dummy yield
(which is itself ~1–2% of the signal).

**Subtraction order** (when all three are active):
1. Random subtraction (signal and each background independently)
2. e⁺ subtraction
3. Dummy subtraction

---

### Step 9 — Asymmetry Calculation

`rsidis_ssa/asymmetry.py` implements the full per-bin SSA and the sin(φ) amplitude
extraction.  It operates on the **final** background-subtracted, charge-normalised
`bh.Histogram` objects produced by Step 8.

#### 9a — Per-bin raw asymmetry (`compute_raw_asymmetry`)

For each φ bin k with fully-subtracted weighted yields W_+ and W_−:

```
A_raw(φ_k) = (W_+(φ_k) − W_-(φ_k)) / (W_+(φ_k) + W_-(φ_k))
```

Bins where W_+ + W_− ≤ 0 (empty after subtraction) are set to NaN and excluded
from the fit without raising an exception.

**Statistical uncertainty — exact error propagation:**

```
σ²_A(φ_k) = 4 [W_+²(φ_k) Var(W_-,φ_k) + W_-²(φ_k) Var(W_+,φ_k)]
             ──────────────────────────────────────────────────────
                           (W_+(φ_k) + W_-(φ_k))⁴
```

This is exact first-order error propagation through the asymmetry formula; no
approximation is made.  The variances `Var(W_±, φ_k)` are read directly from
boost_histogram's `Weight` storage via `h.variances()`.

**What is stored in the variances?**

The `Weight` storage tracks both the weighted sum and the sum of squared weights
separately.  Every arithmetic operation on the histogram (subtraction, scaling)
propagates these through the standard rules:

| Operation | New variance |
|-----------|-------------|
| Fill with weight w | `Var += w²` |
| `h *= c` | `Var *= c²` |
| `h = h_a + h_b × (−1)` | `Var = Var_a + Var_b` |

**Two-stage weighting scheme** (important for understanding the variance):

Events are *not* filled with the full per-run weight `w_r = eff_scale_r / Q_r`.
Instead the pipeline uses a deliberate two-stage approach:

1. **Fill stage** — each event is filled with the charge-independent factor
   `eff_scale_r = ps_r × boil_r / (h_e_r × p_h_r × lt_r)`.  Q_r is *excluded*.
2. **Post-loop rescaling** — after all runs are accumulated the entire histogram is
   multiplied by `1/Q_tot` where `Q_tot = Σ_r Q_r`.

This factoring is correct because it gives the same final yield as filling with
`w_r` individually, while avoiding a problem with variance: if Q_r varied greatly
between runs, filling with the full `w_r` would cause high-charge runs to have
disproportionately small variance per event, skewing the combined uncertainty.
By using `eff_scale_r` (which varies only due to efficiencies and live-time) the
variance is dominated by counting statistics, not by charge fluctuations.

After the full pipeline the variance in each bin contains:

```
Var(W_±, φ_k) = (1/Q_tot)² × [ Σ_r  eff_scale_r²  × n_r^±(φ_k)
                                + Σ_r  (eff_scale_r × win_scale)² × m_r^±(φ_k) ]
```
for signal-only; with e⁺ and dummy contributions added analogously when those
subtractions are active (each uses its own `eff_scale_r` and `Q_tot`).

where:
- `eff_scale_r = ps_r × boil_r / (h_e_r × p_h_r × lt_r)` — charge-independent scale (Step 7)
- `Q_tot = Σ_r Q_r` — total beam charge over all valid runs [mC]
- `win_scale = 1/(n_peaks_lo + n_peaks_hi)` — random-sideband area ratio (Step 7)
- `n_r^±(φ_k)` — raw event count in bin k, helicity ±, run r, passing all cuts
- `m_r^±(φ_k)` — raw event count across all random-sideband windows

**Why edge bins have larger errors:**

The dominant effect is **detector acceptance**: the HMS + SHMS geometric acceptance
falls off near φ = ±π, giving fewer events per bin.  Statistical uncertainty scales
as `1/√(effective N)` and is inherently larger where acceptance is low.

A secondary contribution comes from `eff_scale` variation: the per-bin variance is
`Σ_r eff_scale_r² n_r` rather than `(Σ_r n_r) × eff_scale_avg²`.  Where one run
dominates a bin (because the others have near-zero acceptance there), the single
run's `eff_scale_r²` drives the variance — but `eff_scale` variation between runs
is typically small (a few percent), so this effect is minor compared with the raw
counting statistics.

#### 9b — Physics asymmetry

```
A_phys(φ_k) = A_raw(φ_k) / P_beam
σ_A_phys(φ_k) = σ_A_raw(φ_k) / P_beam
```

`P_beam` is loaded from `run_constants.yaml` (key `beam_polarization` in the run
period block).

#### 9c — sin(φ) amplitude (`fit_sinphi`)

The amplitude A_LU^sinφ is extracted by **analytic single-parameter weighted
least-squares**, fitting the model f(φ_k) = A × sin(φ_k) to A_phys(φ_k).

Bins with NaN A_phys, NaN σ_A_phys, or σ_A_phys = 0 are excluded.  If fewer than
2 valid bins remain all output quantities are NaN.

```
         Σ_k  A_phys(φ_k) × sin(φ_k) / σ²_k
A_fit  = ────────────────────────────────────
              Σ_k  sin²(φ_k) / σ²_k

σ_fit  =  1 / √(Σ_k  sin²(φ_k) / σ²_k)

χ²/ndf = [Σ_k  (A_phys(φ_k) − A_fit × sin(φ_k))² / σ²_k] / (N_valid − 1)
```

No external solver (scipy) is used; the solution is closed-form.

---

#### Comparison with `daveg/SSA/ssa.py`

Both implementations use the same algebraic formula for σ²_A.  The key differences
are in what the histogram values and variances actually contain.

| Aspect | `daveg/SSA/ssa.py` | This workflow |
|--------|-------------------|---------------|
| Yield used for asymmetry | Raw boost_histogram fills from `makehelhistos` at the "after random subtraction" step (`histos[14/15, 3]`) | Fully-subtracted, charge-normalised bh.Histogram from the complete pipeline |
| Normalization weights | Charge normalisation applied globally (÷ Q_tot); individual events likely filled with weight = 1/Q_per_run or similar | Two-stage: fill with `eff_scale_r = ps × boil / (h_e × p_h × lt)` (no Q_r), then divide all histograms by `Q_tot = Σ Q_r` after the loop |
| Variance from backgrounds | Random subtraction propagated via Weight storage | Random + e⁺ + dummy subtraction all propagated via Weight storage |
| Amplitude fit | `scipy.optimize.curve_fit` — numerical χ² minimisation | Analytic WLS (closed-form, equivalent result for a single-parameter linear model) |
| χ²/ndf denominator | Not explicitly computed | N_valid − 1 |
| Bins with σ = 0 | The `+0.000001` guard in the denominator of A_raw prevents division-by-zero but propagates a spuriously small error | Excluded entirely (masked to NaN before fit) |

The `+0.000001` hack in Dave's code (`phi_asy = ... / (phisum + 0.000001)`) does not
affect non-empty bins meaningfully, but it also does not set empty bins to NaN, so
they appear as A ≈ 0 with σ ≈ ∞ in the fit — which scipy's curve_fit down-weights
via the sigma argument.  This workflow's approach (explicit NaN for empty bins,
excluded from WLS) is equivalent in practice but cleaner.

---

## Diagnostic PDF Pages

| Page | Content |
|------|---------|
| 1 | **Coincidence-time distribution** — left: full range with discrete random-peak windows shaded (one span per peak, per-run thin lines); centre: zoom on real peak with per-run window boundaries and config text box; right: ctmean ± ctsigma vs run from CSV with weighted-mean fit |
| 2 | **Normalized yield & run diagnostics** — top row: workflow vs CSV normyield per run for each active run type, flagged runs (>2% residual) marked in red; bottom row: raw event-count comparison (n_real, n_rand×scale, n_rsc vs CSV coin/randoms/ransubcoin); right panel: weight distributions and beam current + normalization components vs run |
| 3 … N | **Per-histogram subtraction** — one page per histogram, 2–4 panels: "Before random sub" → "After random sub" → "After e⁺ sub" → "After dummy sub". Each panel overlays the relevant background histogram scaled to what is actually subtracted (dummy overlay shown as `dummy×scale − random`) |
| N+1 | **Beam SSA** — one row per helicity-split pair (e.g. `phipq_hplus`/`phipq_hminus`): left panel shows A_phys vs φ on auto-scale; right panel shows the same data zoomed to y ∈ (−0.1, 0.1).  Both panels overlay the best-fit A × sin(φ) curve.  Info box: amplitude ± σ, χ²/ndf, N_bins used, P_beam. |
| Last | **Statistics table** — run counts, exclusion reasons, file-skip list, full subtraction summary (fraction subtracted %, max\|pull\| per histogram per subtraction step) |

---

## Configuration Reference

```yaml
setting:
  ebeam:    8.5831          # beam energy [GeV]
  x:        0.25            # Bjorken-x
  Q2:       3.3             # Q² [GeV²]
  z:        0.5             # z_had
  thpq:     2.0             # θ_pq [deg]
  run_type: PI-SIDIS        # PI-SIDIS or PI+SIDIS

target:     LH2             # C, LH2, LD2, …
run_period: period_1        # key into config/run_constants.yaml

do_eplus_subtraction: true
do_dummy_subtraction: true  # LH2/LD2 only; enforced by validator

rootfiles:
  directory: ../data/rootfiles_pass0p1   # relative to this YAML file
  pattern:   skimmed_coin_replay_production_{run}_-1.root
  treename:  T

runlist:
  csv: ../data/rsidis_bigtable_pass0p1.csv  # relative to this YAML file

normalization:
  charge_column: BCM2_Q   # BCM1_Q / BCM2_Q / BCM4A_Q / BCM4B_Q / BCM4C_Q

cuts:
  # HMS electron PID
  hsdelta_lo:    -8.0      # HMS δ lower bound [%]
  hsdelta_hi:     8.0      # HMS δ upper bound [%]
  hcer_npe_min:   1.0      # HMS Cherenkov NPE threshold
  hsshsum_min:    0.7      # HMS calorimeter E/p minimum

  # SHMS pion PID
  psdelta_lo:   -10.0      # SHMS δ lower bound [%]
  psdelta_hi:    20.0      # SHMS δ upper bound [%]
  paero_npe_min:  2.0      # SHMS aerogel NPE threshold
  phgc_npe_min:   1.0      # SHMS HGC NPE threshold
  psshsum_max:    0.8      # SHMS calorimeter E/p maximum (pion rejection)

  # Coincidence time — real peak
  ctime_real_center:          auto   # "auto" → use ctmean from CSV; or fixed ns value
  ctime_real_nsigma:          3.0    # half-window = nsigma × ctsigma  (auto mode only)
                                     # set to null to always use ctime_real_window_fallback
  ctime_real_window_fallback: 2.0    # half-window [ns] when ctsigma is NaN or nsigma is null

  # Coincidence time — random sideband (discrete beam-bunch peak windows)
  # Peak centers:  ctmean ± (n_skip + k) × beam_bunch_ns   for k = 1 … n_peaks
  # win_scale = 1 / (n_peaks_lo + n_peaks_hi)  — constant, independent of run
  ctime_random_n_skip:     1   # beam-bunch peaks to skip on each side (≥ 1)
  ctime_random_n_peaks_lo: 3   # random peaks on the low-ctime side  (≥ 0; lo+hi ≥ 1)
  ctime_random_n_peaks_hi: 3   # random peaks on the high-ctime side (≥ 0; lo+hi ≥ 1)
  # One-sided example: n_peaks_lo: 6, n_peaks_hi: 0

histograms:
  - name:   phipq
    branch: P_kin_secondary_ph_xq  # direct ROOT tree branch
    bins:   16
    xmin:   -3.14159265
    xmax:    3.14159265
    xlabel: "#phi_{pq} (rad)"

  - name:   phipq_hplus
    branch: P_kin_secondary_ph_xq
    bins:   16
    xmin:   -3.14159265
    xmax:    3.14159265
    xlabel: "#phi_{pq} (rad)  [h+]"
    helicity_cut: positive   # fill only when effective helicity > 0

  - name:   phipq_hminus
    branch: P_kin_secondary_ph_xq
    bins:   16
    xmin:   -3.14159265
    xmax:    3.14159265
    xlabel: "#phi_{pq} (rad)  [h-]"
    helicity_cut: negative   # fill only when effective helicity < 0
```

---

## Run Constants Reference

`config/run_constants.yaml` holds run-period-specific constants that are not part of
the event-selection config.  It must sit alongside the analysis YAML.

```yaml
period_1:
  beam_bunch_ns:     4.008   # beam bunch spacing [ns] — places random sideband peaks
  beam_polarization: 0.85    # |P_e| in (0, 1] — converts A_raw → A_phys
  dummy_scale:
    LH2: 7.2323              # dummy / LH2-cell-wall thickness ratio
    LD2: 7.7552              # dummy / LD2-cell-wall thickness ratio
```

`beam_bunch_ns` is loaded once at pipeline startup (`cfg.beam_bunch_ns(config_dir)`)
and stored in `PipelineResult.beam_bunch_ns` for use by the diagnostic plots.

`beam_polarization` is loaded by `cfg.beam_polarization(config_dir)` when the
asymmetry page is generated.  It must be in (0, 1]; a value outside this range raises
`ValueError`.

`dummy_scale` values are loaded only when `do_dummy_subtraction: true`.  The
subtraction scale applied to the dummy histogram is `1 / dummy_scale[target]`.

---

## Branch Name Reference

| Constant | ROOT branch | Description |
|----------|-------------|-------------|
| `BRANCH_HSDELTA` | `H_gtr_dp` | HMS focal-plane δ [%] |
| `BRANCH_HCER_NPE` | `H_cer_npeSum` | HMS Cherenkov NPE sum |
| `BRANCH_HETOTTRACKNORM` | `H_cal_etottracknorm` | HMS calorimeter E/p |
| `BRANCH_PSDELTA` | `P_gtr_dp` | SHMS focal-plane δ [%] |
| `BRANCH_PAERO_NPE` | `P_aero_npeSum` | SHMS aerogel Cherenkov NPE |
| `BRANCH_PHGC_NPE` | `P_hgcer_npeSum` | SHMS heavy-gas Cherenkov NPE |
| `BRANCH_PETOTTRACKNORM` | `P_cal_etottracknorm` | SHMS calorimeter E/p |
| `BRANCH_CTIME` | `CTime_ePiCoinTime_ROC2` | e−π coincidence time [ns] |
| `BRANCH_HELICITY` | `T_helicity_hel` | Raw helicity (+1 / −1) |
| `BRANCH_PPi` | `P_gtr_p` | SHMS pion momentum [GeV/c] |
| `BRANCH_NU` | `H_kin_primary_nu` | Virtual-photon energy ν [GeV] |
| `BRANCH_THETA_PQ` | `P_kin_secondary_th_xq` | θ_pq [rad] |
