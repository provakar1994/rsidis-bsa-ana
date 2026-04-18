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
5. [Diagnostic PDF Pages](#diagnostic-pdf-pages)
6. [Configuration Reference](#configuration-reference)
7. [Branch Name Reference](#branch-name-reference)

---

## Quick Start

```bash
python analysis.py config/example_C_z05_thpq2.yaml
python analysis.py config/example_LH2_z05_thpq2.yaml
```

Each command produces a multi-page PDF (`<config-stem>_diagnostics.pdf`) with weight
distributions, coincidence-time cuts, per-histogram subtraction overlays, and a
statistics table.

---

## Package Layout

```
ssa/
├── analysis.py                  # Top-level driver — generates the diagnostic PDF
├── config/
│   ├── example_C_z05_thpq2.yaml
│   ├── example_LH2_z05_thpq2.yaml
│   └── run_constants.yaml       # Run-period-specific constants (dummy scales, …)
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
    └── (asymmetry.py            # Phase 6 — not yet implemented)

tests/
    test_config_loader.py
    test_runlist.py
    test_normalization.py
    test_reader.py
    test_cuts.py
    test_histograms.py
    test_backgrounds.py
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
(H_gtr_dp         >= hsdelta_lo)  &  (H_gtr_dp         <= hsdelta_hi)
& (H_cer_npeSum   >= hcer_npe_min)
& (H_cal_etottracknorm >= hsshsum_min)
& (P_gtr_dp       >= psdelta_lo)  &  (P_gtr_dp         <= psdelta_hi)
& (P_aero_npeSum  >= paero_npe_min)
& (P_hgcer_npeSum >= phgc_npe_min)
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

**Random sideband mask** (`cuts.build_random_mask`) — PID AND:

```
|CTime_ePiCoinTime_ROC2 − ctime_random_center| ≤ ctime_random_window
```

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
   - `branch: z` or `branch: pt` — read directly from ROOT tree like any other branch
   - `__computed__zhad`: `sqrt(P_gtr_p² + m_π²) / H_kin_primary_nu` (m_π = 0.13957018 GeV/c²)
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
| `random` | PID ∩ random sideband | `w_r × win_scale` |

**`win_scale`** absorbs the difference in window widths so that a simple subtraction
`h_real − h_random` is already correctly normalised:

```
win_scale = real_half_win / random_half_win
```

For example, with a 3σ real window (~1.1 ns half-width) and a ±6 ns random window:
`win_scale ≈ 1.1 / 6.0 ≈ 0.183`.  The random histogram is therefore scaled down to
represent the same exposure as the real window before subtraction.

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
scale        = 1 / (dummy_thickness / cryo_wall_thickness)
h_dum_sub_ran = subtract_randoms(h_dum_real, h_dum_random, scale=1.0)
h_final       = h_prev + h_dum_sub_ran × (−scale)
```

`scale` accounts for the fact that the dummy target is thicker than the actual
aluminium walls of the cryo cell.  The thickness ratios are stored in
`config/run_constants.yaml` under the relevant run period:

```yaml
period_1:
  dummy_scale:
    LH2: 7.2323   # dummy / LH2-wall thickness ratio
    LD2: 7.7552
```

For LH2: `scale = 1/7.2323 ≈ 0.138`, removing only ~1.9% of the signal yield.

**Subtraction order** (when all three are active):
1. Random subtraction (signal and each background separately)
2. e⁺ subtraction
3. Dummy subtraction

---

## Diagnostic PDF Pages

| Page | Function | Content |
|------|----------|---------|
| 1 | `_page_weights` | Histogram of per-run weights for signal and e⁺; mean marked; exclusions noted |
| 2 | `_page_ctime` | **Left**: full ctime distribution with real + random windows shaded. **Right**: zoom on real peak, individual per-run windows shown as gray lines, config parameters in text box |
| 3 … N | `_page_histogram` | One page per histogram. 2–4 panels depending on which subtractions are active (random / e⁺ / dummy). Each panel shows the before/after state with overlays |
| Last | `_page_statistics` | Run counts, exclusion reasons, file-skip list, full subtraction summary table (fraction subtracted %, max\|pull\|) |

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
  hsdelta_lo:    -8.0      # HMS δ lower bound [%]
  hsdelta_hi:     8.0      # HMS δ upper bound [%]
  hcer_npe_min:   1.0      # HMS Cherenkov NPE threshold
  hsshsum_min:    0.7      # HMS calorimeter E/p threshold
  psdelta_lo:   -10.0      # SHMS δ lower bound [%]
  psdelta_hi:    20.0      # SHMS δ upper bound [%]
  paero_npe_min:  2.0      # SHMS aerogel NPE threshold
  phgc_npe_min:   1.0      # SHMS HGC NPE threshold
  ctime_real_center:          auto   # "auto" → use ctmean from CSV; or fixed ns value
  ctime_real_nsigma:          3.0    # half-window = nsigma × ctsigma (auto mode)
  ctime_real_window_fallback: 2.0    # half-window [ns] when ctsigma is NaN
  ctime_random_center:  39.2         # random sideband center [ns]
  ctime_random_window:   6.0         # random sideband half-width [ns]

histograms:
  - name:   phipq
    branch: P_kin_secondary_ph_xq  # direct ROOT tree branch
    bins:   16
    xmin:   -3.14159265
    xmax:    3.14159265
    xlabel: "#phi_{pq} (rad)"
    helicity_cut: positive          # optional: positive / negative / omit for all
```

---

## Branch Name Reference

| Constant | ROOT branch | Description |
|----------|-------------|-------------|
| `BRANCH_HSDELTA` | `H_gtr_dp` | HMS focal-plane δ [%] |
| `BRANCH_HCER_NPE` | `H_cer_npeSum` | HMS Cherenkov NPE sum |
| `BRANCH_HSSHSUM` | `H_cal_etottracknorm` | HMS calorimeter E/p |
| `BRANCH_PSDELTA` | `P_gtr_dp` | SHMS focal-plane δ [%] |
| `BRANCH_PAERO_NPE` | `P_aero_npeSum` | SHMS aerogel Cherenkov NPE |
| `BRANCH_PHGC_NPE` | `P_hgcer_npeSum` | SHMS heavy-gas Cherenkov NPE |
| `BRANCH_CTIME` | `CTime_ePiCoinTime_ROC2` | e−π coincidence time [ns] |
| `BRANCH_HELICITY` | `T_helicity_hel` | Raw helicity (+1 / −1) |
| `BRANCH_PPi` | `P_gtr_p` | SHMS pion momentum [GeV/c] |
| `BRANCH_NU` | `H_kin_primary_nu` | Virtual-photon energy ν [GeV] |
| `BRANCH_THETA_PQ` | `P_kin_secondary_th_xq` | θ_pq [rad] |
