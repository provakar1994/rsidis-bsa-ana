# RSIDIS SSA Analysis Package

Single-spin asymmetry analysis for the Hall C RSIDIS experiment (HMS + SHMS).  
One YAML config file drives one complete analysis (one kinematic setting, one target).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Automation](#pipeline-automation)
3. [Visualization Scripts](#visualization-scripts)
4. [Package Layout](#package-layout)
5. [Module Dependency Tree](#module-dependency-tree)
6. [Workflow: Step by Step](#workflow-step-by-step)
   - [Step 1 — Configuration](#step-1--configuration)
   - [Step 2 — Run Selection](#step-2--run-selection)
   - [Step 3 — Per-Run Normalization Weights](#step-3--per-run-normalization-weights)
   - [Step 4 — ROOT File Reading](#step-4--root-file-reading)
   - [Step 5 — Event Selection Cuts](#step-5--event-selection-cuts)
   - [Step 6 — Histogram Filling](#step-6--histogram-filling)
   - [Step 7 — Run Combination](#step-7--run-combination)
   - [Step 8 — Background Subtraction](#step-8--background-subtraction)
   - [Step 9 — Asymmetry Calculation](#step-9--asymmetry-calculation)
7. [Diagnostic PDF Pages](#diagnostic-pdf-pages)
8. [Configuration Reference](#configuration-reference)
9. [Run Constants Reference](#run-constants-reference)
10. [Branch Name Reference](#branch-name-reference)

---

## Quick Start

Single setting, manually:

```bash
python analysis.py config/C_pip_e10p7_x0p25_q23p3_z0p5_thpq2p0.yaml
```

Add `--verbose` / `-v` to see per-run detail tables and the background subtraction
breakdown (suppressed by default):

```bash
python analysis.py --verbose config/C_pip_e10p7_x0p25_q23p3_z0p5_thpq2p0.yaml
```

Full automated pipeline for all run-period-1 settings:

```bash
python run_pipeline.py \
    --settings data/settings/rpr1_pip_settings.csv \
               data/settings/rpr1_pim_settings.csv
```

Each `analysis.py` run produces:
- A multi-page diagnostic PDF (`output/<stem>/<stem>.pdf`)
- An unbinned asymmetry CSV (`output/<stem>/<stem>.csv`)
- A binned (p_T slice) asymmetry CSV (`output/<stem>/<stem>_binned.csv`)
- A kinematic summary CSV (`output/<stem>/<stem>_summary.csv`)
- A ROOT file with all subtracted histograms (`output/<stem>/<stem>.root`)

---

## Pipeline Automation

`run_pipeline.py` orchestrates the full analysis across all kinematic settings defined
in one or more settings CSVs. It has three sequential steps:

| Step | What it does |
|------|-------------|
| `generate` | Creates `config/<stem>_base.yaml` + `config/<stem>_thpq<val>.yaml` for every family |
| `analyze` | Runs `analysis.py` for each per-thpq config; skips if output CSV already exists |
| `combine` | Runs `combine_asymmetry.py` across all thpq values per family (skips single-thpq families) |

All steps are idempotent: existing outputs are skipped unless `--force` is given.

### Common invocations

```bash
# Full pipeline, all settings
python run_pipeline.py \
    --settings data/settings/rpr1_pip_settings.csv \
               data/settings/rpr1_pim_settings.csv

# Preview every action without executing anything
python run_pipeline.py --settings ... --dry-run

# Generate configs only (no analysis yet)
python run_pipeline.py --settings ... --steps generate

# One target, one z value, all steps
python run_pipeline.py --settings ... --target LH2 --z 0.5

# Parallel exclusive-event process for z = 0.9.
# This keeps the nominal W cut and uses an exclusive missing-mass window.
python run_pipeline.py --settings ... --z 0.9 --process exclusive --binned

# One target, one thpq — run analysis only, then combine the whole family
python run_pipeline.py --settings ... --target C --z 0.5 --thpq -0.8 --steps analyze
python run_pipeline.py --settings ... --target C --z 0.5 --steps combine

# Re-run everything even if outputs exist
python run_pipeline.py --settings ... --force

# Show per-run detail in analysis output
python run_pipeline.py --settings ... --verbose
```

### Settings CSV format

Each row is one kinematic setting (one target + one thpq). The pipeline groups rows
by `(target, ebeam, x, Q2, z, run_type)` to form *families* and generates one base
config per family.

```
target,ebeam,x,Q2,z,thpq,run_type
C,10.6716,0.25,3.3,0.5,-0.8,PI+SIDIS
C,10.6716,0.25,3.3,0.5,2.0,PI+SIDIS
C,10.6716,0.25,3.3,0.5,5.2,PI+SIDIS
```

### YAML config inheritance

Generated configs use a two-file base/override pattern to avoid duplication.
The *base* file (`*_base.yaml`) holds all shared parameters; each per-thpq *override*
file is four lines:

```yaml
# thpq = -0.8
_base: C_pip_e10p7_x0p25_q23p3_z0p5_base.yaml

setting:
  thpq: -0.8
```

`config_loader.load_config()` merges the two files before validation: nested dicts
are merged key-by-key; scalars and lists in the override replace the base value.
Base files must not themselves reference another `_base` (one level only).

### Parallel analysis processes

Use `--process <name>` to run a second analysis stream for the same kinematic
settings without overwriting the nominal SIDIS configs or outputs. The default
process is `sidis`, which preserves the historical filenames. Any other process
name is inserted between the z tag and `thpq`/`base`:

```text
config/C_pip_e10p7_x0p25_q23p3_z0p9_base.yaml
config/C_pip_e10p7_x0p25_q23p3_z0p9_thpq2p0.yaml

config/C_pip_e10p7_x0p25_q23p3_z0p9_exclusive_base.yaml
config/C_pip_e10p7_x0p25_q23p3_z0p9_exclusive_thpq2p0.yaml
```

The built-in `exclusive` process is intended for low-missing-mass exclusive
events, especially at `z = 0.9`. Generated exclusive configs keep the nominal
wide W cut, `2.0 <= W <= 100` GeV, and set the missing-mass cut to
`0.85 <= M_miss <= 1.05` GeV.

Example:

```bash
python run_pipeline.py \
    --settings data/settings/rpr1_pip_settings.csv \
               data/settings/rpr1_pim_settings.csv \
    --z 0.9 \
    --process exclusive \
    --steps generate analyze combine \
    --binned
```

This writes independent outputs such as
`output/C_pip_e10p7_x0p25_q23p3_z0p9_exclusive_thpq2p0/` and combined files
under `output/combined/` with the same `_exclusive_` process tag.

### Combining asymmetries across thpq values

`combine_asymmetry.py` has two run modes:

**Direct mode** (list CSV files explicitly):
```bash
python combine_asymmetry.py output/C_pip_.../C_pip_..._thpqm0p8.csv \
                            output/C_pip_.../C_pip_..._thpq2p0.csv \
                            --output output/combined/C_pip_combined.csv
```

**Config mode** (derive paths from the base config):
```bash
python combine_asymmetry.py \
    --base_config config/C_pip_e10p7_x0p25_q23p3_z0p5_base.yaml \
    --thpq -0.8 2.0 5.2 \
    --is_binned          # omit for unbinned
```
The output path defaults to
`output/combined/<stem>_thpq<v1>AND<v2>...(_binned).csv`.

Process-tagged bases work the same way:

```bash
python combine_asymmetry.py \
    --base_config config/C_pip_e10p7_x0p25_q23p3_z0p9_exclusive_base.yaml \
    --thpq -0.8 2.0 \
    --is_binned
```

Summary CSVs include a `process` column (`sidis` for nominal files,
`exclusive` for the exclusive stream).

Single-setting `<stem>_summary.csv` files include overall rows
(`variable=overall`) and one row per configured kinematic bin (`variable=pt`
for pT). `hmin`, `hmax`, and `bin_center` identify the bin in GeV/c for pT.
Both these files and combined `*_binned_summary.csv` files expose `A_phys`
and `A_phys_error`: the fitted sine amplitude and its statistical uncertainty.
They are aliases of `asym` and `asym_err`, retained for plotting compatibility.
These fit amplitudes differ from the individual φ-bin `A_phys` measurements
in `<stem>_binned.csv`.

Asymmetry CSVs also include diagnostic `N_plus` and `N_minus` counts for
effective helicity. These are signal-sample event counts after scaled random
subtraction, before charge normalization, e⁺ subtraction, or dummy subtraction.
They can be fractional or negative. The asymmetry and its uncertainty still
come from the weighted, background-subtracted histograms.

Combined counts sum only rows with a finite asymmetry and a finite, positive
uncertainty. Missing count values remain unknown (`NaN`), including in summary
totals. When any input CSV lacks count columns, combined bin CSVs omit those
columns and summary counts are unknown. Single-setting summary counts include
all φ bins, so their totals need not equal the contributing-bin combined totals.

Generated `output/` and `backup_output_*/` directories are ignored by Git;
keep any desired result snapshots separately. The local Phase II ROOT-file link
and independent `simulation/simc_gfortran/` checkout are also ignored. See
`simulation/README.md` for the simulation source location.

---

## Visualization Scripts

These scripts operate on the combined output CSVs in `output/combined/` and produce
publication-style PDFs in `output/plots/`.

### `plot_z_dependence.py`

Plots A_LU^sinφ as a function of z for a single target and hadron species.

```bash
python plot_z_dependence.py --target C --particle pip
python plot_z_dependence.py --target LH2 --particle pim
python plot_z_dependence.py --target C --particle pip --output my_plot.pdf
```

Produces a two-page PDF:
- **Page 1** — A_LU^sinφ vs p_T (one curve per z) and vs z (one curve per p_T bin)
- **Page 2** — A_LU^sinφ vs z, one panel per p_T bin, shared y-axis, no gap between panels

### `plot_nuclear_dependence.py`

Compares A_LU^sinφ (and optionally F_LU^sinφ/F_UU) across targets for all z values.
By default π⁺ (blue) and π⁻ (red) appear with small horizontal offsets and a black diamond
showing their inverse-variance weighted average. Individual charge points and
error bars use 45% opacity. Z-averaged page titles list the supplied real z
coordinates (falling back to nominal z for unmapped settings). This applies to comparison,
difference, structure-function, and z-averaged difference pages. Use
`--particle pi+` or `--particle pi-` to plot a single charge without offsets or an
average; `--particle both` is the default.

For paired values, the average is `sum(A / sigma²) / sum(1 / sigma²)` with error
`1 / sqrt(sum(1 / sigma²))`, assuming independent particle uncertainties.
Difference pages average the two particle differences with their propagated errors.
An average is drawn only when both charges have finite values and positive finite
errors at the same nominal z and p_T bin.

`--z` continues to select **nominal** settings. Supply acceptance-averaged vertex
z coordinates with `--real-z NOMINAL=REAL ...`, for example:

```bash
python plot_nuclear_dependence.py --x 0.25 --z 0.5 0.67 \
    --real-z 0.5=0.513 0.67=0.681 --particle both
```

For target-, charge-, or bin-dependent coordinates, use `--real-z-csv real_z.csv`.
Required columns are `z` (nominal) and `real_z`; optional matching columns are
`x`, `target`, `particle`, and `histogram`. Every provided key must be filled;
rows must not overlap for selected data. For example:

```csv
x,target,particle,z,real_z
0.25,C,pi+,0.5,0.513
0.25,C,pi-,0.5,0.519
0.25,LH2,pi+,0.5,0.510
0.25,LH2,pi-,0.5,0.516
```

Unmapped settings use nominal z. CSV matches override `--real-z` mappings.
Charge averages lie at the midpoint of the two supplied coordinates; offsets
are visual only and do not affect any average. Difference points use the
non-reference target's coordinates. The z-averaged pages retain their p_T axis;
their averaging and reference matching always use nominal settings.

```bash
# Default: A_LU only, all targets
python plot_nuclear_dependence.py

# With F_LU/F_UU pages (requires ε = virtual-photon depolarisation)
python plot_nuclear_dependence.py --epsilon 0.7

# Custom y ranges and target subset
python plot_nuclear_dependence.py --epsilon 0.7 \
    --ylim -0.02 0.12  --ylim-diff -0.06 0.06 \
    --ylim-flu -0.05 0.20  --ylim-flu-diff -0.10 0.10 \
    --targets C Cu LD2 LH2 --reference LH2
```

Page layout:
- **Page 1** — A_LU^sinφ, all targets vs z (π⁺ and π⁻ overlaid)
- **Page 2** — A_LU^sinφ difference (A_target − A_ref)
- **Page 3** — F_LU^sinφ/F_UU vs z (`--epsilon` only)
- **Page 4** — F_LU^sinφ/F_UU difference (`--epsilon` only)
- **Page 5** — A_LU^sinφ, individual targets vs p_T, averaged over selected z
- **Page 6** — A_LU^sinφ difference vs p_T, averaged over selected z
- **Page 7** — F_LU^sinφ/F_UU, individual targets vs p_T, averaged over selected z
- **Page 8** — F_LU^sinφ/F_UU difference vs p_T, averaged over selected z

Pages 5–8 require `--z` (alias `--z-avg`); pages 7–8 also require
`--epsilon`. Page numbers assume all eight pages are enabled. Comparison pages
include the reference target and are produced even when no reference difference
is available. Set their y ranges with `--ylim-zavg` and `--ylim-flu-zavg`,
which default to `--ylim` and `--ylim-flu`, respectively.

where `F_LU^sinφ / F_UU = A_LU^sinφ / sqrt(2 ε (1−ε))`.


Systematic bands are optional. Enable them with `--sys --sys-csv FILE`.
The CSV specifies percentages **directly for final plotted values**, including
explicit weighted-average, target-difference, and z-averaged entries:

```csv
process,x,target,reference,particle,view,z,histogram,beam_pol_pct,excl_pct,delta_pct,rho_pct
sidis,0.25,C,,both,z,0.5,phipq_ptbin0,3.0,2.0,1.5,1.0
sidis,0.25,C,LH2,both,z,0.5,phipq_ptbin0,0.0,2.5,1.8,1.2
sidis,0.25,C,LH2,both,zavg,,phipq_ptbin0,0.0,2.0,1.5,1.0
```

- `reference` is empty for an individual target, or names the subtracted target.
- `particle` is `pi+`, `pi-`, or `both`. When both charges are plotted, the band
  represents the black weighted average and uses only `both` entries.
- `view=z` matches nominal `z`; `view=zavg` requires empty `z` and describes the
  final value averaged over the z settings selected for that run. Use a separate
  CSV when different z selections require different z-averaged percentages.
- `histogram` must exactly match the summary CSV's p_T-bin identifier.
- All four source columns are required. Percentages must be finite and
  nonnegative; use `0` for an absent source. Additional `*_pct` columns are allowed.

For the final value `V`, the systematic uncertainty is
`abs(V) * sqrt(sum(source_pct**2)) / 100`. Thus differences and averages use their
own percentages, with no propagation or fallback from constituent points.
Zero-valued points have zero percentage-based systematic uncertainty. The same
percentages apply to A_LU and F_LU/F_UU; their absolute bands follow the observable's
scaling. Statistical error bars and weights are unchanged.

A gray band shows +1σ above a flat baseline, 4% of the y-axis span above the
bottom by default. Its upper edge uses shape-preserving PCHIP interpolation
between real z coordinates or p_T centers, without extrapolation. A single-point
panel uses a short constant strip. `--sys-y Y` optionally sets an absolute baseline
for all pages; it must fit each page's y range. Bands retain the uncertainty's
actual height in observable units.

The example `config/systematics_example.csv` contains **illustrative percentages,
not uncertainty estimates**, for SIDIS, x=0.25, targets LH2/LD2/C/Cu, nominal z=0.5/0.67,
and all three particle selections. Its z-averaged rows are for that exact pair
of nominal z settings. Test all eight pages with:

```bash
python plot_nuclear_dependence.py --x 0.25 --z 0.5 0.67 \
    --targets LD2 C Cu LH2 --epsilon 0.59 \
    --real-z 0.5=0.513 0.67=0.681 \
    --sys --sys-csv config/systematics_example.csv \
    --output output/plots/nuclear_dependence_systematics_example.pdf
```

When enabled, a missing CSV, invalid or duplicate identifiers, or missing entries
for any plotted band point stops the command. Missing-entry checks run before
opening the output PDF. Without `--sys`, bands are disabled; `--sys-csv` and
`--sys-y` require `--sys`.

---

## Package Layout

```
ssa/
├── analysis.py                  # Single-setting driver — generates diagnostic PDF + output CSV
├── combine_asymmetry.py         # Combine asymmetries across thpq values (IVW)
├── run_pipeline.py              # Automated pipeline: generate configs → analyze → combine
├── plot_z_dependence.py         # z-dependence plots for one target/hadron
├── plot_nuclear_dependence.py    # Target-dependence comparison across all targets
├── config/
│   ├── <target>_<pip|pim>_<kin>_base.yaml     # Base config shared across thpq values
│   ├── <target>_<pip|pim>_<kin>_thpq<v>.yaml  # Per-thpq override (4 lines; _base: ...)
│   ├── <target>_<pip|pim>_<kin>_<process>_base.yaml     # Parallel process base
│   ├── <target>_<pip|pim>_<kin>_<process>_thpq<v>.yaml  # Parallel process override
│   └── run_constants.yaml       # Run-period constants (dummy scales, beam_bunch_ns, …)
├── data/
│   ├── rsidis_bigtable_pass0p1.csv   # Master runlist (includes BCM2_Q_hp/BCM2_Q_hm columns)
│   ├── rootfiles_pass0p1/            # Symlinked or local ROOT skim files
│   └── settings/
│       ├── rpr1_pip_settings.csv     # All π⁺ settings for run period 1
│       └── rpr1_pim_settings.csv     # All π⁻ settings for run period 1
├── output/
│   ├── <stem>/                       # Per-setting output directory
│   │   ├── <stem>.csv                # Unbinned asymmetry table
│   │   ├── <stem>_binned.csv         # Binned (pt slice) asymmetry table
│   │   ├── <stem>_summary.csv        # Kinematic summary
│   │   ├── <stem>.root               # All subtracted histograms
│   │   └── <stem>.pdf                # Multi-page diagnostic plots
│   └── combined/
│       └── <stem>_thpq<v1>AND<v2>.csv   # Combined asymmetry across thpq values
└── rsidis_ssa/
    ├── config_loader.py         # YAML → validated Pydantic models; handles _base: inheritance
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
| `normalization` | Luminosity normalization scheme and helicity-gated charge options |
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

#### Helicity-gated charge normalization

When `normalization.use_helicity_gated_charge: true`, the master runlist must contain
two additional columns (names configurable via `charge_hp_column` / `charge_hm_column`,
default `BCM2_Q_hp` / `BCM2_Q_hm`) giving the beam charge delivered during helicity-plus
and helicity-minus gate windows, in μC.

After all runs are filled, helicity-split histograms (`_hplus` / `_hminus` suffixes)
are divided by the summed per-helicity charge (Q_hp_tot / Q_hm_tot) instead of the
total Q_tot.  Inclusive histograms (no helicity suffix) continue to use Q_tot.
This corrects for any asymmetry in the time spent in each helicity state.

When the flag is `false` (default), Q_hp_tot = Q_hm_tot = Q_tot / 2, which is
equivalent to assuming equal helicity-state charges (ratio = 1).

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

When `phgc_p_threshold` is set, the HGC NPE cut is applied only to events where
`P_gtr_p >= phgc_p_threshold`; below that momentum threshold the HGC cut is skipped.

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
| Last | **Run Summary** — four labelled sections rendered as tables: (A) config file path + generation timestamp; (B) kinematic setting, normalization scheme, and per-run-type charge totals with exclusion counts; (C) all PID/acceptance cuts and coincidence-time window parameters; (D) background subtraction table with one row per histogram and paired sub% / \|pull\| columns per active step (random/dummy/eplus), pull cells > 5 highlighted in red. |

---

## Configuration Reference

### Base/override inheritance

To avoid repeating identical blocks across thpq values, configs use a two-file
pattern.  A *base* file holds all shared parameters; a lightweight *override* file
specifies only `_base:` and the differing values:

```yaml
# Per-thpq override — the entire file
_base: C_pip_e10p7_x0p25_q23p3_z0p5_base.yaml

setting:
  thpq: -0.8
```

`config_loader.load_config()` loads the base, deep-merges the override (nested dicts
merge key-by-key; scalars/lists are replaced), then validates the result.  Chained
bases (`_base` inside a base file) are not supported.

### Full config reference

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
  charge_column: BCM2_Q             # BCM1_Q / BCM2_Q / BCM4A_Q / BCM4B_Q / BCM4C_Q
  weight_scheme: charge_only        # eff_corrected_counts | eff_corrected_charge | charge_only
  use_helicity_gated_charge: true   # divide _hplus/_hminus histos by Q_hp/Q_hm instead of Q_tot
  charge_hp_column: BCM2_Q_hp       # runlist column for helicity-plus charge [μC]
  charge_hm_column: BCM2_Q_hm       # runlist column for helicity-minus charge [μC]

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
  phgc_p_threshold: null   # GeV/c; null = unconditional HGC cut; float = apply only above this momentum
  psshsum_max:    0.8      # SHMS calorimeter E/p maximum (pion rejection)

  # Kinematic cuts
  W_lo:       2.0          # invariant mass W lower bound [GeV]
  W_hi:       100          # invariant mass W upper bound [GeV]
  mmass_lo:   1.5          # missing-mass lower bound [GeV]
  mmass_hi:   100          # missing-mass upper bound [GeV]

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


## Example executions
- python run_pipeline.py --settings data/settings/rpr1_pim_settings.csv data/settings/rpr1_pip_settings.csv --z 0.36 0.5 0.67 --steps analyze combine --binned --force
- python plot_nuclear_dependence.py --targets LD2 C Cu LH2 --epsilon 0.59 --reference LH2 --z 0.36 0.5 0.67 --ylim -0.02 0.12  --ylim-diff -0.06 0.06 --ylim-flu -0.05 0.20  --ylim-flu-diff -0.06 0.11 --output output/plots/x0p25_target_dependence_pass1.pdf
- python run_pipeline.py --settings data/settings/rpr1_pim_settings.csv data/settings/rpr1_pip_settings.csv --z 0.9 --steps analyze combine --process exclusive --binned --force
- python plot_nuclear_dependence.py --targets LD2 C Cu LH2 --epsilon 0.59 --reference LH2 --z-avg 0.9 --ylim-zavg-diff -0.19 0.19 --ylim -0.072 0.2  --ylim-diff -0.09 0.12 --ylim-flu -0.15 0.37  --ylim-flu-diff -0.12 0.18 --process exclusive --output output/plots/x0p25_target_dependence_exclusive_pass1.pdf
- python run_pipeline.py --settings data/settings/rpr1_pip_settings.csv --z 0.52 --steps analyze combine --binned --force
- python plot_nuclear_dependence.py --targets LD2 C Cu LH2 --epsilon 0.77 --reference LH2 --z 0.52 --ylim -0.02 0.12  --ylim-diff -0.06 0.06 --ylim-flu -0.05 0.20  --ylim-flu-diff -0.06 0.11 --output output/plots/x0p44_target_dependence_pass1.pdf

### Target-dependence input selection

`plot_nuclear_dependence.py` reads the already combined
`output/combined/*_binned_summary.csv` files. Use `--x` to select one nominal
Bjorken x, then `--z` (alias `--z-avg`) to select the z settings for every page:

```bash
python plot_nuclear_dependence.py --x 0.25 --z 0.5 0.67 --epsilon 0.59
```

The selected z values also define the additional z-averaged difference page.
Without `--z`, all z settings at the selected x are plotted, with no z-average
page. Omitting `--x` is allowed only when the inputs contain a single x value.
Unknown x or z selections produce an error. Nominal settings match within
1e-6, so z=0.50 and z=0.52 remain distinct. No thpq selection is applied.
