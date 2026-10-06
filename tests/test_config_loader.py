"""
Tests for rsidis_ssa.config_loader

Positive tests use the dedicated YAML fixture at tests/fixtures/example_C_z05_thpq2.yaml.
Error-condition tests write minimal temporary YAML files via tmp_path.
"""

from pathlib import Path
import textwrap
import pytest

from rsidis_ssa.config_loader import (
    AnalysisConfig,
    CutsConfig,
    HistogramConfig,
    NormalizationConfig,
    RootfilesConfig,
    RunlistConfig,
    SettingConfig,
    load_config,
)
from rsidis_ssa.runlist import Setting

EXAMPLE_YAML = Path(__file__).parent / "fixtures" / "example_C_z05_thpq2.yaml"


# ---------------------------------------------------------------------------
# Helper: write a temp YAML and return its path
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write *content* as-is to a temp YAML file and return the path."""
    p = tmp_path / "test_cfg.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Minimal valid YAML — already dedented, safe to concatenate raw strings
# ---------------------------------------------------------------------------

MINIMAL_VALID = textwrap.dedent("""\
    setting:
      ebeam:    8.5831
      x:        0.25
      Q2:       3.3
      z:        0.5
      thpq:     2.0
      run_type: PI-SIDIS

    target: C
    run_period: period_1
    do_eplus_subtraction: true
    do_dummy_subtraction: false

    rootfiles:
      directory: /some/path
      pattern:   skimmed_coin_replay_production_{run}_-1.root
      treename:  T

    runlist:
      csv: data/rsidis_bigtable_pass0p1.csv

    cuts:
      hsdelta_lo:    -8.0
      hsdelta_hi:     8.0
      hcer_npe_min:   1.0
      hsshsum_min:    0.7
      psdelta_lo:   -10.0
      psdelta_hi:    20.0
      paero_npe_min:  2.0
      phgc_npe_min:   1.0
      psshsum_max:    0.8
      W_lo: 2.0
      W_hi: 100.0
      mmass_lo: 1.5
      mmass_hi: 100.0
      ctime_real_center:          auto
      ctime_real_nsigma:          3.0
      ctime_real_window_fallback: 2.0
      ctime_random_n_skip:         1
      ctime_random_n_peaks_lo:     3
      ctime_random_n_peaks_hi:     3

    histograms:
      - name:   hsdelta
        branch: H_gtr_dp
        bins:   16
        xmin:   -8.0
        xmax:    8.0
        xlabel: "HMS delta"
""")


# ===========================================================================
# load_config — file I/O
# ===========================================================================

class TestLoadConfig:

    def test_loads_example_yaml(self):
        cfg = load_config(EXAMPLE_YAML)
        assert isinstance(cfg, AnalysisConfig)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_non_mapping_yaml_raises_value_error(self, tmp_path):
        p = _write_yaml(tmp_path, "- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="did not parse to a YAML mapping"):
            load_config(p)

    def test_unknown_top_level_key_raises(self, tmp_path):
        bad = MINIMAL_VALID + "\nunknown_section: true\n"
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="unknown_section"):
            load_config(p)

    def test_validation_error_message_contains_file_path(self, tmp_path):
        bad = MINIMAL_VALID + "\nunknown_key: oops\n"
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError) as exc_info:
            load_config(p)
        assert str(p) in str(exc_info.value)

    def test_loads_minimal_valid_yaml(self, tmp_path):
        p = _write_yaml(tmp_path, MINIMAL_VALID)
        cfg = load_config(p)
        assert isinstance(cfg, AnalysisConfig)


# ===========================================================================
# SettingConfig
# ===========================================================================

class TestSettingConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(EXAMPLE_YAML)

    def test_ebeam_parsed(self, cfg):
        assert abs(cfg.setting.ebeam - 8.5831) < 1e-4

    def test_x_parsed(self, cfg):
        assert abs(cfg.setting.x - 0.25) < 1e-6

    def test_Q2_parsed(self, cfg):
        assert abs(cfg.setting.Q2 - 3.3) < 1e-6

    def test_z_parsed(self, cfg):
        assert abs(cfg.setting.z - 0.5) < 1e-6

    def test_thpq_parsed(self, cfg):
        assert abs(cfg.setting.thpq - 2.0) < 1e-6

    def test_run_type_parsed(self, cfg):
        assert cfg.setting.run_type == "PI-SIDIS"

    def test_to_setting_returns_setting_object(self, cfg):
        s = cfg.setting.to_setting()
        assert isinstance(s, Setting)
        assert s.run_type == "PI-SIDIS"
        assert abs(s.z - 0.5) < 1e-6

    def test_to_setting_matches_top_level_to_setting(self, cfg):
        assert cfg.to_setting() == cfg.setting.to_setting()

    def test_invalid_run_type_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace("run_type: PI-SIDIS", "run_type: ELASTIC")
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="run_type"):
            load_config(p)

    def test_unknown_setting_key_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            "run_type: PI-SIDIS", "run_type: PI-SIDIS\n  extra_key: 1"
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="extra_key"):
            load_config(p)


# ===========================================================================
# RootfilesConfig
# ===========================================================================

class TestRootfilesConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(EXAMPLE_YAML)

    def test_directory_parsed(self, cfg):
        assert cfg.rootfiles.directory == "/some/path"

    def test_pattern_parsed(self, cfg):
        assert "{run}" in cfg.rootfiles.pattern

    def test_treename_defaults_to_T(self, tmp_path):
        minimal = MINIMAL_VALID.replace(
            "treename:  T", ""
        )
        p = _write_yaml(tmp_path, minimal)
        cfg = load_config(p)
        assert cfg.rootfiles.treename == "T"

    def test_get_path_formats_run_number(self, cfg):
        path = cfg.rootfiles.get_path(24120)
        assert "24120" in str(path)
        assert str(path).endswith(".root")

    def test_get_path_uses_directory(self, cfg):
        path = cfg.rootfiles.get_path(99999)
        assert str(path).startswith(cfg.rootfiles.directory)

    def test_pattern_without_run_placeholder_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            "pattern:   skimmed_coin_replay_production_{run}_-1.root",
            "pattern:   coin_replay_production_FIXED.root",
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="\\{run\\}"):
            load_config(p)

    def test_unknown_rootfiles_key_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            "treename:  T", "treename:  T\n  extra: x"
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="extra"):
            load_config(p)


# ===========================================================================
# NormalizationConfig
# ===========================================================================

class TestNormalizationConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(EXAMPLE_YAML)

    def test_charge_column_default_bcm2(self, cfg):
        assert cfg.normalization.charge_column == "BCM2_Q"

    def test_charge_column_accepts_valid_bcms(self, tmp_path):
        for bcm in ("BCM1_Q", "BCM2_Q", "BCM4A_Q", "BCM4B_Q", "BCM4C_Q"):
            content = MINIMAL_VALID + f"\nnormalization:\n  charge_column: {bcm}\n"
            p = _write_yaml(tmp_path, content)
            cfg = load_config(p)
            assert cfg.normalization.charge_column == bcm

    def test_invalid_charge_column_raises(self, tmp_path):
        content = MINIMAL_VALID + "\nnormalization:\n  charge_column: UNKNOWN_BCM\n"
        p = _write_yaml(tmp_path, content)
        with pytest.raises(ValueError):
            load_config(p)

    def test_normalization_section_optional(self, tmp_path):
        """Omitting normalization entirely should use defaults."""
        p = _write_yaml(tmp_path, MINIMAL_VALID)  # MINIMAL_VALID has no normalization section
        cfg = load_config(p)
        assert cfg.normalization.charge_column == "BCM2_Q"


# ===========================================================================
# CutsConfig
# ===========================================================================

class TestCutsConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(EXAMPLE_YAML)

    def test_hsdelta_lo_hi(self, cfg):
        assert cfg.cuts.hsdelta_lo == -8.0
        assert cfg.cuts.hsdelta_hi ==  8.0

    def test_psdelta_lo_hi(self, cfg):
        assert cfg.cuts.psdelta_lo == -10.0
        assert cfg.cuts.psdelta_hi ==  20.0

    def test_ctime_real_center_auto(self, cfg):
        assert cfg.cuts.ctime_real_center == "auto"

    def test_ctime_real_center_auto_property(self, cfg):
        assert cfg.cuts.real_window_ns is None

    def test_ctime_real_center_float(self, tmp_path):
        content = MINIMAL_VALID.replace(
            "ctime_real_center:          auto", "ctime_real_center: 51.2"
        )
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert isinstance(cfg.cuts.ctime_real_center, float)
        assert abs(cfg.cuts.ctime_real_center - 51.2) < 1e-6
        assert cfg.cuts.real_window_ns == cfg.cuts.ctime_real_window_fallback

    def test_hsdelta_reversed_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace("hsdelta_lo:    -8.0", "hsdelta_lo:  9.0")
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="hsdelta_lo.*hsdelta_hi|hsdelta_hi.*hsdelta_lo"):
            load_config(p)

    def test_psdelta_reversed_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace("psdelta_lo:   -10.0", "psdelta_lo:  25.0")
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="psdelta_lo.*psdelta_hi|psdelta_hi.*psdelta_lo"):
            load_config(p)

    def test_negative_nsigma_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace("ctime_real_nsigma:          3.0",
                                    "ctime_real_nsigma: -1.0")
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError):
            load_config(p)

    def test_unknown_cut_key_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            "ctime_random_n_peaks_hi:     3",
            "ctime_random_n_peaks_hi:     3\n  mystery_cut: 99",
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="mystery_cut"):
            load_config(p)

    def test_all_pid_mins_positive(self, cfg):
        assert cfg.cuts.hcer_npe_min  > 0
        assert cfg.cuts.hsshsum_min   > 0
        assert cfg.cuts.paero_npe_min > 0
        assert cfg.cuts.phgc_npe_min  > 0


# ===========================================================================
# HistogramConfig
# ===========================================================================

class TestHistogramConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(EXAMPLE_YAML)

    def test_at_least_one_histogram(self, cfg):
        assert len(cfg.histograms) >= 1

    def test_histogram_names_are_unique(self, cfg):
        names = cfg.histogram_names()
        assert len(names) == len(set(names))

    def test_direct_branch_is_not_computed(self, cfg):
        h = cfg.get_histogram("hsdelta")
        assert not h.is_computed
        assert h.computed_name is None

    def test_computed_branch_is_computed(self, cfg):
        h = cfg.get_histogram("zhad")
        assert h.is_computed
        assert h.computed_name == "zhad"

    def test_computed_pt(self, cfg):
        h = cfg.get_histogram("Pt")
        assert h.is_computed
        assert h.computed_name == "Pt"

    def test_helicity_cut_none_by_default(self, cfg):
        h = cfg.get_histogram("phipq")
        assert h.helicity_cut is None

    def test_helicity_cut_positive(self, cfg):
        h = cfg.get_histogram("phipq_hplus")
        assert h.helicity_cut == "positive"

    def test_helicity_cut_negative(self, cfg):
        h = cfg.get_histogram("phipq_hminus")
        assert h.helicity_cut == "negative"

    def test_invalid_helicity_cut_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            'xlabel: "HMS delta"',
            'xlabel: "HMS delta"\n    helicity_cut: both',
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="helicity_cut"):
            load_config(p)

    def test_xmax_le_xmin_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace("xmax:    8.0", "xmax: -9.0")
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError):
            load_config(p)

    def test_bins_zero_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace("bins:   16", "bins: 0")
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError):
            load_config(p)

    def test_duplicate_histogram_names_raises(self, tmp_path):
        content = MINIMAL_VALID.replace(
            "histograms:\n"
            "  - name:   hsdelta\n"
            "    branch: H_gtr_dp\n"
            "    bins:   16\n"
            "    xmin:   -8.0\n"
            "    xmax:    8.0\n"
            '    xlabel: "HMS delta"\n',
            "histograms:\n"
            "  - name:   hsdelta\n"
            "    branch: H_gtr_dp\n"
            "    bins:   16\n"
            "    xmin:   -8.0\n"
            "    xmax:    8.0\n"
            "  - name:   hsdelta\n"
            "    branch: H_gtr_dp\n"
            "    bins:   16\n"
            "    xmin:   -8.0\n"
            "    xmax:    8.0\n",
        )
        p = _write_yaml(tmp_path, content)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            load_config(p)

    def test_get_histogram_by_name(self, cfg):
        h = cfg.get_histogram("W")
        assert h.name == "W"
        assert h.branch == "H_kin_primary_W"

    def test_get_histogram_unknown_name_raises(self, cfg):
        with pytest.raises(KeyError, match="nonexistent"):
            cfg.get_histogram("nonexistent")

    def test_unknown_histogram_key_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            'xlabel: "HMS delta"',
            'xlabel: "HMS delta"\n    mystery_field: true',
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="mystery_field"):
            load_config(p)


# ===========================================================================
# AnalysisConfig — cross-field validation and helpers
# ===========================================================================

class TestAnalysisConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(EXAMPLE_YAML)

    def test_target_parsed(self, cfg):
        assert cfg.target == "C"

    def test_do_eplus_subtraction_true(self, cfg):
        assert cfg.do_eplus_subtraction is True

    def test_do_dummy_subtraction_false_for_carbon(self, cfg):
        assert cfg.do_dummy_subtraction is False

    def test_apply_boil_corr_false_for_carbon(self, cfg):
        assert cfg.apply_boil_corr is False

    def test_apply_boil_corr_true_for_lh2(self, tmp_path):
        content = MINIMAL_VALID.replace("target: C", "target: LH2")
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.apply_boil_corr is True

    def test_dummy_subtraction_on_solid_target_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            "do_dummy_subtraction: false", "do_dummy_subtraction: true"
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="cryo"):
            load_config(p)

    def test_dummy_subtraction_allowed_for_lh2(self, tmp_path):
        content = (
            MINIMAL_VALID
            .replace("target: C", "target: LH2")
            .replace("do_dummy_subtraction: false", "do_dummy_subtraction: true")
        )
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.do_dummy_subtraction is True

    def test_resolve_runlist_path_relative(self, tmp_path):
        p = _write_yaml(tmp_path, MINIMAL_VALID)
        cfg = load_config(p)
        resolved = cfg.resolve_runlist_path(config_dir=p.parent)
        assert resolved.is_absolute()
        assert "rsidis_bigtable_pass0p1.csv" in str(resolved)

    def test_resolve_runlist_path_absolute(self, tmp_path):
        abs_csv = str(tmp_path / "absolute_runlist.csv")
        content = MINIMAL_VALID.replace(
            "csv: data/rsidis_bigtable_pass0p1.csv",
            f"csv: {abs_csv}",
        )
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert str(cfg.resolve_runlist_path()) == abs_csv

    def test_histogram_names_returns_list(self, cfg):
        names = cfg.histogram_names()
        assert isinstance(names, list)
        assert "hsdelta" in names

    def test_empty_histograms_list_raises(self, tmp_path):
        bad = MINIMAL_VALID.replace(
            "histograms:\n"
            "  - name:   hsdelta\n"
            "    branch: H_gtr_dp\n"
            "    bins:   16\n"
            "    xmin:   -8.0\n"
            "    xmax:    8.0\n"
            '    xlabel: "HMS delta"\n',
            "histograms: []\n",
        )
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError):
            load_config(p)
