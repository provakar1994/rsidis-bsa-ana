"""
Tests for rsidis_ssa.reader

``read_branches`` is tested with a mocked uproot to avoid needing real ROOT
files.  ``required_branches`` is tested with in-memory HistogramConfig objects.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rsidis_ssa.config_loader import CutsConfig, HistogramConfig
from rsidis_ssa.cuts import (
    BRANCH_CTIME,
    BRANCH_HCER_NPE,
    BRANCH_HELICITY,
    BRANCH_HSDELTA,
    BRANCH_HSSHSUM,
    BRANCH_NU,
    BRANCH_PAERO_NPE,
    BRANCH_PHGC_NPE,
    BRANCH_PPi,
    BRANCH_PSDELTA,
    BRANCH_THETA_PQ,
)
from rsidis_ssa.reader import (
    _PID_AND_CTIME_BRANCHES,
    read_branches,
    required_branches,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def cuts_cfg() -> CutsConfig:
    return CutsConfig(
        hsdelta_lo=-8.0,
        hsdelta_hi=8.0,
        hcer_npe_min=1.0,
        hsshsum_min=0.7,
        psdelta_lo=-10.0,
        psdelta_hi=20.0,
        paero_npe_min=2.0,
        phgc_npe_min=1.0,
        ctime_real_center="auto",
        ctime_real_nsigma=3.0,
        ctime_real_window_fallback=2.0,
        ctime_random_center=39.2,
        ctime_random_window=6.0,
    )


def _hcfg(
    name="h",
    branch="H_gtr_dp",
    bins=10,
    xmin=-5.0,
    xmax=5.0,
    helicity_cut=None,
) -> HistogramConfig:
    return HistogramConfig(
        name=name, branch=branch, bins=bins, xmin=xmin, xmax=xmax,
        helicity_cut=helicity_cut,
    )


# ===========================================================================
# required_branches
# ===========================================================================

class TestRequiredBranches:

    def test_always_includes_pid_and_ctime_branches(self, cuts_cfg):
        result = required_branches([], cuts_cfg)
        assert _PID_AND_CTIME_BRANCHES <= result

    def test_includes_direct_histogram_branch(self, cuts_cfg):
        cfgs = [_hcfg(branch="H_kin_primary_W")]
        result = required_branches(cfgs, cuts_cfg)
        assert "H_kin_primary_W" in result

    def test_includes_multiple_histogram_branches(self, cuts_cfg):
        cfgs = [
            _hcfg(name="a", branch="H_kin_primary_W"),
            _hcfg(name="b", branch="P_kin_secondary_ph_xq"),
        ]
        result = required_branches(cfgs, cuts_cfg)
        assert "H_kin_primary_W" in result
        assert "P_kin_secondary_ph_xq" in result

    def test_computed_zhad_adds_prereqs(self, cuts_cfg):
        cfgs = [_hcfg(branch="__computed__zhad", xmin=0.1, xmax=1.0)]
        result = required_branches(cfgs, cuts_cfg)
        assert BRANCH_PPi in result
        assert BRANCH_NU in result

    def test_computed_zhad_does_not_add_computed_branch_itself(self, cuts_cfg):
        cfgs = [_hcfg(branch="__computed__zhad", xmin=0.1, xmax=1.0)]
        result = required_branches(cfgs, cuts_cfg)
        assert "__computed__zhad" not in result

    def test_computed_Pt_adds_prereqs(self, cuts_cfg):
        cfgs = [_hcfg(branch="__computed__Pt", xmin=0.0, xmax=0.5)]
        result = required_branches(cfgs, cuts_cfg)
        assert BRANCH_PPi in result
        assert BRANCH_THETA_PQ in result

    def test_no_helicity_branch_without_helicity_cut(self, cuts_cfg):
        cfgs = [_hcfg(branch="H_gtr_dp")]
        result = required_branches(cfgs, cuts_cfg)
        assert BRANCH_HELICITY not in result

    def test_helicity_branch_when_helicity_cut_present(self, cuts_cfg):
        cfgs = [_hcfg(branch="P_kin_secondary_ph_xq",
                       helicity_cut="positive",
                       xmin=-3.14, xmax=3.14)]
        result = required_branches(cfgs, cuts_cfg)
        assert BRANCH_HELICITY in result

    def test_helicity_branch_added_even_if_only_one_histogram_needs_it(self, cuts_cfg):
        cfgs = [
            _hcfg(name="a", branch="H_gtr_dp"),
            _hcfg(name="b", branch="P_kin_secondary_ph_xq",
                  helicity_cut="negative", xmin=-3.14, xmax=3.14),
        ]
        result = required_branches(cfgs, cuts_cfg)
        assert BRANCH_HELICITY in result

    def test_unknown_computed_raises(self, cuts_cfg):
        cfgs = [_hcfg(branch="__computed__mystery", xmin=0.0, xmax=1.0)]
        with pytest.raises(ValueError, match="Unknown computed quantity"):
            required_branches(cfgs, cuts_cfg)

    def test_returns_set(self, cuts_cfg):
        result = required_branches([], cuts_cfg)
        assert isinstance(result, set)

    def test_result_contains_no_duplicates(self, cuts_cfg):
        # set by definition has no duplicates; check it has sensible size
        cfgs = [
            _hcfg(name="a", branch="__computed__zhad", xmin=0.1, xmax=1.0),
            _hcfg(name="b", branch="__computed__Pt",   xmin=0.0, xmax=0.5),
        ]
        result = required_branches(cfgs, cuts_cfg)
        # BRANCH_PPi should appear only once even though both computeds need it
        assert BRANCH_PPi in result
        assert len(result) == len(set(result))  # trivially true for a set


# ===========================================================================
# read_branches — mocked uproot
# ===========================================================================

def _make_mock_tree(branches: dict[str, np.ndarray]) -> MagicMock:
    """Build a mock uproot tree that returns *branches* via tree.arrays()."""
    tree = MagicMock()
    tree.keys.return_value = list(branches.keys())
    tree.arrays.return_value = branches
    return tree


def _make_mock_file(treename: str, tree_mock: MagicMock) -> MagicMock:
    """
    Build a mock uproot context manager.

    ``with uproot.open(path) as f:`` calls ``__enter__`` to get ``f``.
    We make ``__enter__`` return the mock itself so that ``f[treename]``
    and ``f.keys()`` work as configured.
    """
    mock_file = MagicMock()
    # Context manager protocol: __enter__ returns the file object itself
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_file.__getitem__ = MagicMock(return_value=tree_mock)
    mock_file.keys = MagicMock(return_value=[treename])
    return mock_file


class TestReadBranches:

    def _fake_root(self, tmp_path: Path) -> Path:
        """Create an empty placeholder file so the existence check passes."""
        p = tmp_path / "fake.root"
        p.write_bytes(b"")
        return p

    def test_returns_dict_of_arrays(self, tmp_path):
        root_path = self._fake_root(tmp_path)
        data = {
            "H_gtr_dp": np.array([1.0, 2.0, 3.0]),
            "P_gtr_dp": np.array([4.0, 5.0, 6.0]),
        }
        tree_mock = _make_mock_tree(data)
        mock_file = _make_mock_file("T", tree_mock)

        with patch("uproot.open", return_value=mock_file):
            result = read_branches(root_path, "T", {"H_gtr_dp", "P_gtr_dp"})

        assert set(result.keys()) == {"H_gtr_dp", "P_gtr_dp"}
        np.testing.assert_array_equal(result["H_gtr_dp"], data["H_gtr_dp"])

    def test_values_are_numpy_arrays(self, tmp_path):
        root_path = self._fake_root(tmp_path)
        data = {"H_gtr_dp": np.array([1.0, 2.0])}
        tree_mock = _make_mock_tree(data)
        mock_file = _make_mock_file("T", tree_mock)

        with patch("uproot.open", return_value=mock_file):
            result = read_branches(root_path, "T", {"H_gtr_dp"})

        assert isinstance(result["H_gtr_dp"], np.ndarray)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ROOT file not found"):
            read_branches(tmp_path / "nonexistent.root", "T", {"H_gtr_dp"})

    def test_missing_tree_raises_key_error(self, tmp_path):
        root_path = self._fake_root(tmp_path)
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.__getitem__ = MagicMock(side_effect=KeyError("T"))
        mock_file.keys = MagicMock(return_value=["SomeOtherTree"])

        with patch("uproot.open", return_value=mock_file):
            with pytest.raises(KeyError, match="not found"):
                read_branches(root_path, "T", {"H_gtr_dp"})

    def test_missing_branch_raises_key_error(self, tmp_path):
        root_path = self._fake_root(tmp_path)
        data = {"H_gtr_dp": np.array([1.0])}
        tree_mock = _make_mock_tree(data)
        mock_file = _make_mock_file("T", tree_mock)

        with patch("uproot.open", return_value=mock_file):
            with pytest.raises(KeyError, match="missing"):
                read_branches(root_path, "T", {"H_gtr_dp", "NONEXISTENT_branch"})

    def test_context_manager_used(self, tmp_path):
        """Verify uproot.open is called with the correct path."""
        root_path = self._fake_root(tmp_path)
        data = {"H_gtr_dp": np.array([1.0])}
        tree_mock = _make_mock_tree(data)
        mock_file = _make_mock_file("T", tree_mock)

        with patch("uproot.open", return_value=mock_file) as mock_open:
            read_branches(root_path, "T", {"H_gtr_dp"})

        mock_open.assert_called_once_with(root_path)
