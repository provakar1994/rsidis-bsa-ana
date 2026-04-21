"""
ROOT file reading for one analysis run.

``required_branches(histo_cfgs, cuts_cfg)``
    Derive the minimal set of ROOT-tree branch names needed by the cuts
    and histogram definitions in the YAML config.

``read_branches(root_path, treename, branch_names)``
    Open a ROOT file with uproot and return a plain dict of numpy arrays,
    one per requested branch.

Error handling
--------------
* ``FileNotFoundError`` — ROOT file does not exist.
  The pipeline catches this to exclude the run and continue.
* ``KeyError`` — tree or branch not found in the file.
  This signals a misconfigured YAML or wrong file; it propagates to the
  caller as a fatal error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rsidis_ssa.config_loader import CutsConfig, HistogramConfig
from rsidis_ssa.cuts import (
    BRANCH_CTIME,
    BRANCH_HCER_NPE,
    BRANCH_HELICITY,
    BRANCH_HSDELTA,
    BRANCH_HETOTTRACKNORM,
    BRANCH_NU,
    BRANCH_PAERO_NPE,
    BRANCH_PHGC_NPE,
    BRANCH_PPi,
    BRANCH_PSDELTA,
    BRANCH_PETOTTRACKNORM,
    BRANCH_THETA_PQ,
)


# ---------------------------------------------------------------------------
# Branch-set builder
# ---------------------------------------------------------------------------

#: Branches always required for the PID cuts and coincidence-time cut.
_PID_AND_CTIME_BRANCHES: frozenset[str] = frozenset({
    BRANCH_HSDELTA,
    BRANCH_HCER_NPE,
    BRANCH_HETOTTRACKNORM,
    BRANCH_PSDELTA,
    BRANCH_PAERO_NPE,
    BRANCH_PETOTTRACKNORM,
    BRANCH_PHGC_NPE,
    BRANCH_CTIME,
})

#: Branches required as inputs to each computed quantity.
_COMPUTED_PREREQS: dict[str, frozenset[str]] = {
    "zhad": frozenset({BRANCH_PPi, BRANCH_NU}),
    "Pt":   frozenset({BRANCH_PPi, BRANCH_THETA_PQ}),
}


def required_branches(
    histo_cfgs: list[HistogramConfig],
    cuts_cfg: CutsConfig,  # accepted for future use; currently unused
) -> set[str]:
    """
    Return the minimal set of ROOT-tree branch names needed for this config.

    Includes:
    - All PID + coincidence-time cut branches (always)
    - The branch for each direct histogram
    - The prerequisite branches for each computed histogram
    - The per-event helicity branch (when any histogram has a helicity_cut)

    Parameters
    ----------
    histo_cfgs : list[HistogramConfig]
        Histogram definitions from the YAML config.
    cuts_cfg : CutsConfig
        Cut thresholds from the YAML config (reserved for future branch
        extensions such as a configurable ctime branch name).

    Returns
    -------
    set[str]
    """
    branches: set[str] = set(_PID_AND_CTIME_BRANCHES)

    for hcfg in histo_cfgs:
        if not hcfg.is_computed:
            branches.add(hcfg.branch)
        else:
            cname = hcfg.computed_name
            if cname in _COMPUTED_PREREQS:
                branches |= _COMPUTED_PREREQS[cname]
            else:
                raise ValueError(
                    f"Unknown computed quantity {cname!r} in histogram "
                    f"'{hcfg.name}'.  Register its prerequisites in "
                    "reader._COMPUTED_PREREQS."
                )

    if any(h.helicity_cut is not None for h in histo_cfgs):
        branches.add(BRANCH_HELICITY)

    return branches


# ---------------------------------------------------------------------------
# ROOT reader
# ---------------------------------------------------------------------------

def read_branches(
    root_path: str | Path,
    treename: str,
    branch_names: set[str] | list[str],
) -> dict[str, np.ndarray]:
    """
    Read branches from a ROOT TTree into a plain dict of numpy arrays.

    Parameters
    ----------
    root_path : path-like
        Full path to the ROOT file.
    treename : str
        Name of the TTree inside the file (e.g. 'T').
    branch_names : set or list of str
        Branches to read.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are branch names; values are 1-D numpy arrays of equal length.

    Raises
    ------
    FileNotFoundError
        If *root_path* does not exist.
    KeyError
        If *treename* is not found in the file, or if any requested branch
        is absent from the tree.
    """
    import uproot  # deferred import so the module is importable without uproot

    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    branch_list = sorted(branch_names)  # deterministic order for reproducibility

    with uproot.open(root_path) as f:
        try:
            tree = f[treename]
        except KeyError:
            raise KeyError(
                f"Tree '{treename}' not found in {root_path}.  "
                f"Available keys: {list(f.keys())}"
            )

        available = set(tree.keys())
        missing = set(branch_list) - available
        if missing:
            raise KeyError(
                f"Branch(es) missing from tree '{treename}' in {root_path}:\n"
                f"  {sorted(missing)}\n"
                "Check branch names against the ROOT file and your YAML config."
            )

        raw = tree.arrays(branch_list, library="np")
        return {name: np.asarray(raw[name]) for name in branch_list}
