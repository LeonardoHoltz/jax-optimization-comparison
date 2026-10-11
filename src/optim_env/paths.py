"""Canonical output locations, mirroring configs/base/run_layout.yaml.

Training runs get their directory from Hydra (run_layout.yaml builds it via config
interpolation); these helpers are for tooling that needs to *find* or *place* results
outside a Hydra run -- tools/hpo.py, report aggregation, notebooks. tests/test_paths.py
asserts these stay in agreement with what an actual composed config produces.
"""

from pathlib import Path

OUTPUTS = Path("outputs")
RUNS = OUTPUTS / "runs"
HPO = OUTPUTS / "hpo"
REPORTS = OUTPUTS / "reports"


def run_dir(dataset: str, framework: str, method: str, algorithm: str, variant: str, seed: int) -> Path:
    """outputs/runs/<dataset>/<framework>_<method>/<algorithm>/<variant>/seed_<n>/"""
    return RUNS / dataset / f"{framework}_{method}" / algorithm / variant / f"seed_{seed}"


def hpo_dir(dataset: str, framework: str, method: str, algorithm: str) -> Path:
    """outputs/hpo/<dataset>/<framework>_<method>/<algorithm>/ -- holds trials.csv,
    best_params.json and the per-trial run directories for one search."""
    return HPO / dataset / f"{framework}_{method}" / algorithm

