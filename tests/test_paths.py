"""optim_env.paths mirrors the layout that configs/base/run_layout.yaml builds via
config interpolation. These are two separate expressions of the same convention, so this
test pins them together -- if run_layout.yaml's template changes, the helpers that tooling
uses to *find* results must change with it.
"""

from omegaconf import OmegaConf

from optim_env.paths import hpo_dir, run_dir
from optim_env.resolvers import register_custom_resolvers

register_custom_resolvers()

LAYOUT = "src/optim_env/configs/base/run_layout.yaml"


def _resolved_run_dir(**overrides):
    """Resolve run_dir the way Hydra would, straight from run_layout.yaml."""
    layout = OmegaConf.load(LAYOUT)
    cfg = OmegaConf.merge(layout, OmegaConf.create(overrides))
    return OmegaConf.to_container(cfg, resolve=True)["run_dir"]


def test_run_dir_helper_matches_config_layout():
    expected = _resolved_run_dir(
        framework="jax",
        method="NE",
        variant="baseline",
        seed=3,
        dataset={"name": "california"},
        optimizer={"_target_": "evosax.algorithms.PSO"},
    )
    assert str(run_dir("california", "jax", "NE", "PSO", "baseline", 3)) == expected


def test_run_dir_helper_matches_config_layout_for_torch_hpo_variant():
    expected = _resolved_run_dir(
        framework="torch",
        method="SGD",
        variant="hpo",
        seed=0,
        dataset={"name": "adult"},
        optimizer={"_target_": "torch.optim.AdamW"},
    )
    assert str(run_dir("adult", "torch", "SGD", "AdamW", "hpo", 0)) == expected


def test_hpo_dir_is_separate_from_runs_tree():
    # Search trials are means, not results -- they must not land in runs/, where
    # tools/test.py's default glob would pick them up as if they were real runs.
    hpo = hpo_dir("california", "jax", "NE", "PSO")
    assert "runs" not in hpo.parts
    assert hpo.parts[:2] == ("outputs", "hpo")
