"""Load a finished run (either framework) back into memory: config, model, weights.

Every run directory carries the exact Hydra config it was trained with, which is enough
to rebuild the model and find its checkpoint. tools/test.py and the notebooks both go
through here so there's one implementation of "run directory -> usable model".
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerna.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from optim_env.resolvers import register_custom_resolvers

register_custom_resolvers()


@dataclass
class LoadedRun:
    run_dir: Path
    cfg: DictConfig
    model: Any
    params: Any  # Flax params pytree; None for torch (weights live inside the module)
    framework: str

    @property
    def label(self) -> str:
        """Short human label, e.g. 'jax_NE/PSO/hpo'."""
        c = self.cfg
        return f"{c.get('framework','?')}_{c.get('method','?')}/{c.get('algorithm','?')}/{c.get('variant','?')}"


def load_config(run_dir: Path | str) -> DictConfig:
    config_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No .hydra/config.yaml in {run_dir} -- not a finished run directory")
    return OmegaConf.load(config_path)


def load_dataset(cfg: DictConfig):
    """The DataModule this run trained against (same splits, same preprocessing)."""
    return instantiate(cfg.dataset)


def load_run(run_dir: Path | str) -> LoadedRun:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    is_torch = "torch_env" in cfg.trainer._target_
    model = instantiate(cfg.model)

    if is_torch:
        from optim_env.torch_env.utils.checkpointing import load_checkpoint

        model.load_state_dict(load_checkpoint(run_dir / "best_model.pt"))
        model.eval()
        return LoadedRun(run_dir, cfg, model, None, "torch")

    import jax
    import jax.numpy as jnp
    from flax.training import train_state

    from optim_env.utils.checkpointing import load_checkpoint

    # The checkpoint is deserialized into a template of matching pytree structure, which
    # differs by trainer: SGDTrainer saves the whole TrainState, ESTrainer just the params.
    rng = jax.random.PRNGKey(0)  # only shapes matter here; values get overwritten
    dummy_x = jnp.zeros((1, cfg.dataset.num_features))
    init_params = model.init(rng, dummy_x)
    checkpoint_path = run_dir / "best_model.msgpack"

    if cfg.trainer._target_.endswith("SGDTrainer"):
        template = train_state.TrainState.create(
            apply_fn=model.apply, params=init_params, tx=instantiate(cfg.optimizer)
        )
        params = load_checkpoint(template, checkpoint_path).params
    elif cfg.trainer._target_.endswith("ESTrainer"):
        params = load_checkpoint(init_params, checkpoint_path)
    else:
        raise ValueError(f"Unsupported trainer for loading: {cfg.trainer._target_}")

    return LoadedRun(run_dir, cfg, model, params, "jax")


def predict(run: LoadedRun, X) -> Any:
    """Forward pass in inference mode (dropout off), returned as a numpy array."""
    import numpy as np

    if run.framework == "torch":
        import torch

        run.model.eval()
        with torch.no_grad():
            return run.model(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()

    import jax.numpy as jnp

    return np.asarray(run.model.apply(run.params, jnp.asarray(X), deterministic=True))
