from omegaconf import OmegaConf


def register_custom_resolvers():
    """Registers OmegaConf resolvers used across optim_env configs. Must run before
    any config containing these interpolations is loaded/resolved -- both the training
    entry point (tools/train.py) and anything re-loading a saved run's config (e.g.
    tools/test.py) need to call this first.
    """
    # Regression sets num_classes=0; models use dout: ${max:${dataset.num_classes},1}.
    OmegaConf.register_new_resolver("max", lambda *xs: max(xs), replace=True)

    # "evosax.algorithms.PSO" -> "PSO": run labels taken from _target_.
    OmegaConf.register_new_resolver(
        "basename", lambda s: str(s).replace("/", ".").rsplit(".", 1)[-1], replace=True
    )
