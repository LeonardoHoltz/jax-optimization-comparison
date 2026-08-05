from omegaconf import OmegaConf


def register_custom_resolvers():
    """Registers OmegaConf resolvers used across jax_optim_env configs. Must run before
    any config containing these interpolations is loaded/resolved -- both the training
    entry point (tools/train.py) and anything re-loading a saved run's config (e.g.
    tools/test.py) need to call this first.
    """
    # Regression datasets set num_classes=0 (there's no class count); models still need a
    # real output width of 1 in that case, hence dout: ${max:${dataset.num_classes},1}.
    OmegaConf.register_new_resolver("max", lambda *xs: max(xs), replace=True)
