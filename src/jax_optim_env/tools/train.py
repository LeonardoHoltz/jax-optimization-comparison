import lerna
from lerna.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging

from jax_optim_env.resolvers import register_custom_resolvers

register_custom_resolvers()

@lerna.main(version_base=None, config_path="../configs", config_name="sgd_exp")
def main(config: DictConfig) -> None:

    log = logging.getLogger(config.exp_name)
    
    log.info("==> Showing Config ...")
    log.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")
    trainer = instantiate(config.trainer, cfg=config, _recursive_=False)
    trainer.fit()

if __name__ == "__main__":
    main()
