import lerna
from lerna.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging

@lerna.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig) -> None:

    log = logging.getLogger(config.exp_name)
    
    log.info("==> Showing Config ...")
    log.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")
    trainer = instantiate(config.trainer, cfg=config, _recursive_=False)
    trainer.fit()

if __name__ == "__main__":
    main()
