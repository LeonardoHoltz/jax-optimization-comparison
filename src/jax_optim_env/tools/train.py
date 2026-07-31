import lerna
from lerna.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging

# Regression datasets set num_classes=0 (there's no class count); models still need a
# real output width of 1 in that case, hence dout: ${max:${dataset.num_classes},1}.
OmegaConf.register_new_resolver("max", lambda *xs: max(xs), replace=True)

@lerna.main(version_base=None, config_path="../configs", config_name="sgd_exp")
def main(config: DictConfig) -> None:

    log = logging.getLogger(config.exp_name)
    
    log.info("==> Showing Config ...")
    log.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")
    trainer = instantiate(config.trainer, cfg=config, _recursive_=False)
    trainer.fit()

if __name__ == "__main__":
    main()
