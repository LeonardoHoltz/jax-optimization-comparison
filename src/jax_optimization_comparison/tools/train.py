import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging



@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:

    log = logging.getLogger(cfg.exp_name)
    
    log.info("==> Showing Config ...")
    log.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    #breakpoint()
    trainer = instantiate(cfg.trainer, cfg)
    trainer.fit()
    

if __name__ == "__main__":
    main()
