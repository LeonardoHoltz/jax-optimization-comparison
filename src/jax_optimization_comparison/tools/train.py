import hydra
from omegaconf import DictConfig, OmegaConf
import logging



@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:

    log = logging.getLogger(__name__)
    
    log.info("==> Showing Config ...")
    log.info(OmegaConf.to_yaml(cfg))
    dataset = hydra.utils.instantiate(cfg.dataset)

if __name__ == "__main__":
    main()
