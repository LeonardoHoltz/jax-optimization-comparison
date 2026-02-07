from hydra.utils import instantiate
import logging
from pmlb import fetch_data

class DefaultTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.exp_name)
        self.logger.info(" => Creating dataset ...")
        self.create_datasets()
        self.logger.info(" => Creating model ...")
        self.build_model()
        self.logger.info(" => Creating hooks ...")
        self.create_hooks()

    def fit(self):
        pass

    def create_datasets(self):
        dataset_cfg = self.cfg.dataset
        self.X, self.y = fetch_data(
            dataset_name=dataset_cfg.name,
            return_X_y=True,
            dropna=True,
        )
        self.logger.info(f"Input data: {self.X.shape}")
        self.logger.info(f"Target data: {self.y.shape}")

    def build_model(self):
        model_cfg = self.cfg.model
        model_factory = instantiate(model_cfg, _partial_=True)
        self.model = model_factory(din=self.X.shape[-1], dout=self.y.shape[-1])

    def create_hooks(self):
        for hook_cfg in self.cfg.hooks:
            hook = instantiate(hook_cfg)
            self.hooks.append(hook)
