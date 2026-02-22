from hydra.utils import instantiate
import logging

class DefaultTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.exp_name)
        self.logger.info(" => Creating dataset ...")
        self.prepare_dataset()
        self.logger.info(" => Creating model ...")
        self.build_model()
        #self.logger.info(" => Creating hooks ...")
        #self.create_hooks()

    def fit(self):
        pass

    def prepare_dataset(self):
        dataset_manager = instantiate(self.cfg.dataset)
        self.train_loader = dataset_manager.get_train_loader()
        self.val_loader = dataset_manager.get_val_loader()

    def build_model(self):
        model_cfg = self.cfg.model
        model_factory = instantiate(model_cfg, _partial_=True)
        self.model = model_factory(din=self.X.shape[-1], dout=self.y.shape[-1])

    #def create_hooks(self):
    #    for hook_cfg in self.cfg.hooks:
    #        hook = instantiate(hook_cfg)
    #        self.hooks.append(hook)
