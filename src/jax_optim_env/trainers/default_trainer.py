from hydra.utils import instantiate
import logging
import jax
import jax.numpy as jnp

class DefaultTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.exp_name)
        self.rng = jax.random.PRNGKey(cfg.seed)

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
        self.train_ds = dataset_manager.get_train_ds()
        self.val_ds = dataset_manager.get_val_ds()
        self.train_loader = dataset_manager.get_train_loader(self.train_ds)
        self.val_loader = dataset_manager.get_val_loader(self.val_ds)

    def build_model(self):
        model = instantiate(self.cfg.model)
        dummy_x = jnp.zeros((1, self.train_ds.X.shape[1]))
        self.rng, init_rng = jax.random.split(self.rng)

        params = self.model.init(init_rng, dummy_x)
        

    #def create_hooks(self):
    #    for hook_cfg in self.cfg.hooks:
    #        hook = instantiate(hook_cfg)
    #        self.hooks.append(hook)
