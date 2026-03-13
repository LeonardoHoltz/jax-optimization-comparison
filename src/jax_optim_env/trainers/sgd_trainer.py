from lerna.utils import instantiate
import logging
import jax
import jax.numpy as jnp
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
from lerna.core.hydra_config import HydraConfig

class SGDTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.exp_name)
        output_log_dir = HydraConfig.get().runtime.output_dir
        self.writer = SummaryWriter(output_log_dir)

        self.rng = jax.random.PRNGKey(cfg.seed)

        self.logger.info(jax.devices())

        self.logger.info(" => Creating dataset ...")
        self.prepare_dataset()
        self.logger.info(" => Creating model ...")
        self.prepare_model()
        self.logger.info(" => Creating loss fn ...")
        self.prepare_loss_fn()
        self.logger.info(" => Compiling JIT functions ...")
        self._compile()
        #self.logger.info(" => Creating hooks ...")
        #self.create_hooks()

    def _compile(self):
        def train_step(state, x, y):
            def apply_loss_fn(params):
                pred = state.apply_fn(params, x)
                return self.loss_fn(pred, y)

            # For jax environments, the reference to the model through the training
            # is done using the state rather to the model object itself
            loss, grads = jax.value_and_grad(apply_loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
        self.train_step = jax.jit(train_step)

    def fit(self):
        for self.epoch in range(self.cfg.epochs):
            epoch_train_loss = self.run_train_epoch()
            epoch_train_loss = float(epoch_train_loss)
        
            self.logger.info(
                f"Epoch {self.epoch} | train={epoch_train_loss:.4f}"
            )
            self.writer.add_scalar(
                "train/loss_epoch",
                epoch_train_loss,
                self.epoch
            )
        self.writer.close()
    
    def run_train_epoch(self):
        losses = []
        for i, (x, y) in enumerate(self.train_loader):
            self.state, loss = self.train_step(self.state, x, y)
            loss_value = float(loss)
            losses.append(loss)
            self.writer.add_scalar(
                "train/loss_step",
                loss_value,
                self.epoch * len(self.train_loader) + i
            )
        return jnp.mean(jnp.stack(losses))

    def prepare_dataset(self):
        dataset_manager = instantiate(self.cfg.dataset)
        self.train_ds = dataset_manager.get_train_ds()
        self.val_ds = dataset_manager.get_val_ds()
        self.train_loader = dataset_manager.get_train_loader(self.train_ds)
        self.val_loader = dataset_manager.get_val_loader(self.val_ds)

    def prepare_model(self):
        self.model = instantiate(self.cfg.model)
        dummy_x = jnp.zeros((1, self.train_ds.X.shape[1]))
        self.rng, init_rng = jax.random.split(self.rng)

        params = self.model.init(init_rng, dummy_x)
        optimizer = instantiate(self.cfg.optimizer)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
        )
    
    def prepare_loss_fn(self):
        self.loss_fn = instantiate(self.cfg.loss)

    #def create_hooks(self):
    #    for hook_cfg in self.cfg.hooks:
    #        hook = instantiate(hook_cfg)
    #        self.hooks.append(hook)
