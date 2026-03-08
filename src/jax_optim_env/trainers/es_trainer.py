from lerna.utils import instantiate
import logging
import jax
import jax.numpy as jnp
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
from lerna.core.hydra_config import HydraConfig
import evosax
from jax.flatten_util import ravel_pytree

# Some terminologies are a little different for evolution strategies (ES)
# There are no "epochs" here because there are no batches. Only one optimizer update happens
# considering the entire training dataset

class ESTrainer:
    def __init__(self, pop_size, cfg):
        self.pop_size
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
        def loss_fn(theta):
            params = self.unravel_fn(theta)
            preds = self.model.apply(params, self.x)
            return jnp.mean((preds - self.y) ** 2)

        def train_step(key, state):
            key, subkey = jax.random.split(key)

            population, state = self.strategy.ask(
                subkey,
                state,
                self.es_params
            )
            fitness = jax.vmap(loss_fn)(population)
            state = self.strategy.tell(
                population,
                fitness,
                state,
                self.es_params
            )
            return key, state, fitness

        self.step = jax.jit(train_step)

    def fit(self):
        for self.step in range(self.cfg.epochs):
            epoch_train_loss = self.train_step(self.rng, )
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
        
        loss_fn = self.loss_fn

        losses = []
        for i, (x, y) in enumerate(self.train_loader):
            self.state, loss = train_step(self.state, x, y)
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
        self.train_X = self.train_ds.X
        self.train_y = self.train_ds.y
        self.val_ds = dataset_manager.get_val_ds()
        self.val_X = self.val_ds.X
        self.val_y = self.val_ds.y

    def prepare_model(self):
        self.model = instantiate(self.cfg.model)
        dummy_x = jnp.zeros((1, self.train_ds.X.shape[1]))
        self.rng, init_rng = jax.random.split(self.rng)

        params = self.model.init(init_rng, dummy_x)
        theta0, unravel_fn = ravel_pytree(params)
        num_dims = theta0.shape[0]

        self.strategy = evosax.OpenES(
            popsize=self.pop_size,
            num_dims=num_dims
        )
        self.es_params = self.strategy.default_params
        self.state = self.strategy.initialize(self.rng, self.es_params)
    
    def prepare_loss_fn(self):
        loss_fn = instantiate(self.cfg.loss)
        self.batched_loss = jax.vmap(loss_fn, in_axes=(0, None, None))

    #def create_hooks(self):
    #    for hook_cfg in self.cfg.hooks:
    #        hook = instantiate(hook_cfg)
    #        self.hooks.append(hook)
