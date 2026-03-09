from lerna.utils import instantiate
import logging
import jax
import jax.numpy as jnp
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
from lerna.core.hydra_config import HydraConfig
import evosax
from evosax.algorithms import CMA_ES, SimpleGA
from jax.flatten_util import ravel_pytree

# Some terminologies are a little different for evolution strategies (ES)
# There are no "epochs" here because there are no batches. Only one optimizer update happens
# considering the entire training dataset

class ESTrainer:
    def __init__(self, pop_size, cfg):
        self.pop_size = pop_size
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.exp_name)
        output_log_dir = HydraConfig.get().runtime.output_dir
        self.writer = SummaryWriter(output_log_dir)

        self.key = jax.random.PRNGKey(cfg.seed)

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
        def loss_fn(params):
            pred = self.model.apply(params, self.train_X)
            return self.loss_fn(pred, self.train_y)

        def train_step(key, state):
            key, key_ask, key_tell = jax.random.split(key, 3)

            population, state = self.strategy.ask(
                key_ask,
                state,
                self.params
            )
            fitness = jax.vmap(loss_fn)(population)
            state, _ = self.strategy.tell(
                key_tell,
                population,
                fitness,
                state,
                self.params
            )
            return key, state, fitness

        self.train_step = jax.jit(train_step)

    def fit(self):
        for self.step in range(self.cfg.epochs):
            self.key, self.state, fitness = self.train_step(self.key, self.state)
            fitness_min = float(fitness.min())
            fitness_mean = float(fitness.mean())
            self.writer.add_scalar(
                "train/fitness_min",
                fitness_min,
                self.step
            )
            self.writer.add_scalar(
                "train/fitness_mean",
                fitness_mean,
                self.step
            )
            self.logger.info(
                f"Step {self.step} | train fitness min={fitness_min:.4f}"
            )
        self.writer.close()

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
        self.key, init_key = jax.random.split(self.key)

        solution = self.model.init(init_key, dummy_x) # The model params is referenced in evosax terminology as solution

        self.strategy = CMA_ES(
            population_size=self.pop_size,
            solution=solution # requires a dummy solution
        )
        self.params = self.strategy.default_params
        self.state = self.strategy.init(self.key, solution, self.params)
    
    def prepare_loss_fn(self):
        self.loss_fn = instantiate(self.cfg.loss)

    #def create_hooks(self):
    #    for hook_cfg in self.cfg.hooks:
    #        hook = instantiate(hook_cfg)
    #        self.hooks.append(hook)
