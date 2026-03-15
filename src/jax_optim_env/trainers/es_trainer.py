from lerna.utils import instantiate, get_class
import logging
import os
import jax
import jax.numpy as jnp
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
from lerna.core.hydra_config import HydraConfig
import evosax
from evosax.algorithms import CMA_ES, SimpleGA
from jax.flatten_util import ravel_pytree
from jax_optim_env.utils.metrics import accuracy, mse, mae
from jax_optim_env.utils.checkpointing import save_checkpoint

# Some terminologies are a little different for evolution strategies (ES)
# There are no "epochs" here because there are no batches. Only one optimizer update happens
# considering the entire training dataset

class ESTrainer:
    def __init__(self, cfg):
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
        self.best_fitness = float('inf')

    def _compile(self):
        def loss_fn(params):
            params = self.unravel_fn(params)
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

        def eval_step(params, X, y):
            params = self.unravel_fn(params)
            logits = self.model.apply(params, X)
            return self.loss_fn(logits, y), logits

        self.eval_step = jax.jit(eval_step)

    def fit(self):
        for self.gen in range(self.cfg.generations):
            self.key, self.state, fitness = self.train_step(self.key, self.state)
            fitness_min = float(fitness.min())
            fitness_mean = float(fitness.mean())
            self.writer.add_scalar(
                "train/fitness_min",
                fitness_min,
                self.gen
            )
            self.writer.add_scalar(
                "train/fitness_mean",
                fitness_mean,
                self.gen
            )
            self.logger.info(
                f"Generation {self.gen} | train fitness min={fitness_min:.4f}"
            )

            # Evaluation
            if self.gen % self.cfg.get("eval_every_gen", 10) == 0:
                val_loss, val_acc = self.evaluate()
                self.writer.add_scalar("val/loss", val_loss, self.gen)
                if val_acc is not None:
                    self.writer.add_scalar("val/accuracy", val_acc, self.gen)
                
                self.logger.info(f"Generation {self.gen} | val loss={val_loss:.4f}" + (f" accuracy={val_acc:.4f}" if val_acc is not None else ""))

                # Track best model in memory
                if val_loss < self.best_fitness:
                    self.best_fitness = val_loss
                    self.best_state = self.state
                    self.logger.info(f"Generation {self.gen} | New best val loss: {val_loss:.4f}")

        if hasattr(self, 'best_state') and self.best_state is not None:
            checkpoint_path = os.path.join(self.cfg.save_path, "best_model.msgpack")
            # Extract and unravel best params
            best_params_flat = self._get_best_params(self.best_state)
            best_params = self.unravel_fn(best_params_flat)
            save_checkpoint(best_params, checkpoint_path)
            self.logger.info(f"Final best model saved to {checkpoint_path}")

        self.writer.close()

    def evaluate(self):
        # Dynamically extract best parameters based on algorithm type
        best_params_flat = self._get_best_params(self.state)
        val_loss, logits = self.eval_step(best_params_flat, self.val_X, self.val_y)
        
        val_acc = None
        if self.val_y.ndim == 1 or (self.val_y.ndim == 2 and self.val_y.shape[1] > 1):
             val_acc = float(accuracy(logits, self.val_y))
        
        return float(val_loss), val_acc

    def _get_best_params(self, state):
        """Helper to extract best solution from either distribution or population state."""
        if hasattr(state, "mean"):
            return state.mean
        return state.best_solution

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

        params = self.model.init(init_key, dummy_x)

        # flatten pytree -> vector
        solution, self.unravel_fn = ravel_pytree(params)

        # Creates an evosax.algorithm object using get_class pattern
        strategy_cls = get_class(self.cfg.optimizer._target_)
        self.strategy = strategy_cls(
            population_size=self.cfg.optimizer.population_size,
            solution=solution,
        )

        from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
        from evosax.algorithms.population_based.base import PopulationBasedAlgorithm

        self.params = self.strategy.default_params
        
        if isinstance(self.strategy, DistributionBasedAlgorithm):
            self.state = self.strategy.init(self.key, solution, self.params)
        elif isinstance(self.strategy, PopulationBasedAlgorithm):
            # Population-based expects (key, population, fitness, params)
            # We initialize by broadcasting the initial solution
            init_pop = jnp.tile(solution, (self.strategy.population_size, 1))
            init_fitness = jnp.full(self.strategy.population_size, jnp.inf)
            self.state = self.strategy.init(self.key, init_pop, init_fitness, self.params)
        else:
            # Fallback to base EvolutionaryAlgorithm signature (key, params)
            self.state = self.strategy.init(self.key, self.params)
    
    def prepare_loss_fn(self):
        self.loss_fn = instantiate(self.cfg.loss)

    #def create_hooks(self):
    #    for hook_cfg in self.cfg.hooks:
    #        hook = instantiate(hook_cfg)
    #        self.hooks.append(hook)
