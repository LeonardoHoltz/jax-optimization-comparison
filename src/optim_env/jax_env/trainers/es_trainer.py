from lerna.utils import instantiate, get_class
import inspect
import json
import logging
import math
import os
import jax
import jax.numpy as jnp
import optax
from torch.utils.tensorboard import SummaryWriter
from lerna.core.hydra_config import HydraConfig
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jax.flatten_util import ravel_pytree
from optim_env.utils.metrics import accuracy
from optim_env.utils.checkpointing import save_checkpoint

class ESTrainer:
    """Trains a model with an evosax evolution strategy. One update per generation, no
    epochs: each generation scores the whole population on the training set (or on one
    random mini-batch, with fitness_batch_size)."""

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
        self.best_fitness = float('inf')
        self.best_params = None

    def _batch_sampler(self):
        """key -> (X, y) for one generation."""
        # The whole population is scored at once (vmap); for images or text the full
        # training set won't fit in GPU memory, hence fitness_batch_size.
        fitness_batch_size = self.cfg.get("fitness_batch_size", None)
        n_train = self.train_X.shape[0]
        if fitness_batch_size is None or fitness_batch_size >= n_train:
            return lambda key: (self.train_X, self.train_y)
        batch_size = int(fitness_batch_size)
        self.logger.info(f"ES fitness on random mini-batches of {batch_size} (train n={n_train})")

        def sample(key):
            # One batch per generation, shared by every candidate, so they're ranked on
            # the same data.
            idx = jax.random.choice(key, n_train, (batch_size,), replace=False)
            return self.train_X[idx], self.train_y[idx]
        return sample

    def _eval_step(self, params, X, y):
        params = self.unravel_fn(params)
        logits = self.model.apply(params, X, deterministic=True)
        return self.loss_fn(logits, y), logits

    def _compile(self):
        sample_batch = self._batch_sampler()

        def loss_fn(params, X, y):
            params = self.unravel_fn(params)
            # Dropout off: a candidate's fitness must depend on its weights only.
            pred = self.model.apply(params, X, deterministic=True)
            return self.loss_fn(pred, y)

        def train_step(key, state):
            key, key_ask, key_tell, key_batch = jax.random.split(key, 4)
            X, y = sample_batch(key_batch)

            population, state = self.strategy.ask(
                key_ask,
                state,
                self.params
            )
            fitness = jax.vmap(loss_fn, in_axes=(0, None, None))(population, X, y)
            state, _ = self.strategy.tell(
                key_tell,
                population,
                fitness,
                state,
                self.params
            )
            return key, state, fitness

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(self._eval_step)

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

            # The last generation is checked too, so its weights can become the checkpoint.
            if self.gen % self.cfg.get("eval_every_gen", 10) == 0 or self.gen == self.cfg.generations - 1:
                params = self._get_current_params(self.state)
                val_loss, val_acc = self.evaluate(params)
                self.writer.add_scalar("val/loss", val_loss, self.gen)
                if val_acc is not None:
                    self.writer.add_scalar("val/accuracy", val_acc, self.gen)
                
                self.logger.info(f"Generation {self.gen} | val loss={val_loss:.4f}" + (f" accuracy={val_acc:.4f}" if val_acc is not None else ""))

                # Keep the best-on-validation candidate, as the SGD trainers do. evosax's
                # state.best_solution is the best *training* fitness ever seen, which on
                # mini-batches is the candidate that drew the easiest batch. A NaN run
                # keeps its first candidate so there is always a checkpoint.
                if self.best_params is None or val_loss < self.best_fitness:
                    self.best_fitness = float("inf") if math.isnan(val_loss) else val_loss
                    self.best_params = params
                    self.logger.info(f"Generation {self.gen} | New best val loss: {val_loss:.4f}")

        # HPO trials skip it: they only need the val loss, and big models fill the disk.
        if self.cfg.get("save_checkpoint", True):
            checkpoint_path = os.path.join(self.cfg.save_path, "best_model.msgpack")
            save_checkpoint(self.unravel_fn(self.best_params), checkpoint_path)
            self.logger.info(f"Final best model saved to {checkpoint_path}")

        # Read by tools/hpo.py and the study report; best_val_loss belongs to the checkpoint.
        metrics_path = os.path.join(self.cfg.save_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"best_val_loss": self.best_fitness, "generations": self.cfg.generations}, f)

        self.writer.close()

    def evaluate(self, params_flat):
        val_loss, logits = self.eval_step(params_flat, self.val_X, self.val_y)

        val_acc = None
        is_classification = self.cfg.dataset.num_classes and self.cfg.dataset.num_classes > 0
        if is_classification and (self.val_y.ndim == 1 or (self.val_y.ndim == 2 and self.val_y.shape[1] > 1)):
             val_acc = float(accuracy(logits, self.val_y))

        return float(val_loss), val_acc

    def _get_current_params(self, state):
        """The candidate `fit` validates: the search mean for distribution-based
        algorithms, the current population's best member for population-based ones."""
        if hasattr(state, "mean"):
            return state.mean
        return state.population[jnp.argmin(state.fitness)]

    def prepare_dataset(self):
        dataset_manager = instantiate(self.cfg.dataset)
        self.train_ds = dataset_manager.get_train_ds()
        # Device arrays: the jitted train_step indexes them with a traced index.
        self.train_X = jnp.asarray(self.train_ds.X)
        self.train_y = jnp.asarray(self.train_ds.y)
        self.val_ds = dataset_manager.get_val_ds()
        self.val_X = jnp.asarray(self.val_ds.X)
        self.val_y = jnp.asarray(self.val_ds.y)

    def _init_solution(self):
        """Builds self.model and self.unravel_fn; returns the initial weights as one vector."""
        self.model = instantiate(self.cfg.model)
        dummy_x = jnp.zeros((1, self.train_ds.X.shape[1]))
        self.key, init_key = jax.random.split(self.key)
        solution, self.unravel_fn = ravel_pytree(self.model.init(init_key, dummy_x))
        return solution

    def prepare_model(self):
        solution = self._init_solution()

        # evosax samples with std 1.0 by default; a fresh network's weights have std ~0.07,
        # so default candidates are ~14x too large and losses start in the millions. Every
        # algorithm gets a spread matched to the weights instead (init_std_multiplier lets
        # HPO move it). The three algorithm families take it through different hooks.
        init_std = float(jnp.std(solution)) * float(self.cfg.get("init_std_multiplier", 1.0))
        self.logger.info(f"ES init_std (matched to model's own weight scale): {init_std:.6g}")

        strategy_cls = get_class(self.cfg.optimizer._target_)
        strategy_init_params = inspect.signature(strategy_cls.__init__).parameters
        strategy_kwargs = {"population_size": self.cfg.optimizer.population_size, "solution": solution}
        if "std_schedule" in strategy_init_params:
            # Hook 1, constructor std_schedule (SimpleGA, Open_ES), with opposite needs:
            # - population-based: mutants become the next population, so a constant std
            #   is a random walk that drifts away. Decay to 1% within ~20 generations.
            # - distribution-based: std divides the gradient estimate twice, so shrinking it
            #   amplifies noise (Open_ES went from 0.5 to millions). Keep it constant.
            if issubclass(strategy_cls, PopulationBasedAlgorithm):
                strategy_kwargs["std_schedule"] = optax.exponential_decay(
                    init_value=init_std, transition_steps=20, decay_rate=0.01, end_value=init_std * 0.01
                )
            else:
                strategy_kwargs["std_schedule"] = optax.constant_schedule(init_std)
        if "optimizer" in strategy_init_params:
            # Algorithms that step along an estimated gradient (Open_ES, PGPE, SNES, ARS).
            strategy_kwargs["optimizer"] = optax.sgd(learning_rate=float(self.cfg.get("es_learning_rate", 1e-3)))
        self.strategy = strategy_cls(**strategy_kwargs)
        self.params = self.strategy.default_params

        if isinstance(self.strategy, DistributionBasedAlgorithm):
            # Hook 2, the std_init param (CMA-ES family). An oversized start makes their
            # step-size adaptation keep growing the step instead of correcting it.
            if hasattr(self.params, "std_init"):
                self.params = self.params.replace(std_init=init_std)
            self.state = self.strategy.init(self.key, solution, self.params)
        elif isinstance(self.strategy, PopulationBasedAlgorithm):
            # Hook 3, the initial population. Identical copies give PSO zero velocity
            # forever (every difference term is zero), so start from noisy copies.
            self.key, pop_key = jax.random.split(self.key)
            noise = init_std * jax.random.normal(pop_key, (self.strategy.population_size, solution.shape[0]))
            init_fitness = jnp.full(self.strategy.population_size, jnp.inf)
            self.state = self.strategy.init(self.key, solution[None, :] + noise, init_fitness, self.params)

    def prepare_loss_fn(self):
        self.loss_fn = instantiate(self.cfg.loss)
