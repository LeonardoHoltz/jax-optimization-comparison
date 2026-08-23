from lerna.utils import instantiate, get_class
import inspect
import logging
import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
from lerna.core.hydra_config import HydraConfig
import evosax
from evosax.algorithms import CMA_ES, SimpleGA
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jax.flatten_util import ravel_pytree
from optim_env.utils.metrics import accuracy, mse, mae
from optim_env.utils.checkpointing import save_checkpoint

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
            # Deterministic forward pass: dropout must be inert here since ES fitness
            # signals need to be a stable function of params, not noisy across candidates.
            pred = self.model.apply(params, self.train_X, deterministic=True)
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
            logits = self.model.apply(params, X, deterministic=True)
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

        # Saved from the final training state, not self.best_state: state.best_solution
        # (see _get_best_params) already accumulates the best individual across every single
        # generation of the whole run, which is more complete than self.best_state's
        # eval_every_gen-spaced sampling used for the "New best val loss" progress log above.
        checkpoint_path = os.path.join(self.cfg.save_path, "best_model.msgpack")
        best_params_flat = self._get_best_params(self.state)
        best_params = self.unravel_fn(best_params_flat)
        save_checkpoint(best_params, checkpoint_path)
        self.logger.info(f"Final best model saved to {checkpoint_path}")

        self.writer.close()

    def evaluate(self):
        # Dynamically extract the current-generation's best parameters based on algorithm type
        best_params_flat = self._get_current_params(self.state)
        val_loss, logits = self.eval_step(best_params_flat, self.val_X, self.val_y)

        val_acc = None
        is_classification = self.cfg.dataset.num_classes and self.cfg.dataset.num_classes > 0
        if is_classification and (self.val_y.ndim == 1 or (self.val_y.ndim == 2 and self.val_y.shape[1] > 1)):
             val_acc = float(accuracy(logits, self.val_y))

        return float(val_loss), val_acc

    def _get_current_params(self, state):
        """Best individual as of the *current* generation, for the per-generation progress
        metric logged during training (see `evaluate`). Distribution-based algorithms have
        no discrete population -- their current position is the search-distribution mean,
        which moves (and can get worse) every generation. Population-based algorithms are
        matched to that same "current, not historical" semantics by reading the current
        population's own best individual, using the same raw fitness values already logged
        as "train fitness min" each generation (population/fitness are stored post-`tell`,
        so they're read consistently)."""
        if hasattr(state, "mean"):
            return state.mean
        return state.population[jnp.argmin(state.fitness)]

    def _get_best_params(self, state):
        """Best solution across the *entire run so far*, used for the final saved
        checkpoint. Both algorithm categories track this internally in evosax itself
        (`state.best_solution`/`best_fitness`, updated every single generation -- not just
        at our eval_every_gen checkpoints -- via the same `update_best_solution_and_fitness`
        helper shared by DistributionBasedAlgorithm.tell and PopulationBasedAlgorithm.tell),
        so unlike `_get_current_params` there's no category-specific branching needed here."""
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

        # ---------------------------------------------------------------------------
        # Background: why ES needs a model-scale-aware initial spread
        # ---------------------------------------------------------------------------
        # Problem: every evosax algorithm defaults to a generic, scale-agnostic initial
        # spread (std=1.0). That default assumes nothing about the search space -- but our
        # search space IS a neural network's weights, which live on a much smaller scale
        # set by the init scheme (Kaiming/lecun), not on the "1.0" evosax assumes.
        #
        # Example: a 3-layer MLP with ~235K parameters typically has std(weights) ~ 0.07
        # right after init. Sampling candidates with evosax's default std=1.0 around that
        # means every candidate is ~14x larger than a normally-scaled set of weights for
        # this network. Pushed through several ReLU layers, that alone is enough to produce
        # predictions -- and losses -- in the millions from generation 0, before any real
        # optimization has happened.
        #
        # Fix: compute init_std directly from the freshly-initialized network's own weights,
        # then thread it through evosax wherever each algorithm *category* exposes a hook for
        # it (see the three cases marked below). We branch on category (distribution-based vs
        # population-based, already needed further down for the ask/tell API differences)
        # plus generic introspection (hasattr/inspect.signature), not on specific algorithm
        # class names, so this works uniformly for any current or future evosax algorithm.
        init_std = float(jnp.std(solution))
        self.logger.info(f"ES init_std (matched to model's own weight scale): {init_std:.6g}")

        # Case 1 -- constructor-level `std_schedule` (e.g. SimpleGA, Open_ES).
        # Problem: for these algorithms, the per-generation mutation/perturbation std is not
        # a runtime Params field we could override later -- it's a schedule function, called
        # fresh every generation with the current generation index (see e.g. SimpleGA._tell:
        # `std=self.std_schedule(state.generation_counter)`), fixed once when the strategy
        # object is constructed (default: constant 1.0, forever).
        #
        # The right fix here splits by algorithm *category*, because the two categories use
        # `std` for mathematically different things and need opposite treatment:
        #
        # Population-based (e.g. SimpleGA): mutated offspring directly become the next
        # generation's population -- there is no separate "take a small, bounded step" layer.
        # A *constant* schedule (even one correctly matched to the model's scale at generation
        # 0) is still a fixed-size perturbation applied every generation with nothing shrinking
        # it over time -- an unconstrained random walk whose typical distance from the start
        # grows generation over generation, regardless of whether any given step helped. That
        # is a genuine long-run blowup even though generation 0 itself looked well-scaled.
        # Fix: decay std quickly (exponential decay to 1% of init_std within the first ~20
        # generations, then hold near that floor) so the population stops drifting once early
        # exploration is done.
        #
        # Distribution-based (e.g. Open_ES): std does NOT directly become the step size --
        # it only scales a *gradient estimate* computed from population fitness (see
        # Open_ES._tell: `grad = dot(fitness, (population - mean) / std) / (population_size *
        # std)`). std appears in the denominator *twice*. Shrinking std therefore doesn't cool
        # anything -- it amplifies the gradient estimate's noise, up to 1/std^2. Verified
        # empirically: applying the same fast decay used for SimpleGA to Open_ES made its val
        # loss explode from ~0.5 to millions within 300 generations. The actual step size for
        # these algorithms is already bounded by their own internal optimizer (default
        # optax.sgd(learning_rate=1e-3)), so std should just stay constant at the model-matched
        # scale, exactly like Case 2 below does via std_init.
        strategy_cls = get_class(self.cfg.optimizer._target_)
        strategy_kwargs = {"population_size": self.cfg.optimizer.population_size, "solution": solution}
        if "std_schedule" in inspect.signature(strategy_cls.__init__).parameters:
            if issubclass(strategy_cls, PopulationBasedAlgorithm):
                strategy_kwargs["std_schedule"] = optax.exponential_decay(
                    init_value=init_std, transition_steps=20, decay_rate=0.01, end_value=init_std * 0.01
                )
            else:
                strategy_kwargs["std_schedule"] = optax.constant_schedule(init_std)
        self.strategy = strategy_cls(**strategy_kwargs)

        self.params = self.strategy.default_params

        if isinstance(self.strategy, DistributionBasedAlgorithm):
            # Case 2 -- distribution-based algorithms with a `std_init` Params field
            # (e.g. CMA_ES, Sep_CMA_ES).
            # Problem: here the initial sampling spread IS a runtime-overridable Params
            # field, called std_init -- but it still defaults to the same generic 1.0.
            # Example: a CMA-ES-family algorithm sampling with std_init=1.0 around much
            # smaller-scale weights starts generation 0 already oversized. Its step-size
            # self-adaptation heuristic then reads the resulting noisy, directionless
            # progress (typical for a landscape this high-dimensional) as "we're not
            # converging", and keeps inflating the step size further each generation --
            # compounding, rather than correcting, the original oversized start.
            # Fix: override params.std_init to init_std before calling strategy.init(...).
            # Not every distribution-based algorithm exposes this field (e.g. Open_ES
            # doesn't -- it's handled by Case 1's std_schedule instead), so this is a
            # best-effort override via hasattr, not a hard requirement.
            if hasattr(self.params, "std_init"):
                self.params = self.params.replace(std_init=init_std)
            self.state = self.strategy.init(self.key, solution, self.params)
        elif isinstance(self.strategy, PopulationBasedAlgorithm):
            # Case 3 -- population-based algorithms seeded with duplicate individuals
            # (e.g. PSO, SimpleGA).
            # Problem: the population must be seeded with an explicit initial population
            # array (there's no single "mean" to start from), and doing that by tiling
            # the same solution population_size times gives every individual an
            # identical starting point -- zero diversity to search with.
            # Example: PSO's velocity update is
            #   velocity = inertia*velocity + cognitive*r1*(personal_best - member)
            #                                + social*r2*(global_best - member)
            # With every particle starting at the same point, personal_best equals member
            # and global_best equals member for every particle, so both difference terms
            # are exactly zero. Velocity starts at zero and stays at zero forever -- the
            # particles have no way to start moving.
            # Fix: seed the initial population as solution + init_std * Gaussian noise
            # instead of exact duplicates, so there is real diversity for the algorithm's
            # ask/tell dynamics to work with from generation 0.
            self.key, pop_key = jax.random.split(self.key)
            noise = init_std * jax.random.normal(pop_key, (self.strategy.population_size, solution.shape[0]))
            init_pop = solution[None, :] + noise
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
