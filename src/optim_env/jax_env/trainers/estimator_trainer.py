import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from flax import traverse_util
from jax.flatten_util import ravel_pytree

from optim_env.jax_env.trainers.es_trainer import ESTrainer


def _is_dense_call(ctx):
    return isinstance(ctx.module, nn.Dense) and ctx.method_name == "__call__"


class EstimatorTrainer(ESTrainer):
    """ESTrainer for forward-only methods that estimate a gradient themselves and step
    with optax SGD instead of an evosax strategy. Subclasses define
    _estimate(params, key, X, y) -> (gradient pytree, losses to log)."""

    def prepare_model(self):
        solution = self._init_solution()
        self.optimizer = optax.sgd(
            float(self.cfg.get("es_learning_rate", 1e-3)), momentum=self.cfg.optimizer.get("momentum")
        )
        self.state = (solution, self.optimizer.init(solution))

    def _compile(self):
        sample_batch = self._batch_sampler()

        def train_step(key, state):
            solution, opt_state = state
            key, key_batch, key_noise = jax.random.split(key, 3)
            X, y = sample_batch(key_batch)
            grad, losses = self._estimate(self.unravel_fn(solution), key_noise, X, y)
            updates, opt_state = self.optimizer.update(ravel_pytree(grad)[0], opt_state, solution)
            return key, (optax.apply_updates(solution, updates), opt_state), losses

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(self._eval_step)

    def _get_current_params(self, state):
        return state[0]


class EggrollTrainer(EstimatorTrainer):
    """EGGROLL (Sarkar et al. 2025): antithetic Open-ES whose noise on each Dense kernel
    is low-rank, E = A Bᵀ/√r. The forward adds σ(xA)Bᵀ to xW instead of building W + σE,
    so every candidate shares one copy of the kernels. Other parameters (biases, conv
    kernels, embeddings) get ordinary full-rank noise."""

    def prepare_model(self):
        super().prepare_model()
        # Weight-space noise, matched to the model's own weight scale like ESTrainer's.
        self.sigma = float(jnp.std(self.state[0])) * float(self.cfg.get("init_std_multiplier", 1.0))
        self.logger.info(f"EGGROLL sigma={self.sigma:.6g} rank={self.cfg.optimizer.rank}")

    def _member_loss(self, params, eps, scale, X, y):
        """Loss of params + scale·eps, where eps holds (A, B) for each Dense kernel."""
        r = self.cfg.optimizer.rank
        flat = traverse_util.flatten_dict(params)
        perturbed = {k: v if isinstance(eps[k], tuple) else v + scale * eps[k] for k, v in flat.items()}

        def low_rank(f, args, kwargs, ctx):
            out = f(*args, **kwargs)
            if _is_dense_call(ctx):
                A, B = eps[("params", *ctx.module.path, "kernel")]
                out = out + scale / jnp.sqrt(r) * (args[0] @ A) @ B.T
            return out

        with nn.intercept_methods(low_rank):
            pred = self.model.apply(traverse_util.unflatten_dict(perturbed), X, deterministic=True)
        return self.loss_fn(pred, y)

    def _estimate(self, params, key, X, y):
        pairs, r = self.cfg.optimizer.population_size // 2, self.cfg.optimizer.rank
        flat = traverse_util.flatten_dict(params)
        noise = {}
        for k, key_k in zip(flat, jax.random.split(key, len(flat))):
            shape = flat[k].shape
            # shortcut: a 2-D kernel is taken to be a Dense one; true for every repo model.
            if k[-1] == "kernel" and len(shape) == 2:
                key_a, key_b = jax.random.split(key_k)
                noise[k] = (jax.random.normal(key_a, (pairs, shape[0], r)), jax.random.normal(key_b, (pairs, shape[1], r)))
            else:
                noise[k] = jax.random.normal(key_k, (pairs, *shape))

        member_losses = jax.vmap(self._member_loss, (None, 0, None, None, None))
        plus = member_losses(params, noise, self.sigma, X, y)
        minus = member_losses(params, noise, -self.sigma, X, y)
        # Centered ranks and Σ f̃ᵢεᵢ/(Nσ), as evosax's Open_ES does, so the two differ only
        # in the noise. Raw loss differences went NaN at learning rates Open_ES handles.
        shaped = centered_rank_fitness_shaping_fn(None, jnp.concatenate([plus, minus]), None, None)
        d = (shaped[:pairs] - shaped[pairs:]) / (2 * pairs * self.sigma)
        grad = {
            k: jnp.einsum("n,nir,nor->io", d, *e) / jnp.sqrt(r) if isinstance(e, tuple) else jnp.tensordot(d, e, 1)
            for k, e in noise.items()
        }
        return traverse_util.unflatten_dict(grad), jnp.concatenate([plus, minus])


class DustTrainer(EstimatorTrainer):
    """DUST (github.com/qlabs-eng/dust): forward-only node perturbation. Each draw adds
    Gaussian noise to every Dense output, independently per example; the change in each
    example's loss times that noise estimates δ, the loss gradient at that output. A
    Dense kernel's gradient is then xᵀδ with x its cached input, as in backprop. Dense
    layers only: other parameters would need their own local rule."""

    def _estimate(self, params, key, X, y):
        draws, sigma = self.cfg.optimizer.population_size, float(self.cfg.optimizer.sigma)
        per_example = jax.vmap(lambda pred, t: self.loss_fn(pred[None], t[None]))

        inputs, outputs = {}, {}

        def record(f, args, kwargs, ctx):
            out = f(*args, **kwargs)
            if _is_dense_call(ctx):
                inputs[ctx.module.path], outputs[ctx.module.path] = args[0], out
            return out

        with nn.intercept_methods(record):
            self.model.apply(params, X, deterministic=True)
        keys = jax.random.split(key, len(outputs))
        noise = {path: jax.random.normal(k, (draws, *out.shape)) for (path, out), k in zip(outputs.items(), keys)}

        def jittered_losses(a):
            def jitter(f, args, kwargs, ctx):
                out = f(*args, **kwargs)
                return out + sigma * a[ctx.module.path] if _is_dense_call(ctx) else out

            with nn.intercept_methods(jitter):
                return per_example(self.model.apply(params, X, deterministic=True), y)

        losses = jax.vmap(jittered_losses)(noise)  # (draws, batch)
        # Baseline: the draws' own mean, which keeps (draws-1)/draws of the signal.
        weights = (losses - losses.mean(0)) / (sigma * (draws - 1))

        flat = traverse_util.flatten_dict(params)
        grad = {}
        for path, a in noise.items():
            delta = jnp.einsum("nb,nb...->b...", weights, a)  # d loss_b / d output_b
            grad[("params", *path, "kernel")] = jnp.einsum("b...i,b...o->io", inputs[path], delta) / X.shape[0]
            if ("params", *path, "bias") in flat:
                grad[("params", *path, "bias")] = delta.reshape(-1, delta.shape[-1]).sum(0) / X.shape[0]
        if missing := set(flat) - set(grad):
            raise ValueError(f"DUST trainer handles Dense layers only; no estimate for {sorted(missing)}")
        return traverse_util.unflatten_dict(grad), losses.mean(1)


# Run labels: the optimizer config's _target_ names the algorithm (base/run_layout.yaml).
EGGROLL = EggrollTrainer
DUST = DustTrainer
