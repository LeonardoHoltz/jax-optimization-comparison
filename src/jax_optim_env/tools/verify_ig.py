import jax
import jax.numpy as jnp
import os
from jax_optim_env.utils.explainability import compute_integrated_gradients
from lerna.utils import instantiate
import lerna
from omegaconf import DictConfig
from jax_optim_env.utils.checkpointing import load_checkpoint
from flax.training import train_state

@lerna.main(version_base=None, config_path="../configs", config_name="sgd_exp")
def verify_ig(cfg: DictConfig) -> None:
    print("=> Loading dataset and model...")
    # 1. Instantiate dataset and model directly
    dataset_manager = instantiate(cfg.dataset)
    train_ds = dataset_manager.get_train_ds()
    model = instantiate(cfg.model)
    
    # 2. Get initial state structure for loading
    dummy_x = jnp.zeros((1, train_ds.X.shape[1]))
    init_rng = jax.random.PRNGKey(cfg.seed)
    params = model.init(init_rng, dummy_x)
    
    initial_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=instantiate(cfg.optimizer),
    )
    
    # 3. Load best model parameters
    checkpoint_path = "outputs/SGD_adam_mushroom/best_model.msgpack"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("=> Loading checkpoint...")
    state = load_checkpoint(initial_state, checkpoint_path)
    loaded_params = state.params
    
    # 4. Get a sample from the dataset
    train_loader = dataset_manager.get_train_loader(train_ds)
    X, y = next(iter(train_loader))
    sample_x = X[0]
    sample_y = y[0]
    
    print(f"Sample X shape: {sample_x.shape}, Label: {sample_y}")
    
    # 5. Compute IG
    print("=> Computing Integrated Gradients...")
    ig = compute_integrated_gradients(model.apply, loaded_params, sample_x, target_class=sample_y, steps=50)
    
    print(f"IG shape: {ig.shape}")
    print(f"IG (first 5 features): {ig[:5]}")
    
    # Basic IG property: sum(IG) should be approx (model(x) - model(baseline))
    baseline = jnp.zeros_like(sample_x)
    model_x = model.apply(loaded_params, sample_x)[sample_y]
    model_baseline = model.apply(loaded_params, baseline)[sample_y]
    
    ig_sum = jnp.sum(ig)
    expected_diff = model_x - model_baseline
    
    print(f"Sum of IG: {ig_sum:.4f}")
    print(f"Expected diff (f(x) - f(0)): {expected_diff:.4f}")
    
    # Threshold for check (can be loose due to approximation steps)
    assert jnp.abs(ig_sum - expected_diff) < 0.1, f"IG completeness property failed! Diff: {jnp.abs(ig_sum - expected_diff)}"
    print("Verification successful: IG completeness property holds.")

if __name__ == "__main__":
    verify_ig()
