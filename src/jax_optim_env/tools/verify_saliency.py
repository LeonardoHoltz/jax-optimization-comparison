import jax
import jax.numpy as jnp
import os
from jax_optim_env.utils.explainability import compute_saliency_map
from lerna.utils import instantiate
import lerna
from omegaconf import DictConfig
from jax_optim_env.utils.checkpointing import load_checkpoint
from flax.training import train_state

@lerna.main(version_base=None, config_path="../configs", config_name="sgd_exp")
def verify_saliency(cfg: DictConfig) -> None:
    print("=> Loading dataset and model...")
    # 1. Instantiate dataset and model directly
    dataset_manager = instantiate(cfg.dataset)
    train_ds = dataset_manager.get_train_ds()
    model = instantiate(cfg.model)
    
    # 2. Get initial state structure for loading
    dummy_x = jnp.zeros((1, train_ds.X.shape[1]))
    init_rng = jax.random.PRNGKey(cfg.seed)
    params = model.init(init_rng, dummy_x)
    
    # For SGD, we need to load into a TrainState
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
    
    # 5. Compute saliency
    print("=> Computing saliency map...")
    saliency = compute_saliency_map(model.apply, loaded_params, sample_x, target_class=sample_y)
    
    print(f"Saliency map shape: {saliency.shape}")
    print(f"Saliency (first 5 features): {saliency[:5]}")
    
    # Simple check: saliency should not be all zeros
    assert not jnp.all(saliency == 0), "Saliency map is all zeros!"
    print("Verification successful: Saliency map computed and is non-zero.")

if __name__ == "__main__":
    verify_saliency()
