import jax.numpy as jnp
import numpy as np
import os
from jax_optim_env.utils.checkpointing import save_checkpoint, load_checkpoint

def test_checkpointing():
    state = {
        "params": {
            "Dense_0": {"kernel": jnp.array([1.0, 2.0]), "bias": jnp.array([0.0])},
            "Dense_1": {"kernel": jnp.array([3.0, 4.0]), "bias": jnp.array([1.0])}
        },
        "step": 10
    }
    path = "/tmp/test_checkpoint.msgpack"
    
    save_checkpoint(state, path)
    assert os.path.exists(path)
    
    loaded_state = load_checkpoint(state, path)
    
    # Assert nested structure matches
    assert np.allclose(loaded_state["params"]["Dense_0"]["kernel"], state["params"]["Dense_0"]["kernel"])
    assert np.allclose(loaded_state["params"]["Dense_1"]["bias"], state["params"]["Dense_1"]["bias"])
    assert loaded_state["step"] == state["step"]
    
    os.remove(path)
