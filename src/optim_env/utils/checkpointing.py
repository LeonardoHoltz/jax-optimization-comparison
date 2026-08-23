import os
from flax.serialization import to_bytes, from_bytes
from pathlib import Path

def save_checkpoint(state, path):
    """Saves the trainer state to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        f.write(to_bytes(state))

def load_checkpoint(state, path):
    """Loads the trainer state from a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
        
    with open(path, "rb") as f:
        return from_bytes(state, f.read())
