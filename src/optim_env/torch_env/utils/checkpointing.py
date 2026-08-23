from pathlib import Path

import torch


def save_checkpoint(state_dict, path):
    """Saves a model's state_dict to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint(path, map_location="cpu"):
    """Loads a model's state_dict from a file. Unlike the Flax checkpointing utility,
    no template object is needed -- torch.load reconstructs the state_dict directly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    return torch.load(path, map_location=map_location)
