from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pathlib import Path
import logging
import numpy as np
import jax.numpy as jnp

class DataModule:
    def __init__(self, name, num_features, num_classes, root, val_split, batch_size, seed, normalize=False):
        self.logger = logging.getLogger()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.num_features = num_features
        self.num_classes = num_classes
        self.val_split = val_split
        self.batch_size = batch_size
        self.seed = seed
        self.normalize = normalize
        self.resolve()
        self.split()

    def resolve(self):
        raise NotImplementedError

    def _standardize(self, arr):
        """Zero-mean, unit-variance standardization, matching the paper's practice of
        standardizing features/regression targets. Fit-on-everything approximation
        since this DataModule does not keep a held-out fit-only split."""
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std, mean, std
    
    def split(self):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            test_size=self.val_split,
            random_state=self.seed
        )
        self.train_data = {"X": X_train, "y": y_train}
        self.val_data = {"X": X_val, "y": y_val}
    
    def get_train_loader(self):
        raise NotImplementedError
    def get_val_loader(self):
        raise NotImplementedError

    def collate_fn(self, batch):
        return batch
    
    def jax_collate_fn(self, batch):
        xs, ys = zip(*batch)
        X = jnp.asarray(np.stack(xs))
        y = jnp.asarray(np.stack(ys))
        return X, y
    
class XYDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]