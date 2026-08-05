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
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.resolve()
        self.split()

    def resolve(self):
        """Subclasses populate data here. Either set self.X/self.y for a single pool (in
        which case `split()` below creates a random train/val split, with no test data),
        or -- for datasets shipped with genuine, fixed splits -- set self.train_data/
        self.val_data/self.test_data directly, in which case `split()` is a no-op."""
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
        """Fallback only: if resolve() already populated train_data/val_data directly
        (datasets with genuine, fixed splits), this is a no-op -- their split boundaries
        are left exactly as resolve() set them, including test_data if present.
        Otherwise, falls back to a random train/val split of self.X/self.y; no test
        partition exists in that case."""
        if self.train_data is not None and self.val_data is not None:
            return
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            test_size=self.val_split,
            random_state=self.seed
        )
        self.train_data = {"X": X_train, "y": y_train}
        self.val_data = {"X": X_val, "y": y_val}

    def get_train_ds(self):
        return XYDataset(self.train_data["X"], self.train_data["y"])

    def get_val_ds(self):
        return XYDataset(self.val_data["X"], self.val_data["y"])

    def get_test_ds(self):
        """Held-out test split, distinct from the train/val split above. Only datasets
        distributed with a genuine train/val/test partition (e.g. TabMDataModule) can
        provide this -- others raise, since there's nothing honest to return."""
        if self.test_data is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} '{self.name}' does not provide a held-out test split."
            )
        return XYDataset(self.test_data["X"], self.test_data["y"])

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