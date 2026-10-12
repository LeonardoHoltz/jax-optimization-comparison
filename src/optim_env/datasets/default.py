from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from pathlib import Path
import logging
import numpy as np
import jax.numpy as jnp
import torch

class DataModule:
    """Dataset loading and splitting shared by the JAX and PyTorch paths. Everything is
    numpy until the collate step, which `framework` picks."""

    def __init__(self, name, num_features, num_classes, root, val_split, batch_size, seed, normalize=False, framework="jax"):
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
        if framework not in ("jax", "torch"):
            raise ValueError(f"Unknown framework '{framework}', expected 'jax' or 'torch'")
        self.framework = framework
        self.collate_fn = self.jax_collate_fn if framework == "jax" else self.torch_collate_fn
        self.resolve()
        self.split()

    def resolve(self):
        """Subclasses load data here: set train_data/val_data/test_data for fixed splits,
        or self.X/self.y for a pool that `split()` cuts into train/val."""
        raise NotImplementedError

    def _check_features(self, X):
        """Raise when the config's num_features disagrees with the data."""
        if self.num_features is not None and X.shape[1] != self.num_features:
            raise ValueError(
                f"Dataset '{self.name}' has {X.shape[1]} features but the config says "
                f"num_features={self.num_features}"
            )

    def _standardize(self, arr):
        """Zero mean, unit variance per column."""
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std, mean, std

    def split(self):
        """Random train/val split of self.X/self.y, unless resolve() set fixed splits."""
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

    def get_train_loader(self, dataset):
        # Seeded shuffle, so a rerun sees the same batches in the same order.
        order = RandomSampler(dataset, generator=torch.Generator().manual_seed(self.seed))
        return self._loader(dataset, order, drop_last=True)

    def get_val_loader(self, dataset):
        return self._loader(dataset, SequentialSampler(dataset), drop_last=False)

    def _loader(self, dataset, order, drop_last):
        # The sampler yields whole index lists, so each batch is one numpy fancy-index
        # instead of batch_size separate __getitem__ calls.
        return DataLoader(dataset, sampler=BatchSampler(order, self.batch_size, drop_last),
                          batch_size=None, num_workers=0, collate_fn=self.collate_fn)

    def jax_collate_fn(self, batch):
        X, y = batch
        return jnp.asarray(X), jnp.asarray(y)

    def torch_collate_fn(self, batch):
        X, y = batch
        return torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(np.ascontiguousarray(y))

class XYDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]