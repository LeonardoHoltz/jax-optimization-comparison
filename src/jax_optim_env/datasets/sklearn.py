from pathlib import Path

import numpy as np
from sklearn import datasets as sk_datasets
from torch.utils.data import DataLoader

from .default import DataModule, XYDataset

DATASET_LOADING_MAP = {
    "california_housing": lambda: sk_datasets.fetch_california_housing(),
    "diabetes": lambda: sk_datasets.load_diabetes(),
    "iris": lambda: sk_datasets.load_iris(),
    "breast_cancer": lambda: sk_datasets.load_breast_cancer(),
}

class SklearnDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info(f"Sklearn Dataset {self.name} loaded. Shape info:")
        self.logger.info(f"Input data: {self.X.shape}")
        self.logger.info(f"Target data: {self.y.shape}")

    def get_train_ds(self):
        return XYDataset(self.train_data["X"], self.train_data["y"])

    def get_val_ds(self):
        return XYDataset(self.val_data["X"], self.val_data["y"])

    def get_train_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.jax_collate_fn,
            drop_last=True,
        )

    def get_val_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.jax_collate_fn,
        )

    def resolve(self):
        ds_dir = self.root / self.name
        X_path = ds_dir / "X.npy"
        y_path = ds_dir / "y.npy"

        if X_path.exists() and y_path.exists():
            self.X = np.load(X_path).astype(np.float32)
            self.y = np.load(y_path)
        else:
            self.X, self.y = self._load_dataset(self.name)
            self.X = self.X.astype(np.float32)

            ds_dir.mkdir(parents=True, exist_ok=True)
            np.save(X_path, self.X)
            np.save(y_path, self.y)

        if self.num_features is not None and self.X.shape[1] != self.num_features:
            self.X = self._align_features(self.X, self.num_features)

        if self.num_classes > 0:
            self.y = self.y.astype(np.int32)
        else:
            self.y = self.y.astype(np.float32)

    def _load_dataset(self, name):
        if name in DATASET_LOADING_MAP:
            data = DATASET_LOADING_MAP[name]()
            return data.data, data.target

        raise ValueError(f"Unsupported sklearn dataset: {name}")

    def _synthetic_dataset(self, n_samples, n_features, classification):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
        if classification:
            y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
        else:
            y = (
                2.0 * X[:, 0]
                - 1.5 * X[:, 1]
                + np.sin(X[:, 2])
                + 0.1 * rng.normal(size=n_samples)
            ).astype(np.float32)
        return X, y

    def _align_features(self, X, n_features):
        if X.shape[1] < n_features:
            padding = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=X.dtype)
            return np.concatenate([X, padding], axis=1)
        return X[:, :n_features]
