from pathlib import Path

import numpy as np
from sklearn import datasets as sk_datasets

from .default import DataModule

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

        self._check_features(self.X)

        if self.num_classes > 0:
            self.y = self.y.astype(np.int32)
        else:
            self.y = self.y.astype(np.float32)

        if self.normalize:
            self.X, self._x_mean, self._x_std = self._standardize(self.X)
            if self.num_classes == 0:
                y, self._y_mean, self._y_std = self._standardize(self.y.reshape(-1, 1))
                self.y = y.reshape(-1).astype(np.float32)

    def _load_dataset(self, name):
        if name in DATASET_LOADING_MAP:
            data = DATASET_LOADING_MAP[name]()
            return data.data, data.target

        raise ValueError(f"Unsupported sklearn dataset: {name}")

