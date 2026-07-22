from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

try:
    import kagglehub
except ImportError:  # pragma: no cover - exercised only when package is absent
    kagglehub = None

from .default import DataModule, XYDataset


class AdultIncomeDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info(f"Adult Income dataset {self.name} loaded. Shape info:")
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
        ds_dir.mkdir(parents=True, exist_ok=True)

        X_path = ds_dir / "X.npy"
        y_path = ds_dir / "y.npy"

        if X_path.exists() and y_path.exists():
            self.X = np.load(X_path).astype(np.float32)
            self.y = np.load(y_path)
            return

        csv_path = self._resolve_csv_path(ds_dir)
        if csv_path is None:
            downloaded_dir = self._download_dataset()
            csv_path = self._find_csv(downloaded_dir)
            if csv_path is None:
                raise FileNotFoundError("No CSV file was found in the downloaded Kaggle dataset")
            destination_path = ds_dir / csv_path.name
            destination_path.write_bytes(csv_path.read_bytes())
            csv_path = destination_path

        self.X, self.y = self._load_csv(csv_path)
        self.X = self.X.astype(np.float32)

        np.save(X_path, self.X)
        np.save(y_path, self.y)

        if self.num_features is not None and self.X.shape[1] != self.num_features:
            self.X = self._align_features(self.X, self.num_features)

        if self.num_classes > 0:
            self.y = self.y.astype(np.int32)
        else:
            self.y = self.y.astype(np.float32)

    def _resolve_csv_path(self, ds_dir):
        candidate_paths = [
            ds_dir / "adult.csv",
            self.root / "adult.csv",
            ds_dir / "adult.data.csv",
            self.root / "adult.data.csv",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate
        return None

    def _download_dataset(self):
        if kagglehub is None:
            raise ImportError("kagglehub is required to download the Adult Income dataset")

        download_path = kagglehub.dataset_download("uciml/adult-census-income")
        return Path(download_path)

    def _find_csv(self, root_dir):
        csv_files = sorted(root_dir.rglob("*.csv"))
        if not csv_files:
            return None
        return csv_files[0]

    def _load_csv(self, csv_path):
        frame = pd.read_csv(csv_path, na_values="?", keep_default_na=True, skipinitialspace=True)
        if "income" not in frame.columns:
            raise ValueError("Expected an 'income' column in the Adult Income dataset")

        target_series = frame["income"].astype(str).str.strip()
        y = (target_series.str.startswith(">")).astype(np.int32)

        features = frame.drop(columns=["income"]).copy()
        features = features.fillna("Unknown")

        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in features.columns if col not in numeric_cols]

        for col in numeric_cols:
            features[col] = features[col].astype(np.float32)

        encoded = pd.get_dummies(features, columns=categorical_cols, dtype=np.float32)
        X = encoded.to_numpy(dtype=np.float32)
        return X, y

    def _align_features(self, X, n_features):
        if X.shape[1] < n_features:
            padding = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=X.dtype)
            return np.concatenate([X, padding], axis=1)
        return X[:, :n_features]
