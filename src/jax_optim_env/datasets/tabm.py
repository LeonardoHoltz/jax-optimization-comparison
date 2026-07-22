import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .default import DataModule, XYDataset

SPLITS = ("train", "val", "test")


class TabMDataModule(DataModule):
    """Loads datasets shaped like https://huggingface.co/datasets/rototoHF/tabm-data:
    per-split X_num_*.npy / X_bin_*.npy / X_cat_*.npy / Y_*.npy files plus an info.json.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info(f"TabM Dataset {self.name} loaded. Shape info:")
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
            if not self._raw_files_present(ds_dir):
                raise FileNotFoundError(
                    f"TabM dataset '{self.name}' was not found at {ds_dir}. Download it first with:\n"
                    f"  pixi run python -m jax_optim_env.prepare_dataset.huggingface "
                    f"--repo-id rototoHF/tabm-data --name {self.name} --root {self.root}"
                )

            self.X, self.y = self._load_raw(ds_dir)
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

    def _raw_files_present(self, ds_dir):
        return any((ds_dir / f"Y_{split}.npy").exists() for split in SPLITS)

    def _load_raw(self, ds_dir):
        num_parts, bin_parts, cat_parts, y_parts = [], [], [], []

        for split in SPLITS:
            y_path = ds_dir / f"Y_{split}.npy"
            if not y_path.exists():
                continue
            y_parts.append(np.load(y_path))

            for prefix, parts in (("X_num", num_parts), ("X_bin", bin_parts), ("X_cat", cat_parts)):
                path = ds_dir / f"{prefix}_{split}.npy"
                if path.exists():
                    parts.append(np.load(path, allow_pickle=True))

        y = np.concatenate(y_parts)

        feature_blocks = []
        if num_parts:
            feature_blocks.append(np.concatenate(num_parts).astype(np.float32))
        if bin_parts:
            feature_blocks.append(np.concatenate(bin_parts).astype(np.float32))
        if cat_parts:
            cat = np.concatenate(cat_parts)
            cat_frame = pd.DataFrame(cat, columns=[f"cat_{i}" for i in range(cat.shape[1])])
            encoded = pd.get_dummies(cat_frame, columns=cat_frame.columns, dtype=np.float32)
            feature_blocks.append(encoded.to_numpy(dtype=np.float32))

        if not feature_blocks:
            raise ValueError(f"No feature files (X_num/X_bin/X_cat) found for TabM dataset at {ds_dir}")

        X = np.concatenate(feature_blocks, axis=1)
        return X, y

    def _align_features(self, X, n_features):
        if X.shape[1] < n_features:
            padding = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=X.dtype)
            return np.concatenate([X, padding], axis=1)
        return X[:, :n_features]
