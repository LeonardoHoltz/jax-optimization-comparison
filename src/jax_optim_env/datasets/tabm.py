import numpy as np
import pandas as pd

from .default import DataModule

SPLITS = ("train", "val", "test")


class TabMDataModule(DataModule):
    """Loads datasets shaped like https://huggingface.co/datasets/rototoHF/tabm-data:
    per-split X_num_*.npy / X_bin_*.npy / X_cat_*.npy / Y_*.npy files plus an info.json.
    Every split present on disk (train, val, and test when available) is exposed as-is
    via train_data/val_data/test_data -- no random re-splitting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info(f"TabM Dataset {self.name} loaded. Shape info:")
        self.logger.info(f"Train: {self.train_data['X'].shape}, Val: {self.val_data['X'].shape}")
        if self.test_data is not None:
            self.logger.info(f"Test: {self.test_data['X'].shape}")

    def resolve(self):
        ds_dir = self.root / self.name
        if not self._raw_files_present(ds_dir):
            raise FileNotFoundError(
                f"TabM dataset '{self.name}' was not found at {ds_dir}. Download it first with:\n"
                f"  pixi run python -m jax_optim_env.prepare_dataset.huggingface "
                f"--repo-id rototoHF/tabm-data --name {self.name} --root {self.root}"
            )

        per_split = self._load_raw(ds_dir)

        for split, (X, y) in per_split.items():
            X = X.astype(np.float32)
            if self.num_features is not None and X.shape[1] != self.num_features:
                X = self._align_features(X, self.num_features)
            y = y.astype(np.int32) if self.num_classes > 0 else y.astype(np.float32)
            per_split[split] = (X, y)

        if self.normalize:
            # Fit standardization stats on the combined pool (matching the previous
            # behavior), then apply the same stats to every split independently.
            pooled_X = np.concatenate([X for X, _ in per_split.values()])
            _, self._x_mean, self._x_std = self._standardize(pooled_X)
            if self.num_classes == 0:
                pooled_y = np.concatenate([y for _, y in per_split.values()]).reshape(-1, 1)
                _, self._y_mean, self._y_std = self._standardize(pooled_y)

            for split, (X, y) in per_split.items():
                X = (X - self._x_mean) / self._x_std
                if self.num_classes == 0:
                    y = (((y.reshape(-1, 1) - self._y_mean) / self._y_std).reshape(-1)).astype(np.float32)
                per_split[split] = (X.astype(np.float32), y)

        self.train_data = {"X": per_split["train"][0], "y": per_split["train"][1]}
        self.val_data = {"X": per_split["val"][0], "y": per_split["val"][1]} if "val" in per_split else None
        self.test_data = {"X": per_split["test"][0], "y": per_split["test"][1]} if "test" in per_split else None

        # Kept only for introspection/logging -- the pooled view of every split found.
        self.X = np.concatenate([X for X, _ in per_split.values()])
        self.y = np.concatenate([y for _, y in per_split.values()])

    def _raw_files_present(self, ds_dir):
        return any((ds_dir / f"Y_{split}.npy").exists() for split in SPLITS)

    def _load_raw(self, ds_dir):
        """Assembles features per split, one-hot encoding categorical columns ONCE across
        all splits combined (so the resulting columns line up consistently across
        train/val/test) before slicing back out by each split's own row count."""
        num_parts, bin_parts, cat_parts, y_parts = [], [], [], []
        splits_present, sizes = [], []

        for split in SPLITS:
            y_path = ds_dir / f"Y_{split}.npy"
            if not y_path.exists():
                continue
            y = np.load(y_path)
            splits_present.append(split)
            sizes.append(len(y))
            y_parts.append(y)

            for prefix, parts in (("X_num", num_parts), ("X_bin", bin_parts), ("X_cat", cat_parts)):
                path = ds_dir / f"{prefix}_{split}.npy"
                if path.exists():
                    parts.append(np.load(path, allow_pickle=True))

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
        y = np.concatenate(y_parts)

        per_split = {}
        offset = 0
        for split, n in zip(splits_present, sizes):
            per_split[split] = (X[offset:offset + n], y[offset:offset + n])
            offset += n
        return per_split

    def _align_features(self, X, n_features):
        if X.shape[1] < n_features:
            padding = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=X.dtype)
            return np.concatenate([X, padding], axis=1)
        return X[:, :n_features]
