from torch.utils.data import DataLoader
import numpy as np
from pmlb import fetch_data
from .default import DatasetManager, XYDataset

class PMLBDatasetManager(DatasetManager):
    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)
        self.logger.info(f"PLMB Dataset {dataset_cfg.name} loaded. Shape info:")
        self.logger.info(f"Input data: {self.X.shape}")
        self.logger.info(f"Target data: {self.y.shape}")
    
    def get_train_ds(self):
        return XYDataset(self.train_data["X"], self.train_data["y"])
    
    def get_val_ds(self):
        return XYDataset(self.val_data["X"], self.val_data["y"])

    def get_train_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.dataset_cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.jax_collate_fn
        )

    def get_val_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.dataset_cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.jax_collate_fn
        )
    
    def resolve(self):
        dataset_name = self.dataset_cfg.name
        ds_dir = self.root / dataset_name
        X_path = ds_dir / "X.npy"
        y_path = ds_dir / "y.npy"

        if X_path.exists() and y_path.exists():
            # Already downloaded
            self.X = np.load(X_path)
            self.y = np.load(y_path)
        else:
            # Downloads and saves
            X, y = fetch_data(
                dataset_name=dataset_name,
                return_X_y=True,
                dropna=True,
            )
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)

            ds_dir.mkdir(parents=True, exist_ok=True)
            np.save(X_path, X)
            np.save(y_path, y)
    
