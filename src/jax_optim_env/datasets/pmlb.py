from torch.utils.data import DataLoader
import numpy as np
from pmlb import fetch_data
from .default import DatasetManager, XYDataset

class PMLBDatasetManager(DatasetManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info(f"PLMB Dataset {self.name} loaded. Shape info:")
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
            collate_fn=self.jax_collate_fn
        )
    
    def resolve(self):
        ds_dir = self.root / self.name
        X_path = ds_dir / "X.npy"
        y_path = ds_dir / "y.npy"

        if X_path.exists() and y_path.exists():
            # Already downloaded
            self.X = np.load(X_path).astype(np.float32)
            self.y = np.load(y_path)
        else:
            # Downloads and saves
            self.X, self.y = fetch_data(
                dataset_name=self.name,
                return_X_y=True,
                dropna=True,
            )
            self.X = self.X.astype(np.float32)
            
            ds_dir.mkdir(parents=True, exist_ok=True)
            np.save(X_path, self.X)
            np.save(y_path, self.y)

        # Cast y based on task type
        if self.num_classes > 0:
            self.y = self.y.astype(np.int32)
        else:
            self.y = self.y.astype(np.float32)
    
