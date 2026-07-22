from torch.utils.data import DataLoader
import numpy as np
from .default import DataModule, XYDataset

class PMLBDataModule(DataModule):
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

        if not (X_path.exists() and y_path.exists()):
            raise FileNotFoundError(
                f"PMLB dataset '{self.name}' was not found at {ds_dir}. Download it first with:\n"
                f"  pixi run python -m jax_optim_env.prepare_dataset.pmlb --name {self.name} --root {self.root}"
            )

        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path)

        # Cast y based on task type
        if self.num_classes > 0:
            self.y = self.y.astype(np.int32)
        else:
            self.y = self.y.astype(np.float32)
    
