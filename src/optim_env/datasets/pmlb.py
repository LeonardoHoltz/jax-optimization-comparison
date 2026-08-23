import numpy as np
from .default import DataModule

class PMLBDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info(f"PLMB Dataset {self.name} loaded. Shape info:")
        self.logger.info(f"Input data: {self.X.shape}")
        self.logger.info(f"Target data: {self.y.shape}")

    def resolve(self):
        ds_dir = self.root / self.name
        X_path = ds_dir / "X.npy"
        y_path = ds_dir / "y.npy"

        if not (X_path.exists() and y_path.exists()):
            raise FileNotFoundError(
                f"PMLB dataset '{self.name}' was not found at {ds_dir}. Download it first with:\n"
                f"  pixi run python -m optim_env.prepare_dataset.pmlb_dataset --name {self.name} --root {self.root}"
            )

        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path)

        # Cast y based on task type
        if self.num_classes > 0:
            self.y = self.y.astype(np.int32)
        else:
            self.y = self.y.astype(np.float32)
    
