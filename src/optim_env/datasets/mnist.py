import numpy as np

from .default import DataModule

class MNISTDataModule(DataModule):
    """MNIST, or Fashion-MNIST (name: fashion_mnist), from an mnist.npz bundle.

    Official 10K test set; validation is carved from the 60K training images. Images are
    flattened to 784 columns, and the CNN models reshape them back.
    """

    def __init__(self, *args, val_size: int = 10000, **kwargs):
        self.val_size = val_size
        super().__init__(*args, **kwargs)
        self.logger.info(f"MNIST loaded. Train: {self.train_data['X'].shape}, "
                         f"Val: {self.val_data['X'].shape}, Test: {self.test_data['X'].shape}")

    def resolve(self):
        npz_path = self.root / self.name / "mnist.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{self.name} was not found at {npz_path}. Download it first with:\n"
                f"  pixi run python -m optim_env.prepare_dataset.images --name {self.name} --root {self.root}"
            )

        with np.load(npz_path) as d:
            x_train, y_train = d["x_train"], d["y_train"]
            x_test, y_test = d["x_test"], d["y_test"]

        # Scale to [0, 1] rather than per-feature standardization: border pixels are
        # constant-zero across the whole dataset, so standardizing them is meaningless,
        # and [0,1] is the convention these architectures are tuned for.
        def prep(x):
            return (x.reshape(len(x), -1).astype(np.float32) / 255.0)

        x_train, x_test = prep(x_train), prep(x_test)
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

        # Seeded shuffle before carving off validation, so the split doesn't depend on
        # the file's own ordering.
        order = np.random.default_rng(self.seed).permutation(len(x_train))
        x_train, y_train = x_train[order], y_train[order]
        n_val = self.val_size

        self.train_data = {"X": x_train[n_val:], "y": y_train[n_val:]}
        self.val_data = {"X": x_train[:n_val], "y": y_train[:n_val]}
        self.test_data = {"X": x_test, "y": y_test}

        # Pooled view, for introspection/logging parity with the other DataModules.
        self.X = np.concatenate([self.train_data["X"], self.val_data["X"], self.test_data["X"]])
        self.y = np.concatenate([self.train_data["y"], self.val_data["y"], self.test_data["y"]])
