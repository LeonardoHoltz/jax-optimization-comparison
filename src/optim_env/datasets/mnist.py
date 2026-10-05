import numpy as np

from .default import DataModule

MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


class MNISTDataModule(DataModule):
    """MNIST from the standard mnist.npz bundle.

    Keeps the official 10k test set as the test split and carves a validation split off
    the 60k training set, so the test set stays untouched by model selection.

    Images are flattened to 784 features to satisfy the shared DataModule/loader contract
    (everything downstream, including the collate functions, assumes 2-D X). Convolutional
    models reshape back to (28, 28, 1) internally from their `input_shape` config, which
    keeps the image layout a property of the model rather than of the data pipeline.
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
                f"MNIST was not found at {npz_path}. Download it first with:\n"
                f"  curl -sL --create-dirs -o {npz_path} {MIRROR}"
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
