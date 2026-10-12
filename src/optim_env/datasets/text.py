import json

import numpy as np

from .default import DataModule

SPLITS = ("train", "val", "test")


class TextDataModule(DataModule):
    """Text classification data written by optim_env.prepare_dataset.text.

    features="tokens": X is (n, max_len) int32 token ids, for models with an embedding
    layer (fastText, Kim CNN).
    features="glove_mean": X is (n, 100), the mean GloVe vector of each text's known
    words, so a plain linear model can classify it.
    """

    def __init__(self, *args, features="tokens", vocab_size=None, **kwargs):
        if features not in ("tokens", "glove_mean"):
            raise ValueError(f"Unknown text features '{features}'")
        self.features = features
        self.vocab_size = vocab_size
        super().__init__(*args, **kwargs)

    def resolve(self):
        ds_dir = self.root / self.name
        if not (ds_dir / "meta.json").exists():
            raise FileNotFoundError(
                f"Text dataset '{self.name}' was not found at {ds_dir}. Prepare it first with:\n"
                f"  pixi run python -m optim_env.prepare_dataset.text --name {self.name} --root {self.root}"
            )
        meta = json.loads((ds_dir / "meta.json").read_text())
        if self.vocab_size is not None and self.vocab_size != meta["vocab_size"]:
            raise ValueError(f"Config vocab_size={self.vocab_size}, but the data has {meta['vocab_size']}")
        glove = np.load(ds_dir / "glove100.npy") if self.features == "glove_mean" else None

        for split in SPLITS:
            ids = np.load(ds_dir / f"{split}_ids.npy")
            X = ids if glove is None else self._glove_mean(ids, glove)
            self._check_features(X)
            setattr(self, f"{split}_data", {"X": X, "y": np.load(ds_dir / f"{split}_y.npy")})

        self.X = np.concatenate([getattr(self, f"{s}_data")["X"] for s in SPLITS])
        self.y = np.concatenate([getattr(self, f"{s}_data")["y"] for s in SPLITS])

    @staticmethod
    def _glove_mean(ids, glove, chunk=8192):
        # Chunked: all of AG News at once would be a 4 GB (n, max_len, 100) array.
        # Padding, unknown and non-GloVe words have zero rows, so they don't count.
        has_vector = np.abs(glove).sum(1) > 0
        out = np.empty((len(ids), glove.shape[1]), dtype=np.float32)
        for i in range(0, len(ids), chunk):
            block = ids[i:i + chunk]
            known = has_vector[block].sum(1, keepdims=True)
            out[i:i + chunk] = glove[block].sum(1) / np.maximum(known, 1)
        return out
