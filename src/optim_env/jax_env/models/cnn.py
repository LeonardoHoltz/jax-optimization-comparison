from typing import Sequence

import flax.linen as nn


class CNN(nn.Module):
    """Small convolutional classifier.

    Takes flattened input and reshapes to `input_shape` internally, so the shared
    tabular-style data pipeline (which hands out 2-D X) needs no special case for images.

    Kept deliberately small: these models are also optimized by evolutionary strategies,
    which search the raw parameter vector and degrade badly as its dimension grows. The
    default (16, 32) stack is ~20K parameters, roughly a tenth of the tabular MLP.
    """

    input_shape: Sequence[int]  # e.g. (28, 28, 1)
    features: Sequence[int]  # channels per conv block
    dout: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = x.reshape((x.shape[0], *self.input_shape))

        for channels in self.features:
            x = nn.Conv(features=channels, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        if self.dropout > 0:
            x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        return nn.Dense(self.dout)(x)
