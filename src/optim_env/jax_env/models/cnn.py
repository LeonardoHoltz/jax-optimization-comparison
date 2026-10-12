from typing import Sequence

import flax.linen as nn


class CNN(nn.Module):
    """Small CNN (~20K parameters with the default (16, 32) channels), kept small for
    evolution strategies. Takes flattened images and reshapes them to `input_shape`."""

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


class LeNet5(nn.Module):
    """LeNet-5 (LeCun et al. 1998), ~61.7K parameters, in its common modern form: ReLU and
    max pooling instead of the original tanh and subsampling layers."""

    dout: int = 10

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = x.reshape((x.shape[0], 28, 28, 1))
        x = nn.max_pool(nn.relu(nn.Conv(6, (5, 5), padding="SAME")(x)), (2, 2), strides=(2, 2))
        x = nn.max_pool(nn.relu(nn.Conv(16, (5, 5), padding="VALID")(x)), (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(120)(x))
        x = nn.relu(nn.Dense(84)(x))
        return nn.Dense(self.dout)(x)
