from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F

# Twins of jax_env.models.cnn for utils.torch_export. Layers carry Flax's auto names
# (Conv_0, Dense_0, ...) so weights copy over by name.


class CNN(nn.Module):
    def __init__(self, input_shape: Sequence[int], features: Sequence[int], dout: int, dropout: float = 0.0):
        super().__init__()
        self.input_shape = tuple(input_shape)
        h, w, c = self.input_shape
        for i, channels in enumerate(features):
            setattr(self, f"Conv_{i}", nn.Conv2d(c, channels, 3, padding="same"))
            h, w, c = h // 2, w // 2, channels
        self.n_convs = len(features)
        self.dropout = nn.Dropout(dropout)
        self.Dense_0 = nn.Linear(h * w * c, dout)

    def forward(self, x):
        x = x.reshape(-1, *self.input_shape).permute(0, 3, 1, 2)
        for i in range(self.n_convs):
            x = F.max_pool2d(F.relu(getattr(self, f"Conv_{i}")(x)), 2)
        # Flatten channels-last, as Flax does, so Dense_0's rows line up.
        return self.Dense_0(self.dropout(x.permute(0, 2, 3, 1).flatten(1)))


class LeNet5(nn.Module):
    def __init__(self, dout: int = 10):
        super().__init__()
        self.Conv_0 = nn.Conv2d(1, 6, 5, padding="same")
        self.Conv_1 = nn.Conv2d(6, 16, 5)
        self.Dense_0 = nn.Linear(5 * 5 * 16, 120)
        self.Dense_1 = nn.Linear(120, 84)
        self.Dense_2 = nn.Linear(84, dout)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.Conv_0(x)), 2)
        x = F.max_pool2d(F.relu(self.Conv_1(x)), 2)
        x = x.permute(0, 2, 3, 1).flatten(1)
        return self.Dense_2(F.relu(self.Dense_1(F.relu(self.Dense_0(x)))))
