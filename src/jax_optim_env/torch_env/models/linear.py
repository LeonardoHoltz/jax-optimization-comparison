from typing import Sequence

import torch.nn as nn


class Linear(nn.Module):
    # din is required up front since PyTorch doesn't lazily infer input width from the
    # first call.
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.dense = nn.Linear(din, dout)

    def forward(self, x):
        return self.dense(x)


class MLP(nn.Module):
    # Dropout is toggled via .train()/.eval() on nn.Dropout, the idiomatic PyTorch way.
    def __init__(self, din: int, hidden_dims: Sequence[int], dout: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = din
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
