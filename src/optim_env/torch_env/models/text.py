import torch
import torch.nn as nn
import torch.nn.functional as F

# Twins of jax_env.models.text for utils.torch_export; layer names match Flax's.


class FastText(nn.Module):
    def __init__(self, vocab_size: int, dout: int, embed_dim: int = 10):
        super().__init__()
        self.Embed_0 = nn.Embedding(vocab_size, embed_dim)
        self.Dense_0 = nn.Linear(embed_dim, dout)

    def forward(self, x):
        ids = x.long()
        mask = (ids > 0).unsqueeze(-1)
        mean = (self.Embed_0(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.Dense_0(mean)


class KimCNN(nn.Module):
    def __init__(self, vocab_size: int, dout: int, embed_dim: int = 300, widths: tuple = (3, 4, 5),
                 features: int = 100, dropout: float = 0.5):
        super().__init__()
        self.Embed_0 = nn.Embedding(vocab_size, embed_dim)
        self.widths = tuple(widths)
        for i, w in enumerate(self.widths):
            setattr(self, f"Conv_{i}", nn.Conv1d(embed_dim, features, w))
        self.dropout = nn.Dropout(dropout)
        self.Dense_0 = nn.Linear(features * len(self.widths), dout)

    def forward(self, x):
        ids = x.long()
        emb = (self.Embed_0(ids) * (ids > 0).unsqueeze(-1)).transpose(1, 2)
        pooled = []
        for i in range(len(self.widths)):
            h = F.relu(getattr(self, f"Conv_{i}")(emb))
            # Same padding mask as the Flax model: windows starting on padding are dropped.
            starts_on_word = (ids[:, : h.shape[2]] > 0).unsqueeze(1)
            pooled.append(torch.where(starts_on_word, h, 0.0).amax(2))
        return self.Dense_0(self.dropout(torch.cat(pooled, 1)))
