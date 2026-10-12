import flax.linen as nn
import jax.numpy as jnp


class FastText(nn.Module):
    """fastText (Joulin et al. 2016) without n-grams: average the word embeddings, then
    one linear layer. Token id 0 is padding and is left out of the average."""

    vocab_size: int
    dout: int
    embed_dim: int = 10

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        ids = x.astype(jnp.int32)
        mask = (ids > 0)[..., None]
        emb = nn.Embed(self.vocab_size, self.embed_dim)(ids) * mask
        mean = emb.sum(1) / jnp.maximum(mask.sum(1), 1)
        return nn.Dense(self.dout)(mean)


class KimCNN(nn.Module):
    """CNN-rand from Kim (2014): random 300-d embeddings trained with the model, filter
    widths 3/4/5 with 100 maps each, max-over-time pooling, dropout 0.5. The paper's
    max-norm weight constraint is left out."""

    vocab_size: int
    dout: int
    embed_dim: int = 300
    widths: tuple = (3, 4, 5)
    features: int = 100
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        ids = x.astype(jnp.int32)
        emb = nn.Embed(self.vocab_size, self.embed_dim)(ids) * (ids > 0)[..., None]
        pooled = []
        for w in self.widths:
            h = nn.relu(nn.Conv(self.features, (w,), padding="VALID")(emb))
            # Windows that start on padding hold no words; keep them out of the max, so
            # the output doesn't depend on how much padding follows the text.
            starts_on_word = (ids[:, : h.shape[1]] > 0)[..., None]
            pooled.append(jnp.where(starts_on_word, h, 0.0).max(axis=1))
        h = nn.Dropout(self.dropout, deterministic=deterministic)(jnp.concatenate(pooled, -1))
        return nn.Dense(self.dout)(h)
