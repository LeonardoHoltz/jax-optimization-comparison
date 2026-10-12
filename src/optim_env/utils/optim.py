import optax


def adamw_clipped(learning_rate, weight_decay, max_norm=1.0):
    """AdamW with gradient-norm clipping, the TabM paper's training setup. A named
    function so runs get the label "adamw_clipped" instead of "chain"."""
    return optax.chain(optax.clip_by_global_norm(max_norm), optax.adamw(learning_rate, weight_decay=weight_decay))
