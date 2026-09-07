"""Comparing the hidden representations of two independently-trained networks.

The reason this needs dedicated machinery: hidden units have no canonical identity.
Two networks computing nearly the same function can have their units in any order, and
with ReLU you can scale a unit's incoming weights by a>0 and its outgoing weights by 1/a
for free. So comparing "unit i of A" against "unit i of B" is meaningless, and raw
activation magnitudes aren't comparable either.

Two ways around that, both provided here:
  - linear_cka: a similarity score for whole layers, invariant to orthogonal transforms
    (which includes permutation) and to isotropic scaling. Kornblith et al. 2019,
    "Similarity of Neural Network Representations Revisited".
  - match_neurons: undoes the permutation explicitly by finding the assignment of A's
    units to B's that maximizes total correlation, so individual units can be compared.
"""

import numpy as np


def _center_columns(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear CKA between two (n_samples, n_units) activation matrices. 1.0 = identical
    representation up to rotation/scaling; 0.0 = unrelated. Unit counts may differ."""
    A, B = _center_columns(np.asarray(A, dtype=np.float64)), _center_columns(np.asarray(B, dtype=np.float64))
    # ||A^T B||_F^2 / (||A^T A||_F ||B^T B||_F), computed via the feature-space form so
    # cost scales with n_units rather than n_samples^2.
    cross = np.linalg.norm(A.T @ B, ord="fro") ** 2
    norm_a = np.linalg.norm(A.T @ A, ord="fro")
    norm_b = np.linalg.norm(B.T @ B, ord="fro")
    if norm_a == 0 or norm_b == 0:  # a fully dead layer has no representation to compare
        return float("nan")
    return float(cross / (norm_a * norm_b))


def match_neurons(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one matching of A's units to B's by absolute correlation.

    Returns (pairs, correlations): `pairs[k] = (i, j)` matches unit i of A to unit j of B,
    and `correlations[k]` is their |Pearson r|. High values across the board mean the two
    networks learned the same features in a different order; a long tail near zero means
    genuinely different features.
    """
    from scipy.optimize import linear_sum_assignment

    corr = _abs_correlation(A, B)
    rows, cols = linear_sum_assignment(-corr)  # maximize total correlation
    return np.stack([rows, cols], axis=1), corr[rows, cols]


def _abs_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n_units_a, n_units_b) matrix of |Pearson r| between every pair of units."""
    A, B = _center_columns(np.asarray(A, dtype=np.float64)), _center_columns(np.asarray(B, dtype=np.float64))
    # Dead/constant units have zero variance; guard so they score 0 rather than NaN.
    sd_a = np.linalg.norm(A, axis=0)
    sd_b = np.linalg.norm(B, axis=0)
    sd_a[sd_a == 0] = np.inf
    sd_b[sd_b == 0] = np.inf
    return np.abs((A / sd_a).T @ (B / sd_b))


def dead_unit_fraction(A: np.ndarray, tol: float = 0.0) -> float:
    """Fraction of units that never activate on any sample -- a ReLU unit stuck off
    contributes nothing to the function and represents wasted capacity."""
    return float((np.asarray(A).max(axis=0) <= tol).mean())


def effective_rank(A: np.ndarray) -> float:
    """Shannon-entropy effective rank of the representation: how many dimensions the layer
    genuinely uses. Much lower than the unit count means the layer is collapsed onto a
    small subspace regardless of how many units it nominally has."""
    s = np.linalg.svd(_center_columns(np.asarray(A, dtype=np.float64)), compute_uv=False)
    total = s.sum()
    if total == 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def compare_layers(acts_a: dict[str, np.ndarray], acts_b: dict[str, np.ndarray]) -> list[dict]:
    """Layer-by-layer summary for two runs' activations (matching layer names)."""
    rows = []
    for name in acts_a:
        if name not in acts_b:
            continue
        A, B = acts_a[name], acts_b[name]
        _, corrs = match_neurons(A, B)
        rows.append({
            "layer": name,
            "units_a": A.shape[1],
            "units_b": B.shape[1],
            "cka": linear_cka(A, B),
            "matched_corr_mean": float(np.mean(corrs)),
            "matched_corr_median": float(np.median(corrs)),
            "dead_frac_a": dead_unit_fraction(A),
            "dead_frac_b": dead_unit_fraction(B),
            "eff_rank_a": effective_rank(A),
            "eff_rank_b": effective_rank(B),
        })
    return rows
