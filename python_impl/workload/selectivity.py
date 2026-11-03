"""True selectivity of range queries, computed with numpy (no Python loops over rows).

Selectivity of a query = fraction of rows that satisfy it.
A row satisfies a query when  lo_i <= x_i <= hi_i  for EVERY column i
(closed interval, exactly like Query.doesSatisfy in Query.java).
A column the query does not restrict is stored as [0, 1], which is the
whole domain, so it never filters anything out.

Paper: Section 2.1 ("selectivity" of a predicate B_i is the fraction of
tuples inside the hyperrectangle B_i).
"""

from __future__ import annotations

import numpy as np

# Max number of booleans we create at once (~50 MB). Bigger = faster but more memory.
_MAX_CELLS = 50_000_000


def true_selectivity(data: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Selectivity of many queries at once.

    Args:
        data: (N, d) array, one row per tuple.
        lo:   (Q, d) lower bounds, one row per query.
        hi:   (Q, d) upper bounds.

    Returns:
        (Q,) array of selectivities in [0, 1].
    """
    data = np.asarray(data, dtype=float)
    lo = np.atleast_2d(np.asarray(lo, dtype=float))
    hi = np.atleast_2d(np.asarray(hi, dtype=float))
    n_rows, d = data.shape
    if lo.shape != hi.shape or lo.shape[1] != d:
        raise ValueError(f"query shape {lo.shape}/{hi.shape} does not match data with {d} columns")
    if n_rows == 0:
        return np.zeros(len(lo))

    out = np.empty(len(lo))
    # Process queries in chunks so the (chunk, N, d) boolean array fits in memory.
    chunk = max(1, _MAX_CELLS // (n_rows * d))
    rows = data[None, :, :]  # shape (1, N, d)
    for start in range(0, len(lo), chunk):
        l = lo[start:start + chunk, None, :]  # shape (q, 1, d)
        h = hi[start:start + chunk, None, :]
        inside = ((rows >= l) & (rows <= h)).all(axis=2)  # (q, N): row inside query?
        out[start:start + chunk] = inside.mean(axis=1)
    return out


def count_satisfying(data: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int:
    """Number of rows inside ONE query (port of SyntheticDataGenerator.countSatisfyingItems)."""
    data = np.asarray(data, dtype=float)
    return int(((data >= lo) & (data <= hi)).all(axis=1).sum())
