"""Query (range predicate) workload generators.

A query is a hyperrectangle: one interval [lo_i, hi_i] per column. Each pair
(query, true selectivity) is what the paper calls an "observed query" and what
the Java code calls an Assertion (src/java/edu/illinois/quicksel/Assertion.java).

Each generator is a 1-to-1 port of a Java method in
test/java/edu/illinois/quicksel/basic/SyntheticDataGenerator.java or a
function in test/python/queries/gen_gaussian_queries.py. Wherever Java uses
`new Random(seed)` we use JavaRandom(seed), so the rectangles are exactly the
same as the Java ones. The Python 2 code used numpy's unseeded global RNG;
we use a seeded numpy Generator instead (same logic, reproducible).

Most generators return only the bounds (lo, hi). `finalize` then clamps them
to [0, 1], rounds them to 6 decimals (what gets written to file) and computes
the true selectivity on the ROUNDED bounds, so the file is exactly right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .java_random import JavaRandom
from .selectivity import true_selectivity

Bounds = tuple[np.ndarray, np.ndarray]  # (lo, hi), each of shape (Q, d)

DECIMALS = 6  # bounds are written with 6 decimals


@dataclass
class QuerySet:
    """Q queries over d columns: lo[q, i] <= x_i <= hi[q, i], with selectivity sel[q]."""

    lo: np.ndarray   # (Q, d)
    hi: np.ndarray   # (Q, d)
    sel: np.ndarray  # (Q,)

    def __len__(self) -> int:
        return len(self.sel)

    @property
    def dims(self) -> int:
        return self.lo.shape[1]

    def take(self, idx: Sequence[int] | np.ndarray | slice) -> "QuerySet":
        """Subset of the queries (e.g. qs.take(slice(0, 100)) = first 100)."""
        return QuerySet(self.lo[idx], self.hi[idx], self.sel[idx])

    @staticmethod
    def concat(sets: Sequence["QuerySet"]) -> "QuerySet":
        return QuerySet(np.concatenate([s.lo for s in sets]),
                        np.concatenate([s.hi for s in sets]),
                        np.concatenate([s.sel for s in sets]))


def finalize(bounds: Bounds, data: np.ndarray) -> QuerySet:
    """Clamp bounds to [0, 1], round to 6 decimals, compute true selectivity on `data`."""
    lo, hi = (np.round(np.clip(np.asarray(b, dtype=float), 0.0, 1.0), DECIMALS) for b in bounds)
    assert (lo <= hi).all(), "every query must have lo <= hi"
    return QuerySet(lo, hi, true_selectivity(data, lo, hi))


def evenly_spaced(qs: QuerySet, k: int) -> QuerySet:
    """Pick k queries spread evenly over a sequence (used for test sets of sequence workloads)."""
    idx = np.linspace(0, len(qs) - 1, num=min(k, len(qs))).round().astype(int)
    return qs.take(idx)


# ===========================================================================
# Ports of SyntheticDataGenerator.java
# ===========================================================================

def random_overlapping(n: int, seed: int = 0) -> Bounds:
    """twoDimGaussianDatasetRandomAssertions: paper Fig. 2, Case 1 (highly overlapping).

    Each side is [U*0.5, U*0.5 + 0.5], so every rectangle contains the centre
    (0.5, 0.5) and the queries overlap heavily.
    seed=0 matches Java's training queries; seed=1 matches
    twoDimGaussianDatasetRandomTestQueries (Java's test queries).
    """
    rand = JavaRandom(seed)                     # Random rand = new Random(0);
    lo, hi = np.empty((n, 2)), np.empty((n, 2))
    for i in range(n):
        for dim in (0, 1):                      # r1.put(dim, Pair.of(a, b))  (Java evaluates a then b)
            lo[i, dim] = rand.next_double() * 0.5
            hi[i, dim] = rand.next_double() * 0.5 + 0.5
    return lo, hi


def random_scattered(n: int, seed: int = 0) -> Bounds:
    """twoDimGaussianDatasetRandomAssertions2: paper Fig. 2, Case 2 (scattered).

    Small 0.1 x 0.1 squares placed uniformly at random, so they rarely overlap.
    Used by SpeedComparison.java.
    """
    rand = JavaRandom(seed)
    lo = np.empty((n, 2))
    for i in range(n):
        lo[i, 0] = rand.next_double() * 0.9    # double d1 = rand.nextDouble()*0.9;
        lo[i, 1] = rand.next_double() * 0.9    # double d2 = rand.nextDouble()*0.9;
    return lo, lo + 0.1                         # Pair.of(d, d+0.1)


def n_dim_random(n: int, dims: int, seed: int = 0) -> Bounds:
    """nDimRandomAssertions: in every column, [U*0.5, U*0.5 + 0.5].

    (The Java comment says the width is 0.1^(1/d), but the Java code uses 0.5.
    We copy the code, not the comment.)
    """
    rand = JavaRandom(seed)
    lo = np.empty((n, dims))
    for i in range(n):
        for j in range(dims):
            lo[i, j] = rand.next_double() * 0.5   # left = rand.nextDouble() * 0.5
    return lo, lo + 0.5                            # right = left + 0.5


def n_dim_fixed_selectivity(n: int, dims: int, seed: int = 0) -> Bounds:
    """nDimRandomAssertionsSelectivityFixed: every side has width w = 0.1^(1/d),
    so the box VOLUME is always 0.1 (selectivity ~10% on uniform data)."""
    rand = JavaRandom(seed)
    w = 0.1 ** (1.0 / dims)
    lo = np.empty((n, dims))
    for i in range(n):
        for j in range(dims):
            # Java: while (left < 0) left = rand.nextDouble()*(1-w);  (runs exactly once)
            lo[i, j] = rand.next_double() * (1 - w)
    return lo, lo + w


def workflow(n: int | None = None, seed: int = 0) -> Bounds:
    """workflowAssertion: a 'sliding' workload whose 0.2 x 0.2 box drifts along the
    diagonal from (0.1, 0.1) to (0.9, 0.9), as if a user explores the data step by step.

    n=None reproduces Java's loop exactly:  for (i = 0.1; i <= 0.9; i += 0.0008)
    (about 1001 queries, with the same floating-point steps). With n given, the
    n centres are spread evenly from 0.1 to 0.9 instead.

    Note: the Java comment says the centre gets randn() noise, but the code uses
    nextDouble(): cr = (c - r*0.1, c + r*0.1). We copy the code. Java used an
    unseeded Random(); we use JavaRandom(seed).
    """
    if n is None:
        cc: list[float] = []
        c = 0.1
        while c <= 0.9:          # same float accumulation as Java's double loop
            cc.append(c)
            c += 0.0008
    else:
        cc = list(np.linspace(0.1, 0.9, n))
    rand = JavaRandom(seed)
    lo, hi = np.empty((len(cc), 2)), np.empty((len(cc), 2))
    for i, c in enumerate(cc):
        r = rand.next_double()
        cx, cy = c - r * 0.1, c + r * 0.1                  # Pair cr = (c-r*0.1, c+r*0.1)
        lo[i] = [max(0.0, cx - 0.1), max(0.0, cy - 0.1)]   # (x-0.1)<0 ? 0 : x-0.1
        hi[i] = [1.0 if cx + 0.1 >= 1 else cx + 0.1,       # (x+0.1)>=1 ? 1 : x+0.1
                 1.0 if cy + 0.1 >= 1 else cy + 0.1]
    return lo, hi


def default_query(dims: int) -> QuerySet:
    """twoDimDefaultAssertion / nDimDefaultAssertion: the whole domain [0,1]^d,
    selectivity 1.0. QuickSel always adds it as a 'permanent' assertion (it tells
    the model that the total probability is 1)."""
    return QuerySet(np.zeros((1, dims)), np.ones((1, dims)), np.array([1.0]))


def grid_strips(grid: int, dims: int = 2) -> Bounds:
    """twoDimPermanentAssertions: `grid` strips along column 0, then `grid` along
    column 1, ... (each strip restricts ONE column; the others are [0, 1])."""
    step = 1.0 / grid
    lo, hi = [], []
    for dim in range(dims):
        for i in range(grid):
            l, h = np.zeros(dims), np.ones(dims)
            l[dim], h[dim] = step * i, step * (i + 1)
            lo.append(l)
            hi.append(h)
    return np.array(lo), np.array(hi)


def grid_cells(grid: int) -> Bounds:
    """twoDimGaussianDatasetGridTestQueries: all grid x grid cells of a 2-D grid
    (row i outer, column j inner, same order as Java)."""
    step = 1.0 / grid
    lo = np.array([[step * i, step * j] for i in range(grid) for j in range(grid)])
    return lo, lo + step


# ===========================================================================
# Ports of test/python/queries/gen_gaussian_queries.py  (Python 2)
# ===========================================================================

DELTA = 0.1  # `delta` in the Python 2 file


def _is_cold(data: np.ndarray, center: np.ndarray) -> bool:
    """check_cold: the cube center +- DELTA holds fewer than N * DELTA^d points.
    (vectorized version of the Python 2 loop; for d=3 it is exactly the same rule)"""
    inside = ((data >= center - DELTA) & (data <= center + DELTA)).all(axis=1).sum()
    return inside < len(data) * DELTA ** data.shape[1]


def _hot_cold_queries(data: np.ndarray, n: int, cold: int, seed: int) -> Bounds:
    """gen_query(data, cold, size). cold=0: anywhere, 1: hot (dense) areas only,
    2: cold (sparse) areas only. Side lengths U(0.05, 0.15), centre U(0.05, 0.95)."""
    rng = np.random.default_rng(seed)
    d = data.shape[1]
    lo, hi = [], []
    tries = 0
    while len(lo) < n:
        tries += 1
        if tries > 1000 * n:  # safety net (the Python 2 loop could run forever)
            raise RuntimeError(f"found only {len(lo)} of {n} {'hot' if cold == 1 else 'cold'} "
                               "queries; the data may have no such areas")
        sides = rng.uniform(0.05, 0.15, d)       # [l1, l2, l3] = uniform(0.05, 0.15, 3)
        center = rng.uniform(0.05, 0.95, d)      # center = uniform(0.05, 0.95, 3)
        if cold == 1 and _is_cold(data, center):
            continue
        if cold == 2 and not _is_cold(data, center):
            continue
        lo.append(np.maximum(0, center - sides / 2))   # ll = [max(0, x) ...]
        hi.append(np.minimum(1, center + sides / 2))   # ur = [min(1, x) ...]
    return np.array(lo), np.array(hi)


def hot(data: np.ndarray, n: int, seed: int = 0) -> Bounds:
    """gen_hot: queries only in dense areas of the data."""
    return _hot_cold_queries(data, n, cold=1, seed=seed)


def cold(data: np.ndarray, n: int, seed: int = 0) -> Bounds:
    """gen_cold: queries only in sparse areas of the data."""
    return _hot_cold_queries(data, n, cold=2, seed=seed)


def all_area(data: np.ndarray, n: int, seed: int = 0) -> Bounds:
    """gen_all: queries anywhere."""
    return _hot_cold_queries(data, n, cold=0, seed=seed)


def transition(dims: int, n: int = 3200, seed: int = 0) -> Bounds:
    """gen_query_trans: the query centre starts at (0.49, ...) and moves a tiny
    step down the diagonal after every query (plus small Gaussian noise), ending
    near (0.09, ...). A slowly changing workload.

    The Python 2 code had n=3200 and step=-1.25e-4; for another n we rescale
    the step so the centre covers the same path.
    """
    rng = np.random.default_rng(seed)
    step = -1.25e-4 * 3200 / n
    exp_center = np.full(dims, 0.49)
    lo, hi = np.empty((n, dims)), np.empty((n, dims))
    for i in range(n):
        sides = rng.uniform(0.05, 0.15, dims)
        noise = rng.multivariate_normal(np.zeros(dims), 1e-4 * np.eye(dims))
        center = exp_center + noise
        lo[i] = np.maximum(0, center - sides / 2)
        hi[i] = np.minimum(1, center + sides / 2)
        exp_center = exp_center + step               # exp_center = [x + step ...]
    return lo, hi


# ===========================================================================
# New generators (no original code in the repo)
# ===========================================================================

def workload_shift(dims: int = 2, n_regions: int = 3, per_region: int = 100,
                   seed: int = 0, region_seed: int | None = None,
                   noise: float = 0.03) -> Bounds:
    """Paper Section 5.6 / Figure 7: 'after observing a certain number of queries
    (100) around a certain region of data, the query workload suddenly jumps to
    a novel region'.

    Queries [k*per_region, (k+1)*per_region) belong to region k. Region centres
    are random in [0.2, 0.8]^d and at least 0.3 apart; each query is a box with
    sides U(0.05, 0.15) around the region centre + Normal(0, noise) jitter.

    region_seed picks the region centres (default = seed); seed picks the queries.
    Use the same region_seed with another seed to get test queries in the same regions.
    """
    region_rng = np.random.default_rng(seed if region_seed is None else region_seed)
    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = []
    tries = 0
    while len(centers) < n_regions:
        c = region_rng.uniform(0.2, 0.8, dims)
        tries += 1
        # keep it only if it is far from every earlier centre (give up on the rule after many tries)
        if tries > 10_000 or all(np.linalg.norm(c - o) >= 0.3 for o in centers):
            centers.append(c)
    lo, hi = [], []
    for c in centers:
        for _ in range(per_region):
            sides = rng.uniform(0.05, 0.15, dims)
            center = c + rng.normal(0, noise, dims)
            lo.append(np.maximum(0, center - sides / 2))
            hi.append(np.minimum(1, center + sides / 2))
    return np.array(lo), np.array(hi)


def dmv_queries(n: int, seed: int = 0,
                date_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
                t_range: tuple[float, float] = (0.773, 0.908)) -> Bounds:
    """Queries for DMV (paper Section 5.1): 'the number of valid registrations for
    vehicles produced within a certain date range'. Columns, in order:
    model_year, registration_date, expiration_date.

    Reverse-engineered from the repo file assertion_dmv.txt:
      * model_year in [a, a + 0.2], a ~ U(0, 1)           (a 0.2-wide window)
      * one date t: the registration is valid on t, i.e.
            registration_date <= t   -> [0, t]
            expiration_date   >= t   -> [t, 1]
        (In the repo file t ~ U(0.773, 0.908) and the two date columns appear as
        [t,1] then [0,t], so that file seems to list expiration before registration.)

    date_ranges = ((reg_min, reg_max), (exp_min, exp_max)) in days (from
    load_csv_dataset). If given, a REAL date T is drawn where both columns have
    data and converted into each column's own normalized scale, which is needed
    because each column is normalized separately. If None, the same normalized t
    is used for both columns, exactly like the repo file.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.empty((n, 3)), np.empty((n, 3))
    for i in range(n):
        a = rng.uniform(0, 1)
        if date_ranges is None:
            t_reg = t_exp = rng.uniform(*t_range)
        else:
            (r0, r1), (e0, e1) = date_ranges
            low, high = max(r0, e0), min(r1, e1)
            if low >= high:          # no overlap: just use the registration range
                low, high = r0, r1
            T = rng.uniform(low, high)
            t_reg = (T - r0) / (r1 - r0) if r1 > r0 else 0.0
            t_exp = (T - e0) / (e1 - e0) if e1 > e0 else 0.0
        lo[i] = [a, 0.0, t_exp]
        hi[i] = [a + 0.2, t_reg, 1.0]
    return lo, hi


def instacart_queries(n: int, dims: int = 2, seed: int = 0, width: float = 0.5) -> Bounds:
    """Queries for Instacart (paper Section 5.1: order_hour_of_day, days_since_prior).

    Reverse-engineered from assertion_instacart.txt: every side is 0.5 wide and
    its centre is U(0, 1) (the repo file keeps bounds like -0.117 and the Java
    reader clamps them; `finalize` clamps them here).
    """
    rng = np.random.default_rng(seed)
    center = rng.uniform(0, 1, (n, dims))
    return center - width / 2, center + width / 2
