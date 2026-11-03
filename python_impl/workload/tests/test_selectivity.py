"""True-selectivity tests on tiny hand-made data where we know the answer."""

import numpy as np

from python_impl.workload.queries import default_query
from python_impl.workload.selectivity import count_satisfying, true_selectivity

# 5 points in 2-D, chosen by hand.
TINY = np.array([
    [0.1, 0.1],
    [0.2, 0.8],
    [0.5, 0.5],   # sits exactly on the boundary of query A below
    [0.7, 0.3],
    [0.9, 0.9],
])


def test_exact_selectivity_on_tiny_data():
    lo = np.array([[0.0, 0.0],    # A: x in [0,0.5], y in [0,0.5] -> (0.1,0.1), (0.5,0.5) = 2/5
                   [0.6, 0.0],    # B: x in [0.6,1], y in [0,1]   -> (0.7,0.3), (0.9,0.9) = 2/5
                   [0.3, 0.3],    # C: nothing inside              -> 0
                   [0.0, 0.0]])   # D: whole domain                -> 5/5
    hi = np.array([[0.5, 0.5],
                   [1.0, 1.0],    # y unconstrained = [0, 1]
                   [0.4, 0.4],
                   [1.0, 1.0]])
    np.testing.assert_allclose(true_selectivity(TINY, lo, hi), [0.4, 0.4, 0.0, 1.0])


def test_boundary_is_inclusive_like_java():
    # Query.doesSatisfy rejects only x < lo or x > hi, so a point exactly on the edge counts.
    assert count_satisfying(TINY, np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 1


def test_default_query_selects_everything():
    qs = default_query(2)
    assert qs.sel[0] == 1.0
    assert true_selectivity(TINY, qs.lo, qs.hi)[0] == 1.0


def test_chunking_gives_same_answer(monkeypatch):
    import python_impl.workload.selectivity as S
    rng = np.random.default_rng(0)
    data = rng.random((500, 3))
    lo = rng.random((40, 3)) * 0.5
    hi = lo + 0.4
    full = true_selectivity(data, lo, hi)
    monkeypatch.setattr(S, "_MAX_CELLS", 1)  # force one query per chunk
    np.testing.assert_array_equal(S.true_selectivity(data, lo, hi), full)
