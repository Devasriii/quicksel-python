"""Query generator tests: valid ranges, reproducibility, and Java-exactness."""

import numpy as np
import pytest

from python_impl.workload import data as D
from python_impl.workload import queries as Q
from python_impl.workload.java_random import JavaRandom

DATA2 = D.two_dim_gaussian(2000, seed=0)
DATA3 = D.three_dim_gaussian(2000, seed=0)


def test_java_random_matches_java():
    # Known outputs of  new java.util.Random(0).nextDouble()  (called twice).
    r = JavaRandom(0)
    assert r.next_double() == 0.730967787376657
    assert r.next_double() == 0.24053641567148587


def test_overlapping_first_query_matches_java():
    lo, hi = Q.random_overlapping(1, seed=0)
    # Java: Pair.of(r1*0.5, r2*0.5+0.5) for column 0
    assert lo[0, 0] == 0.730967787376657 * 0.5
    assert hi[0, 0] == 0.24053641567148587 * 0.5 + 0.5


GENERATORS = {
    "overlapping": lambda s: Q.random_overlapping(200, s),
    "scattered": lambda s: Q.random_scattered(200, s),
    "ndim_random": lambda s: Q.n_dim_random(200, 2, s),
    "ndim_fixed": lambda s: Q.n_dim_fixed_selectivity(200, 2, s),
    "workflow": lambda s: Q.workflow(None, s),
    "transition": lambda s: Q.transition(2, 300, s),
    "shift": lambda s: Q.workload_shift(2, 3, 50, s),
    "instacart": lambda s: Q.instacart_queries(200, 2, s),
    "grid": lambda s: Q.grid_cells(5),
}


@pytest.mark.parametrize("name", GENERATORS)
def test_queries_are_valid(name):
    qs = Q.finalize(GENERATORS[name](0), DATA2)
    assert (qs.lo <= qs.hi).all()
    assert (qs.lo >= 0).all() and (qs.hi <= 1).all()
    assert ((qs.sel >= 0) & (qs.sel <= 1)).all()


@pytest.mark.parametrize("name", [n for n in GENERATORS if n != "grid"])
def test_same_seed_same_queries_and_test_differs(name):
    a, b, c = GENERATORS[name](0), GENERATORS[name](0), GENERATORS[name](1)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])  # seed+1 (test set) is different


def test_workflow_java_loop_length():
    lo, _ = Q.workflow(None, 0)
    assert len(lo) in (1000, 1001)  # float accumulation of 0.1 += 0.0008 up to 0.9


@pytest.mark.parametrize("fn", [Q.hot, Q.cold, Q.all_area])
def test_hot_cold_all(fn):
    qs = Q.finalize(fn(DATA3, 30, seed=0), DATA3)
    assert len(qs) == 30 and qs.dims == 3
    assert ((qs.sel >= 0) & (qs.sel <= 1)).all()


def test_hot_queries_are_denser_than_cold():
    hot = Q.finalize(Q.hot(DATA3, 50, 0), DATA3).sel.mean()
    cold = Q.finalize(Q.cold(DATA3, 50, 0), DATA3).sel.mean()
    assert hot > cold


def test_dmv_shape_matches_repo_pattern():
    lo, hi = Q.dmv_queries(100, seed=0)
    np.testing.assert_allclose(hi[:, 0] - lo[:, 0], 0.2)   # model_year window 0.2 wide
    assert (lo[:, 1] == 0).all() and (hi[:, 2] == 1).all()  # registration <= t, expiration >= t
    np.testing.assert_allclose(hi[:, 1], lo[:, 2])          # same t in both columns


def test_grid_strips_count_and_shape():
    lo, hi = Q.grid_strips(4)
    assert lo.shape == (8, 2)
    assert (lo[:4, 1] == 0).all() and (hi[:4, 1] == 1).all()  # first strips restrict column 0 only


def test_shift_regions_shared_between_train_and_test():
    tr_lo, tr_hi = Q.workload_shift(2, 3, 100, seed=0, region_seed=0)
    te_lo, te_hi = Q.workload_shift(2, 3, 30, seed=1, region_seed=0)
    for k in range(3):
        c_train = ((tr_lo + tr_hi) / 2)[k * 100:(k + 1) * 100].mean(axis=0)
        c_test = ((te_lo + te_hi) / 2)[k * 30:(k + 1) * 30].mean(axis=0)
        assert np.linalg.norm(c_train - c_test) < 0.05
