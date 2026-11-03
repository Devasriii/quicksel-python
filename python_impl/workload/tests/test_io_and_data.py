"""File format, repo files, dataset generators and the CSV loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from python_impl.workload import data as D
from python_impl.workload import io as qio
from python_impl.workload import queries as Q

REPO_RESOURCES = Path(r"C:\Users\Uma\quicksel\test\java\edu\illinois\quicksel\resources")


# ---------------- io ----------------

def test_write_read_round_trip(tmp_path):
    data = D.two_dim_gaussian(1000, seed=0)
    qs = Q.finalize(Q.random_scattered(50, 0), data)
    qio.write_queries(tmp_path / "q.txt", qs)
    back = qio.read_queries(tmp_path / "q.txt")
    np.testing.assert_array_equal(back.lo, qs.lo)  # bounds were rounded to 6 decimals already
    np.testing.assert_array_equal(back.hi, qs.hi)
    np.testing.assert_allclose(back.sel, qs.sel, atol=1e-9)


def test_written_line_format(tmp_path):
    qs = Q.QuerySet(np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]]), np.array([0.5]))
    qio.write_queries(tmp_path / "q.txt", qs)
    assert (tmp_path / "q.txt").read_text() == "0.100000,0.300000,0.200000,0.400000,0.500000000\n"


def test_reads_bundled_repo_files():
    train, test, perm = qio.read_repo_assertions("DMV")
    assert (len(train), len(test), train.dims) == (1000, 100, 3)
    train, test, perm = qio.read_repo_assertions("Instacart")
    assert (len(train), len(test), train.dims) == (1000, 100, 2)
    assert perm.sel[0] == 1.0
    # The Instacart file has bounds like -0.117; the reader clamps them like Java.
    assert train.lo.min() >= 0 and train.hi.max() <= 1
    raw = qio.read_queries(qio.RESOURCES / "Instacart/assertion_instacart.txt", clamp=False)
    assert raw.lo.min() < 0


@pytest.mark.skipif(not REPO_RESOURCES.exists(), reason="original repo not on this machine")
def test_reads_original_repo_files():
    for name in ("DMV", "Instacart"):
        a = qio.read_repo_assertions(name, root=REPO_RESOURCES)
        b = qio.read_repo_assertions(name)
        np.testing.assert_array_equal(a[0].lo, b[0].lo)


def test_dataset_round_trip(tmp_path):
    data = np.round(D.n_dim_gaussian(3, 100, seed=0), 6)
    qio.write_dataset(tmp_path / "d.csv", data, ["a", "b", "c"])
    back, cols = qio.read_dataset(tmp_path / "d.csv")
    assert cols == ["a", "b", "c"]
    np.testing.assert_allclose(back, data, atol=1e-6)


# ---------------- data ----------------

@pytest.mark.parametrize("make", [
    lambda: D.two_dim_gaussian(3000, seed=1),
    lambda: D.two_dim_gaussian(3000, correlation=0.5, seed=1),
    lambda: D.n_dim_gaussian(5, 3000, seed=1),
    lambda: D.three_dim_gaussian(3000, seed=1),
])
def test_gaussians_inside_unit_box_with_exact_rows(make):
    data = make()
    assert len(data) == 3000
    assert data.min() >= 0 and data.max() <= 1


def test_correlation_parameter_works():
    data = D.two_dim_gaussian(20000, correlation=0.8, seed=0)
    assert np.corrcoef(data.T)[0, 1] > 0.6  # truncation lowers it a bit, but clearly positive


def test_load_csv_converts_and_normalizes(tmp_path):
    pd.DataFrame({
        "year": [2000, 2010, 2020, None],
        "date": ["01/01/2020", "01/11/2020", "01/21/2020", "01/31/2020"],
        "color": ["red", "blue", "red", "green"],
        "ignored": [1, 2, 3, 4],
    }).to_csv(tmp_path / "t.csv", index=False)
    df, ranges = D.load_csv_dataset(tmp_path / "t.csv", ["year", "date", "color"])
    assert list(df.columns) == ["year", "date", "color"]
    assert len(df) == 3                                  # row with missing year dropped
    np.testing.assert_allclose(df["year"], [0, 0.5, 1])
    np.testing.assert_allclose(df["date"], [0, 0.5, 1])  # Jan 1, 11, 21 -> evenly spaced days
    assert set(df["color"]) == {0.0, 1.0}                # blue, red (green was dropped with row 4)
    assert ranges["year"] == (2000.0, 2020.0)


def test_demo_csvs_load(tmp_path):
    df, _ = D.load_csv_dataset(D.make_demo_dmv_csv(tmp_path / "d.csv", 500), D.DMV_COLUMNS)
    assert df.shape[1] == 3 and df.to_numpy().min() >= 0 and df.to_numpy().max() <= 1
    df, _ = D.load_csv_dataset(D.make_demo_instacart_csv(tmp_path / "i.csv", 500), D.INSTACART_COLUMNS)
    assert df.shape[1] == 2 and len(df) < 500            # first orders (no days_since_prior) dropped
