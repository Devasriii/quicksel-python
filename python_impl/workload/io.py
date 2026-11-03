"""Read and write query files in the repo format.

One query per line, comma-separated, no header:

    lo1,hi1,lo2,hi2,...,loN,hiN,selectivity

All values are normalized to [0, 1]. This is the format of
test/java/edu/illinois/quicksel/resources/DMV/assertion_dmv.txt and of
Instacart/assertion_instacart.txt, so the Java AssertionReader can read
our files too.

NOTE: always import this as  `from python_impl.workload import io as qio`
and run scripts with `python -m ...`, so it never hides Python's built-in `io`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .queries import QuerySet

# Copies of the repo's query files (copied from C:\Users\Uma\quicksel\test\java\
# edu\illinois\quicksel\resources\) so this folder works on its own.
RESOURCES = Path(__file__).parent / "resources"

_REPO_FILES = {
    "DMV": ("DMV/assertion_dmv.txt", "DMV/permanent_assertion_dmv.txt"),
    "Instacart": ("Instacart/assertion_instacart.txt", "Instacart/permanent_assertion_instacart.txt"),
}


def write_queries(path: str | Path, qs: QuerySet) -> None:
    """Write queries as  lo1,hi1,...,loN,hiN,sel  (bounds 6 decimals, selectivity 9 decimals)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for lo, hi, sel in zip(qs.lo, qs.hi, qs.sel):
            pairs = ",".join(f"{l:.6f},{h:.6f}" for l, h in zip(lo, hi))
            f.write(f"{pairs},{sel:.9f}\n")


def read_queries(path: str | Path, clamp: bool = True) -> QuerySet:
    """Read a query file (port of AssertionReader.readAssertion(file)).

    columns = (values_per_line - 1) / 2. With clamp=True every bound is clipped
    to [0, 1], as the Java checkInBoundary(x, 1) does (the repo's Instacart file
    contains bounds like -0.117 and 1.25).
    """
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    if arr.shape[1] % 2 != 1:
        raise ValueError(f"{path}: expected 2*d+1 values per line, got {arr.shape[1]}")
    lo, hi, sel = arr[:, 0:-1:2], arr[:, 1:-1:2], arr[:, -1]
    if clamp:
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    return QuerySet(lo, hi, sel)


def read_repo_assertions(dataset: str = "DMV", root: str | Path | None = None
                         ) -> tuple[QuerySet, QuerySet, QuerySet]:
    """Read the repo's DMV or Instacart query files.

    Returns (train, test, permanent) split exactly like DMVSpeedComparison.java /
    InstacartSpeedComparison.java: training = lines 0..999 (they use the first
    10, 20, ..., 1000 of them), test = lines 1000..1099, permanent = the
    whole-domain query.

    root: folder holding DMV/ and Instacart/; defaults to our bundled copies.
    For the original repo use
    r"C:\\Users\\Uma\\quicksel\\test\\java\\edu\\illinois\\quicksel\\resources".
    """
    if dataset not in _REPO_FILES:
        raise ValueError(f"dataset must be one of {list(_REPO_FILES)}")
    root = Path(root) if root is not None else RESOURCES
    main_file, perm_file = _REPO_FILES[dataset]
    allq = read_queries(root / main_file)
    return allq.take(slice(0, 1000)), allq.take(slice(1000, 1100)), read_queries(root / perm_file)


def write_dataset(path: str | Path, data: np.ndarray, columns: list[str]) -> None:
    """Save the (normalized) dataset as CSV with a header row."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=columns).to_csv(path, index=False, float_format="%.6f")


def read_dataset(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a dataset CSV written by write_dataset -> (array (N, d), column names)."""
    df = pd.read_csv(path)
    return df.to_numpy(dtype=float), list(df.columns)
