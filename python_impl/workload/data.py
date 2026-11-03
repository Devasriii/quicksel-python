"""Datasets: synthetic Gaussians (ported from Java/Python 2) and a real-CSV loader.

Every dataset returned here is a numpy array of shape (rows, columns) whose
values all lie in [0, 1], the domain QuickSel works on (paper Section 2.1).

Difference from the Java code (agreed with the team): Java keeps Gaussian
points that fall outside [0, 1] (about 18% of rows for std 0.3). We redraw
them instead, so every row is inside the [0, 1] box. That way an
unconstrained column written as "0,1" in a query file really means
"no restriction", and the default query has selectivity exactly 1.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sample_in_unit_box(sampler: Callable[[int], np.ndarray], size: int) -> np.ndarray:
    """Call `sampler(k)` (returns k rows) repeatedly, keeping only rows inside [0,1]^d,
    until exactly `size` rows have been collected (rejection sampling)."""
    kept: list[np.ndarray] = []
    have = 0
    while have < size:
        batch = sampler(max(1000, 2 * (size - have)))
        batch = batch[((batch >= 0) & (batch <= 1)).all(axis=1)]
        kept.append(batch)
        have += len(batch)
    return np.concatenate(kept)[:size]


# ---------------------------------------------------------------------------
# synthetic datasets
# ---------------------------------------------------------------------------

def two_dim_gaussian(size: int, correlation: float | None = None, seed: int = 0) -> np.ndarray:
    """2-D Gaussian with mean (0.5, 0.5).

    correlation=None -> SyntheticDataGenerator.twoDimGaussianDataset(size):
        fixed covariance [[0.09, 0.04], [0.04, 0.09]]  (correlation ~0.44)
    correlation=c    -> twoDimGaussianDataset(correlation, size):
        std 0.3 for both columns, covariance = c * 0.3 * 0.3
    Used by SpeedComparison.java (no correlation) and PerAttAndSamplingTest.java (0.5).
    """
    if correlation is None:
        cov = np.array([[0.09, 0.04], [0.04, 0.09]])
    else:
        var = 0.3 * 0.3
        cov = np.array([[var, correlation * var], [correlation * var, var]])
    rng = np.random.default_rng(seed)
    return _sample_in_unit_box(lambda k: rng.multivariate_normal([0.5, 0.5], cov, k), size)


def n_dim_gaussian(n: int, size: int, seed: int = 0) -> np.ndarray:
    """SyntheticDataGenerator.nDimGaussianDataset(n, size):
    every column is drawn independently from Normal(mean=0.5, std=0.2)."""
    rng = np.random.default_rng(seed)
    return _sample_in_unit_box(lambda k: rng.normal(0.5, 0.2, size=(k, n)), size)


def three_dim_gaussian(size: int = 100_000, seed: int = 1) -> np.ndarray:
    """test/python/queries/gaussian_data.py gen_gaussian_3d(): the 3-D correlated
    Gaussian the old Python 2 hot/cold/all/transition workloads were run on."""
    cov = np.array([[0.10, 0.06, 0.02],
                    [0.06, 0.10, 0.03],
                    [0.02, 0.03, 0.10]])
    rng = np.random.default_rng(seed)
    return _sample_in_unit_box(lambda k: rng.multivariate_normal([0.5, 0.5, 0.5], cov, k), size)


# ---------------------------------------------------------------------------
# real datasets (DMV, Instacart, ...)
# ---------------------------------------------------------------------------

def _to_number(col: pd.Series) -> pd.Series:
    """Turn one column into numbers: numbers stay, dates -> days, text -> category codes."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float)
    if pd.api.types.is_datetime64_any_dtype(col):
        dates = col
    else:
        # Try to read the text as dates; accept it if (almost) every value parses.
        dates = pd.to_datetime(col, errors="coerce", format="mixed")
        if dates.notna().sum() < 0.9 * col.notna().sum():
            # Not dates -> categories. Codes follow sorted order; missing values -> NaN.
            codes = col.astype("category").cat.codes.astype(float)
            return codes.where(col.notna())
    # Days since 1970-01-01 (missing dates stay NaN).
    return (dates - pd.Timestamp("1970-01-01")).dt.days.astype(float)


def load_csv_dataset(path: str | Path, columns: Sequence[str]
                     ) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Load a real dataset and normalize it to [0, 1].

    Steps: read only `columns` -> convert dates/categories to numbers ->
    drop rows with missing values -> min-max normalize each column:
        x_norm = (x - min) / (max - min)        (a constant column becomes 0)

    Returns:
        (normalized DataFrame, {column: (original_min, original_max)})
        The min/max let you turn a normalized bound back into a real value:
        real = min + x_norm * (max - min).
    """
    df = pd.read_csv(path, usecols=list(columns), low_memory=False)
    df = df[list(columns)]  # keep the order the user asked for
    df = df.apply(_to_number).dropna().reset_index(drop=True)
    ranges: dict[str, tuple[float, float]] = {}
    for c in df.columns:
        lo, hi = float(df[c].min()), float(df[c].max())
        ranges[c] = (lo, hi)
        df[c] = (df[c] - lo) / (hi - lo) if hi > lo else 0.0
    return df, ranges


# Columns used in the paper (Section 5.1).
DMV_COLUMNS = ["model_year", "registration_date", "expiration_date"]
INSTACART_COLUMNS = ["order_hour_of_day", "days_since_prior"]


def make_demo_dmv_csv(path: str | Path, rows: int = 20_000, seed: int = 0) -> Path:
    """Write a FAKE DMV-like CSV (same column names as the real NY DMV data) so the
    real-data pipeline can be tested before the real file is downloaded."""
    rng = np.random.default_rng(seed)
    model_year = np.clip(np.round(rng.normal(2008, 6, rows)), 1950, 2019).astype(int)
    reg = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, rows), unit="D")
    exp = reg + pd.to_timedelta(rng.integers(365, 3 * 365, rows), unit="D")
    color = rng.choice(["BLACK", "WHITE", "SILVER", "RED", "BLUE"], rows)
    df = pd.DataFrame({
        "model_year": model_year,
        "registration_date": reg.strftime("%m/%d/%Y"),   # text dates, like the real file
        "expiration_date": exp.strftime("%m/%d/%Y"),
        "color": color,                                  # a text column (category)
    })
    df.to_csv(path, index=False)
    return Path(path)


def make_demo_instacart_csv(path: str | Path, rows: int = 20_000, seed: int = 0) -> Path:
    """Write a FAKE Instacart orders.csv (same column names) for testing.
    Like the real file, days_since_prior is empty for a user's first order."""
    rng = np.random.default_rng(seed)
    hour = np.clip(np.round(rng.normal(13.5, 4.2, rows)), 0, 23).astype(int)
    days = np.clip(np.round(rng.exponential(10, rows)), 0, 30)
    days[rng.random(rows) < 0.06] = np.nan  # first orders have no "prior"
    df = pd.DataFrame({
        "order_id": np.arange(1, rows + 1),
        "eval_set": rng.choice(["prior", "train", "test"], rows, p=[0.9, 0.07, 0.03]),
        "order_hour_of_day": hour,
        "days_since_prior": days,
    })
    df.to_csv(path, index=False)
    return Path(path)
