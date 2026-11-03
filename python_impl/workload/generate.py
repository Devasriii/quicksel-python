"""Command-line workload generator.

Example (run from the project root):

    python -m python_impl.workload.generate --mode scattered --rows 10000 \
        --train 1000 --test 100 --seed 0 --out output/scattered

Writes into --out:
    data.csv               the dataset (normalized to [0,1], header = column names)
    train_queries.txt      training ("observed") queries   lo1,hi1,...,loN,hiN,sel
    test_queries.txt       separate test queries (paper: 100 test queries not used for training)
    permanent_queries.txt  the whole-domain query with selectivity 1.0
    meta.json              every option used + column names (+ original min/max for CSV data)

Seeds: dataset and training queries use --seed, test queries use --seed + 1.
With --seed 0 the overlapping/scattered/ndim/workflow training queries are
exactly the Java ones, and the overlapping test queries are exactly Java's
twoDimGaussianDatasetRandomTestQueries (new Random(1)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import data as D
from . import io as qio
from . import queries as Q

MODES = ["overlapping", "scattered", "ndim_random", "ndim_fixed", "workflow",
         "hot", "cold", "all", "transition", "shift", "grid", "dmv", "instacart"]

# Dataset used when --data is not given.
DEFAULT_DATA = {
    "overlapping": "gaussian2d", "scattered": "gaussian2d", "workflow": "gaussian2d",
    "grid": "gaussian2d", "shift": "gaussian2d",
    "ndim_random": "gaussiannd", "ndim_fixed": "gaussiannd",
    "hot": "gaussian3d", "cold": "gaussian3d", "all": "gaussian3d", "transition": "gaussian3d",
    "dmv": "csv", "instacart": "csv",
}

DEFAULT_COLUMNS = {"dmv": D.DMV_COLUMNS, "instacart": D.INSTACART_COLUMNS}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m python_impl.workload.generate",
                                description="Generate a QuickSel query workload.")
    p.add_argument("--mode", required=True, choices=MODES, help="which query generator to use")
    p.add_argument("--data", choices=["gaussian2d", "gaussian3d", "gaussiannd", "csv"],
                   help="dataset (default depends on --mode)")
    p.add_argument("--rows", type=int, default=None,
                   help="dataset rows (synthetic default 10000; for csv: random sample of this "
                        "many rows, default = all rows)")
    p.add_argument("--train", type=int, default=None,
                   help="number of training queries (default 1000; workflow: Java's ~1001; "
                        "transition: 3200; shift: 100 per region; grid: 2*grid strips)")
    p.add_argument("--test", type=int, default=100, help="number of test queries (default 100)")
    p.add_argument("--seed", type=int, default=0, help="random seed (test queries use seed+1)")
    p.add_argument("--out", default="output", help="output folder (default: output)")
    p.add_argument("--corr", type=float, default=None,
                   help="gaussian2d correlation (default: Java's fixed covariance)")
    p.add_argument("--dims", type=int, default=4, help="gaussiannd columns (default 4)")
    p.add_argument("--csv", default=None,
                   help="CSV file for --data csv (default: create a demo CSV in --out)")
    p.add_argument("--columns", nargs="+", default=None,
                   help="CSV columns to use (default for dmv/instacart: the paper's columns)")
    p.add_argument("--grid", type=int, default=10, help="grid size for --mode grid (default 10)")
    p.add_argument("--regions", type=int, default=3, help="regions for --mode shift (default 3)")
    return p


def load_data(args: argparse.Namespace, out: Path) -> tuple[np.ndarray, list[str], dict]:
    """Build or load the dataset. Returns (data array, column names, extra info for meta.json)."""
    kind = args.data or DEFAULT_DATA[args.mode]
    rows = args.rows or 10_000
    if kind == "gaussian2d":
        data = D.two_dim_gaussian(rows, args.corr, seed=args.seed)
    elif kind == "gaussian3d":
        data = D.three_dim_gaussian(rows, seed=args.seed)
    elif kind == "gaussiannd":
        data = D.n_dim_gaussian(args.dims, rows, seed=args.seed)
    else:
        columns = args.columns or DEFAULT_COLUMNS.get(args.mode)
        if columns is None:
            raise SystemExit("--data csv needs --columns")
        csv = args.csv
        if csv is None:  # no real file yet: make a fake one with the same columns
            if args.mode not in ("dmv", "instacart"):
                raise SystemExit("--data csv needs --csv PATH")
            make = D.make_demo_dmv_csv if args.mode == "dmv" else D.make_demo_instacart_csv
            csv = make(out / f"demo_{args.mode}.csv", rows=20_000, seed=args.seed)
            print(f"(no --csv given: using demo file {csv})")
        df, ranges = D.load_csv_dataset(csv, columns)
        if args.rows and args.rows < len(df):
            df = df.sample(n=args.rows, random_state=args.seed).reset_index(drop=True)
        return np.round(df.to_numpy(float), 6), list(df.columns), {"csv": str(csv), "ranges": ranges}
    # Round like the written data.csv, so recomputing selectivity from data.csv gives the same numbers.
    return np.round(data, 6), [f"x{i}" for i in range(data.shape[1])], {}


def make_queries(args: argparse.Namespace, data: np.ndarray, info: dict
                 ) -> tuple[Q.QuerySet, Q.QuerySet]:
    """Return (train, test) query sets for the chosen mode."""
    m, s, d = args.mode, args.seed, data.shape[1]
    n_train = args.train or 1000

    # --- independent-query modes: test = same generator, seed + 1 ---
    simple = {
        "overlapping": lambda n, sd: Q.random_overlapping(n, sd),
        "scattered": lambda n, sd: Q.random_scattered(n, sd),
        "ndim_random": lambda n, sd: Q.n_dim_random(n, d, sd),
        "ndim_fixed": lambda n, sd: Q.n_dim_fixed_selectivity(n, d, sd),
        "hot": lambda n, sd: Q.hot(data, n, sd),
        "cold": lambda n, sd: Q.cold(data, n, sd),
        "all": lambda n, sd: Q.all_area(data, n, sd),
        "instacart": lambda n, sd: Q.instacart_queries(n, d, sd),
    }
    if m in ("overlapping", "scattered") and d != 2:
        raise SystemExit(f"--mode {m} needs 2-D data")
    if m == "dmv":
        if d != 3:
            raise SystemExit("--mode dmv needs 3 columns: model_year, registration_date, expiration_date")
        ranges = info.get("ranges")
        date_ranges = None
        if ranges:
            cols = list(ranges)
            date_ranges = (tuple(ranges[cols[1]]), tuple(ranges[cols[2]]))
        simple["dmv"] = lambda n, sd: Q.dmv_queries(n, sd, date_ranges)
    if m in simple:
        gen = simple[m]
        return Q.finalize(gen(n_train, s), data), Q.finalize(gen(args.test, s + 1), data)

    # --- sequence modes: test = evenly spaced queries from a seed+1 run ---
    if m == "workflow":
        if d != 2:
            raise SystemExit("--mode workflow needs 2-D data")
        train = Q.finalize(Q.workflow(args.train, s), data)  # train=None -> exact Java loop
        test = Q.evenly_spaced(Q.finalize(Q.workflow(args.train, s + 1), data), args.test)
        return train, test
    if m == "transition":
        n = args.train or 3200
        train = Q.finalize(Q.transition(d, n, s), data)
        return train, Q.evenly_spaced(Q.finalize(Q.transition(d, n, s + 1), data), args.test)
    if m == "shift":
        # Same regions (region_seed=s) for train and test, different queries inside them.
        per_region = (args.train or 100 * args.regions) // args.regions
        train = Q.workload_shift(d, args.regions, per_region, seed=s, region_seed=s)
        test = Q.workload_shift(d, args.regions, max(1, args.test // args.regions),
                                seed=s + 1, region_seed=s)
        return Q.finalize(train, data), Q.finalize(test, data)
    if m == "grid":
        # Java: twoDimPermanentAssertions (strips) to learn from, grid cells to test on.
        if d != 2:
            raise SystemExit("--mode grid needs 2-D data")
        return Q.finalize(Q.grid_strips(args.grid), data), Q.finalize(Q.grid_cells(args.grid), data)
    raise SystemExit(f"unknown mode {m}")


def _stats(name: str, qs: Q.QuerySet) -> str:
    return (f"{name:<6} {len(qs):>6} queries   selectivity  min {qs.sel.min():.6f}   "
            f"mean {qs.sel.mean():.6f}   max {qs.sel.max():.6f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, columns, info = load_data(args, out)
    train, test = make_queries(args, data, info)
    perm = Q.default_query(data.shape[1])

    qio.write_dataset(out / "data.csv", data, columns)
    qio.write_queries(out / "train_queries.txt", train)
    qio.write_queries(out / "test_queries.txt", test)
    qio.write_queries(out / "permanent_queries.txt", perm)
    meta = {"args": vars(args), "columns": columns, "rows": len(data),
            "n_train": len(train), "n_test": len(test), **info}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"mode={args.mode}  data={args.data or DEFAULT_DATA[args.mode]}  "
          f"rows={len(data)}  columns={columns}  seed={args.seed}")
    print(_stats("train", train))
    print(_stats("test", test))
    print(f"files written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
