"""Draw 2-D queries as rectangles over the data points (like paper Figure 2).

Plot a generated workload (reads data.csv + train_queries.txt from a folder):
    python -m python_impl.workload.plot_queries --out output/scattered --n 50

Make the two-panel "Case 1 / Case 2" picture like paper Figure 2:
    python -m python_impl.workload.plot_queries --figure2 --png output/figure2_cases.png

For data with more than 2 columns, --cols picks which two to draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import data as D
from . import io as qio
from . import queries as Q

# Colors: neutral ink for the data, one accent blue for the queries.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
POINTS = "#9a9893"
QUERY = "#2a78d6"


def draw(ax: plt.Axes, data: np.ndarray, qs: Q.QuerySet, title: str,
         cols: tuple[int, int] = (0, 1), max_points: int = 3000) -> None:
    """Scatter the data (a random subset, so it stays readable) and outline each query."""
    i, j = cols
    rng = np.random.default_rng(0)
    pts = data[rng.choice(len(data), min(max_points, len(data)), replace=False)]
    ax.scatter(pts[:, i], pts[:, j], s=3, color=POINTS, alpha=0.5, linewidths=0, zorder=1)
    for lo, hi in zip(qs.lo, qs.hi):
        ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j],
                               facecolor=QUERY, alpha=0.06, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j],
                               facecolor="none", edgecolor=QUERY, linewidth=1.0, zorder=3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")


def _save(fig: plt.Figure, png: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {png.resolve()}")


def plot_folder(folder: Path, n: int, png: Path, query_file: str, cols: tuple[int, int]) -> None:
    """Plot the first n queries of a folder written by generate.py."""
    data, names = qio.read_dataset(folder / "data.csv")
    qs = qio.read_queries(folder / query_file).take(slice(0, n))
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    draw(ax, data, qs, f"{folder.name}: first {len(qs)} queries ({query_file})", cols)
    ax.set_xlabel(names[cols[0]], color=INK_SECONDARY)
    ax.set_ylabel(names[cols[1]], color=INK_SECONDARY)
    _save(fig, png)


def plot_figure2(png: Path, n: int = 30, rows: int = 5000, seed: int = 0) -> None:
    """Paper Figure 2: (a) highly-overlapping vs (b) scattered query workloads."""
    data = D.two_dim_gaussian(rows, seed=seed)
    case1 = Q.finalize(Q.random_overlapping(n, seed), data)
    case2 = Q.finalize(Q.random_scattered(n, seed), data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=SURFACE)
    draw(axes[0], data, case1, f"(a) Case 1: highly-overlapping ({n} queries)")
    draw(axes[1], data, case2, f"(b) Case 2: scattered ({n} queries)")
    fig.suptitle("Query workloads over a 2-D Gaussian dataset", color=INK, x=0.06,
                 ha="left", fontsize=13)
    _save(fig, png)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m python_impl.workload.plot_queries")
    p.add_argument("--out", help="folder written by generate.py")
    p.add_argument("--n", type=int, default=50, help="number of queries to draw (default 50)")
    p.add_argument("--file", default="train_queries.txt", help="query file inside --out")
    p.add_argument("--cols", type=int, nargs=2, default=(0, 1), help="two columns to draw")
    p.add_argument("--png", help="output image (default: <out>/queries.png)")
    p.add_argument("--figure2", action="store_true", help="make the Figure 2 style picture")
    args = p.parse_args(argv)

    if args.figure2:
        plot_figure2(Path(args.png or "output/figure2_cases.png"), n=min(args.n, 30))
    elif args.out:
        folder = Path(args.out)
        plot_folder(folder, args.n, Path(args.png or folder / "queries.png"), args.file,
                    tuple(args.cols))
    else:
        p.error("give --out FOLDER or --figure2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
