# Query workload generator: handoff

This is step 1 of our QuickSel pipeline. It produces **datasets** and **query workloads**
(range queries plus their true selectivities) in the same file format the original repo
uses. Training, estimation and evaluation code can read these files directly.

## Install and test

```bash
pip install -r python_impl/workload/requirements.txt   # numpy, pandas (+ matplotlib, pytest)
python -m pytest python_impl/workload/tests -q          # 57 tests, ~3 s
```

Run every command from the project root (the folder that contains `python_impl/`).
Always use `python -m ...`.

## Run it

```bash
python -m python_impl.workload.generate --mode scattered --rows 10000 \
    --train 1000 --test 100 --seed 0 --out output/scattered
```

Output:

```
mode=scattered  data=gaussian2d  rows=10000  columns=['x0', 'x1']  seed=0
train    1000 queries   selectivity  min 0.000400   mean 0.011454   max 0.024300
test      100 queries   selectivity  min 0.000600   mean 0.011591   max 0.023200
```

Files written to `--out`:

| File | Contents |
|---|---|
| `data.csv` | dataset, header = column names, every value in [0,1] (6 decimals) |
| `train_queries.txt` | training ("observed") queries, in order |
| `test_queries.txt` | test queries, **never** used for training |
| `permanent_queries.txt` | one line: whole domain `0,1,...,0,1,1.0`. Add it to QuickSel as the permanent assertion |
| `meta.json` | all options used, column names, rows, and for CSV data the original min/max of each column |

### Options

| Option | Default | Meaning |
|---|---|---|
| `--mode` | (required) | query generator, see the table below |
| `--data` | depends on mode | `gaussian2d`, `gaussian3d`, `gaussiannd`, `csv` |
| `--rows` | 10000 (synthetic) / all rows (csv) | dataset size; for csv, a random sample of this many rows |
| `--train` | 1000 (exceptions below) | number of training queries |
| `--test` | 100 | number of test queries |
| `--seed` | 0 | dataset + training queries use `seed`, test queries use `seed+1` |
| `--out` | `output` | output folder |
| `--corr` | none | correlation for `gaussian2d`; none = Java's fixed covariance `[[.09,.04],[.04,.09]]` |
| `--dims` | 4 | columns for `gaussiannd` |
| `--csv` | auto demo file | CSV path for `--data csv`; dmv/instacart create a **fake** demo CSV if omitted |
| `--columns` | paper columns | CSV columns to use, in this order |
| `--grid` | 10 | grid size for `--mode grid` |
| `--regions` | 3 | regions for `--mode shift` |

### Modes

| Mode | Default data | Queries | Origin |
|---|---|---|---|
| `overlapping` | gaussian2d | each side `[U·0.5, U·0.5+0.5]`, big and heavily overlapping (paper Fig 2, Case 1) | `twoDimGaussianDatasetRandomAssertions` |
| `scattered` | gaussian2d | random 0.1×0.1 squares (paper Fig 2, Case 2) | `twoDimGaussianDatasetRandomAssertions2` |
| `ndim_random` | gaussiannd | each side `[U·0.5, U·0.5+0.5]` in all d columns | `nDimRandomAssertions` |
| `ndim_fixed` | gaussiannd | each side has width `0.1^(1/d)` (box volume 0.1) | `nDimRandomAssertionsSelectivityFixed` |
| `workflow` | gaussian2d | a 0.2×0.2 box sliding along the diagonal; default = Java's exact loop (1000 queries) | `workflowAssertion` |
| `hot` / `cold` / `all` | gaussian3d | small boxes (sides 0.05–0.15) only in dense / sparse / any areas | `gen_gaussian_queries.py` |
| `transition` | gaussian3d | a small box whose centre slowly moves from 0.49 to 0.09 (default 3200 queries) | `gen_query_trans` |
| `shift` | gaussian2d | 100 queries per region, then a sudden jump to a new region (paper §5.6, Fig 7) | new |
| `grid` | gaussian2d | train = 2·grid one-column strips, test = grid² cells | `twoDimPermanentAssertions` / `...GridTestQueries` |
| `dmv` | csv | model_year window of 0.2, registration ≤ T ≤ expiration (valid on date T) | rebuilt from `assertion_dmv.txt` |
| `instacart` | csv | 0.5-wide box in each column, centre uniform | rebuilt from `assertion_instacart.txt` |

For the sequence modes (`workflow`, `transition`, `shift`), the test set is `--test` queries
taken at evenly spaced positions from a second run with `seed+1`. For `shift`, the test set
uses the same regions, with `--test // regions` queries per region, in region order
(so 99 for the defaults).

## File format (exactly the repo's)

One query per line, no header, comma-separated:

```
lo1,hi1,lo2,hi2,...,loN,hiN,selectivity
0.657871,0.757871,0.216483,0.316483,0.007800000
```

- A query = rows where `lo_i <= x_i <= hi_i` for **every** column i (closed intervals, as in `Query.java`).
- All bounds are in [0,1]. A column the query doesn't restrict is written `0.000000,1.000000`.
- `selectivity` = fraction of rows of `data.csv` inside the box. It was computed on the
  rounded bounds and the rounded data, so recomputing it from the files gives the same number.
- The Java `AssertionReader.readAssertion(file)` can read these files too.

## Loading in Python

```python
from python_impl.workload import io as qio
from python_impl.workload.selectivity import true_selectivity

train = qio.read_queries("output/scattered/train_queries.txt")
test  = qio.read_queries("output/scattered/test_queries.txt")
perm  = qio.read_queries("output/scattered/permanent_queries.txt")
data, columns = qio.read_dataset("output/scattered/data.csv")

train.lo, train.hi   # numpy arrays, shape (Q, d)
train.sel            # numpy array, shape (Q,)
first_100 = train.take(slice(0, 100))     # "observe" only the first 100 queries

rms = ((my_estimates - test.sel) ** 2).mean() ** 0.5   # paper metric (×100 for %)
```

Without our package:

```python
import numpy as np
a = np.loadtxt("output/scattered/train_queries.txt", delimiter=",", ndmin=2)
lo, hi, sel = a[:, 0:-1:2], a[:, 1:-1:2], a[:, -1]
```

To read the **original paper query files** that ship with the repo (copies are in `workload/resources/`):

```python
train, test, perm = qio.read_repo_assertions("DMV")        # or "Instacart"
# train = lines 0..999, test = lines 1000..1099: the same split as DMVSpeedComparison.java
```

## Which file for which experiment

| Experiment | What to use |
|---|---|
| **Fig 2**: overlapping vs scattered | `--mode overlapping` and `--mode scattered`; picture: `python -m python_impl.workload.plot_queries --figure2` |
| **Fig 3**: QuickSel vs query-driven histograms, DMV and Instacart, observe 10…1000 queries, 100 test | Use the repo files `qio.read_repo_assertions("DMV")` / `("Instacart")`: these are the paper's actual queries, but the repo has no data rows for them. When we get the real CSVs: `--mode dmv --csv <file> --columns <3 cols> --train 1000 --test 100` (and the same for `instacart`), then use `train.take(slice(0, k))` for k = 10, 20, … |
| **Fig 4**: QuickSel vs AutoHist/AutoSample | **Not fully covered.** 4(c) Gaussian, dimensions 2–10: approximate with `--mode ndim_random --dims N` (the paper restricts a random N/2 of the columns, which we didn't build). 4(a) needs data that changes (new tuples with correlation 0.1, 0.2, … every 100 queries), which isn't built either. Both are extra work. |
| **Fig 7**: sudden workload shifts at #100 and #200 | `--mode shift --regions 3 --train 300 --test 99`. Feed `train_queries.txt` in order; after each query, measure error on the test queries of the current region (test lines `[33k, 33k+33)` for region k). |
| `SpeedComparison.java` | `--mode scattered --rows 10000 --train 500 --test 1000` |
| `PerAttAndSamplingTest.java` | `--mode overlapping --corr 0.5 --rows 10000 --train 1000 --test 1000` |

## Things to know

- **Same seed → same files.** With `--seed 0` the training rectangles of `overlapping`,
  `scattered`, `ndim_*` and `workflow` are bit-for-bit the Java ones (we re-implemented
  `java.util.Random`). The overlapping test set equals Java's `twoDimGaussianDatasetRandomTestQueries`.
  The Gaussian **data** can't match Java (Java's sampler is unseeded), so it uses numpy with `--seed`.
- **Truncated Gaussian:** we redraw Gaussian points that fall outside [0,1], so all data is
  in [0,1] and the permanent query really has selectivity 1.0. Java kept those points.
- **Train ≠ test:** the Java `SpeedComparison`/`PerAttAndSamplingTest` accidentally test on
  their training queries (same seed). We always use `seed+1` for test.
- **Demo CSVs are fake.** Until we have the real files, `dmv`/`instacart` run on generated
  demo data with the right column names. The Instacart `orders.csv` already has `order_hour_of_day`
  and `days_since_prior`. In the NY DMV file the three columns probably have other names
  (e.g. "Model Year", "Reg Valid Date", "Reg Expiration Date"), so pass them with `--columns`
  in the order model_year, registration, expiration.
- **Speed:** selectivity is vectorized; 1000 queries on 10k rows takes well under a second.
  On the full DMV data (~12M rows) expect a few minutes, or use `--rows 1000000`.
