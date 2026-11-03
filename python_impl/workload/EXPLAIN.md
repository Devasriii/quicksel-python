# How my part works (plain-language guide)

## The big picture

QuickSel learns **how data is distributed** by watching queries and their answers, without
scanning the table (paper Section 1–2). To learn it needs examples like:

> "The query `0.2 ≤ x ≤ 0.4 AND 0.5 ≤ y ≤ 0.9` matched 3.1% of the rows."

Each example is a **range query** (a rectangle, or a box in more dimensions) plus its
**selectivity** (the fraction of rows inside it). The paper calls these *observed queries*,
and the Java code calls them `Assertion`s (`Assertion.java` = a `Query` + a `freq`).

My generator makes:
1. a **dataset** (a table of numbers between 0 and 1),
2. a list of **training queries** with their true selectivities (what QuickSel learns from),
3. a separate list of **test queries** (to measure how good QuickSel's guesses are, using
   RMS error, paper Section 5.1).

## Files, one by one

### `java_random.py`: a copy of Java's random number generator
The Java code makes its rectangles with `new Random(0)`. Python's random numbers are
different, so I copied Java's generator formula (`seed = seed × 0x5DEECE66D + 11`,
keeping 48 bits). Now `JavaRandom(0).next_double()` returns `0.730967787376657`,
exactly like Java, so our rectangles are **exactly the same** as the original code's.

### `data.py`: making and loading datasets
| My function | Java/Python 2 original | What it does |
|---|---|---|
| `two_dim_gaussian(size)` | `twoDimGaussianDataset(size)` | 2 columns, bell-shaped around (0.5, 0.5), fixed covariance |
| `two_dim_gaussian(size, correlation=c)` | `twoDimGaussianDataset(correlation, size)` | same, but you choose how correlated x and y are |
| `n_dim_gaussian(n, size)` | `nDimGaussianDataset(n, size)` | n independent bell-shaped columns |
| `three_dim_gaussian(size)` | `gaussian_data.py: gen_gaussian_3d` | the 3-D data the old Python 2 script used |
| `load_csv_dataset(path, columns)` | (not in repo) | reads a real CSV, turns dates into day numbers and text into category numbers, drops empty rows, then squeezes each column into [0,1] with `(x − min) / (max − min)` |
| `make_demo_dmv_csv`, `make_demo_instacart_csv` | (not in repo) | **fake** CSVs with the real column names, for testing before we get the real files |

**One change from Java:** a bell curve sometimes produces numbers below 0 or above 1.
Java kept them; I redraw them (`_sample_in_unit_box`). Everything then lives inside the
[0,1] box, so "no restriction on this column" can simply be written as `0,1`.

### `selectivity.py`: the true answer of every query
`true_selectivity(data, lo, hi)` checks, for each query, which rows satisfy
`lo ≤ x ≤ hi` in every column, and returns the fraction. It is the fast version of Java's
`countSatisfyingItems` / `Query.doesSatisfy`. Instead of a Python loop over rows, numpy
compares a whole block of queries against all rows at once ("vectorization"). The
boundary counts as inside, just like Java.

### `queries.py`: the query generators (the heart of my part)
Every generator returns the rectangles' lower corners `lo` and upper corners `hi`.
`finalize(...)` then clamps them to [0,1], rounds to 6 decimals and computes selectivity.

| My function | Original (file) | Idea | Paper |
|---|---|---|---|
| `random_overlapping` | `twoDimGaussianDatasetRandomAssertions` (Java) | big boxes that all cover the middle and overlap a lot | Fig 2, Case 1 |
| *(same, seed 1)* | `twoDimGaussianDatasetRandomTestQueries` (Java) | the test version: Java used `Random(1)` | |
| `random_scattered` | `twoDimGaussianDatasetRandomAssertions2` (Java) | small 0.1×0.1 squares scattered around | Fig 2, Case 2 |
| `n_dim_random` | `nDimRandomAssertions` (Java) | width-0.5 boxes in d dimensions | §5.3 (dimensions) |
| `n_dim_fixed_selectivity` | `nDimRandomAssertionsSelectivityFixed` (Java) | boxes whose volume is always 0.1 | |
| `workflow` | `workflowAssertion` (Java) | a box sliding step by step along the diagonal | |
| `hot` / `cold` / `all_area` | `gen_hot` / `gen_cold` / `gen_all` → `gen_query` (Python 2) | small boxes only in dense / only in sparse / anywhere | |
| `transition` | `gen_trans` → `gen_query_trans` (Python 2) | a box slowly drifting, a gradually changing workload | |
| `workload_shift` | (new) | 100 queries in one area, then a sudden jump to another area | §5.6, Fig 7 |
| `default_query` | `twoDimDefaultAssertion` / `nDimDefaultAssertion` (Java) | the whole domain, selectivity 1.0 (tells QuickSel "everything sums to 1") | §3 |
| `grid_strips` | `twoDimPermanentAssertions` (Java) | thin strips that restrict one column | |
| `grid_cells` | `twoDimGaussianDatasetGridTestQueries` (Java) | every cell of a grid | |
| `dmv_queries` | (not in repo; rebuilt from `assertion_dmv.txt`) | "registrations valid on date T for cars from a model-year window" | §5.1, Fig 3 |
| `instacart_queries` | (not in repo; rebuilt from `assertion_instacart.txt`) | 0.5-wide boxes on hour-of-day × days-since-prior | §5.1, Fig 3 |
| `_is_cold` | `check_cold` (Python 2) | "is this spot sparse?" | |
| `evenly_spaced` | (new) | picks test queries spread over a sequence | |
| `QuerySet` | `Assertion` list | holds `lo`, `hi`, `sel` arrays | |

How I rebuilt the DMV/Instacart generators: I measured the existing files. Every DMV line
has a 0.2-wide model_year window, and the two date columns always share one number t
(`[t,1]` and `[0,t]`). Every Instacart box is exactly 0.5 wide.

### `io.py`: reading and writing files
| My function | Original | What it does |
|---|---|---|
| `write_queries` | (Java only reads) | writes `lo1,hi1,...,loN,hiN,sel` lines |
| `read_queries` | `AssertionReader.readAssertion(file)` | reads them back; clamps to [0,1] like `checkInBoundary` |
| `read_repo_assertions` | `readAssertion(file, permanentFile)` + the 1000/100 split in `DMVSpeedComparison.java` | reads the repo's DMV/Instacart query files |
| `write_dataset` / `read_dataset` | (new) | saves/loads `data.csv` |

### `generate.py`: the command-line tool
It ties everything together: it builds the data, makes train and test queries, writes all
files and prints a summary. Run
`python -m python_impl.workload.generate --help` to see every option. HANDOFF.md has
the full table.

### `plot_queries.py`: the pictures for the presentation
It draws the data as grey dots and every query as a blue rectangle.
`--figure2` makes the side-by-side "Case 1 vs Case 2" image like paper Figure 2.

### `tests/`: automatic checks (run `python -m pytest python_impl/workload/tests -q`)
- `test_selectivity.py`: on a 5-point dataset I know the exact answers; boundary points count; the whole-domain query = 1.0.
- `test_queries.py`: every generator gives `lo ≤ hi`, bounds and selectivities in [0,1]; the same seed gives the same queries; the Java RNG gives Java's numbers.
- `test_io_and_data.py`: write → read gives the same queries; the repo files load (1100 lines); the CSV loader converts dates and text correctly.
- `test_generate.py`: every mode runs end to end, and the selectivity in each file matches a recomputation from `data.csv`.

## Things I noticed in the original code (good to mention in the presentation)

1. **Java tests on its training queries.** `SpeedComparison.java` and `PerAttAndSamplingTest.java`
   generate training and test queries with the same seed, so the "test" queries are the
   training queries. The paper says test queries were *not* used for training. I use
   `seed + 1` for test, so they really are different.
2. **Comments vs code.** In `nDimRandomAssertions` the comment says width `0.1^(1/d)` but
   the code uses 0.5. In `workflowAssertion` the comment says `randn()` (bell-curve noise)
   but the code uses `nextDouble()` (uniform). I copied the **code**, not the comments.
3. **Unseeded randomness.** Java's datasets and `workflowAssertion`, and all of the Python 2
   script, had no fixed seed, so every run was different. Mine are all seeded.
4. **"Hot" ≈ "all".** The Python 2 rule only calls a spot "cold" when it is very empty
   (fewer than 1/8 of the points uniform data would have there). On the Gaussian most spots
   count as hot, so `hot` and `all` give similar queries. This is the original logic, kept as is.
5. **DMV column order.** In `assertion_dmv.txt` the date columns appear as `[t,1]` then `[0,t]`,
   which reads as "expiration ≥ t, registration ≤ t", so that file seems to list expiration
   first. I keep the paper's order (model_year, registration_date, expiration_date).

## Words you may get asked about
- **Selectivity**: fraction of rows a query returns (0 = none, 1 = all).
- **Normalize**: rescale a column so its smallest value is 0 and its largest is 1.
- **Seed**: the starting number of a random generator; the same seed gives the same "random" numbers.
- **RMS error**: √(average of (true − estimated)²) over the test queries, the paper's accuracy metric.
- **Vectorization**: letting numpy process whole arrays at once instead of a Python `for` loop (much faster).
