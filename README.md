# QuickSel in Python 3

A university re-implementation of **"QuickSel: Quick Selectivity Learning with Mixture Models"**
(Park, Zhong, Mozafari, SIGMOD 2020, [arXiv:1812.10568](https://arxiv.org/abs/1812.10568)),
based on the authors' original Java repository.

## Status

| Pipeline step | Folder | Status |
|---|---|---|
| 1. Query workload generator | [`python_impl/workload/`](python_impl/workload/) | done, 57 tests passing |
| 2+. Model, training, evaluation | (teammates) | in progress |

## Quick start

```bash
pip install -r python_impl/workload/requirements.txt
python -m pytest python_impl/workload/tests -q
python -m python_impl.workload.generate --mode scattered --rows 10000 --train 1000 --test 100 --seed 0 --out output/scattered
```

This writes `data.csv`, `train_queries.txt`, `test_queries.txt` and `permanent_queries.txt` to `output/scattered/`.
Always run from this folder with `python -m ...`.

## Docs

- [`python_impl/workload/HANDOFF.md`](python_impl/workload/HANDOFF.md): for teammates. All options, the file format,
  how to load the files, and which mode to use for which paper figure.
- [`python_impl/workload/EXPLAIN.md`](python_impl/workload/EXPLAIN.md): beginner guide. Every file and function, mapped to
  the original Java/Python 2 code and to the paper.

## Credits

The query generators are ported from the original QuickSel repository (Apache License 2.0).
The files in `python_impl/workload/resources/` are copied unchanged from that repository's
`test/java/edu/illinois/quicksel/resources/`.
