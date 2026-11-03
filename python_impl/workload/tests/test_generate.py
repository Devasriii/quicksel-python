"""End-to-end: every CLI mode runs and writes valid files."""

import numpy as np
import pytest

from python_impl.workload import generate, io as qio
from python_impl.workload.selectivity import true_selectivity


@pytest.mark.parametrize("mode", generate.MODES)
def test_every_mode_writes_valid_files(mode, tmp_path):
    out = tmp_path / mode
    assert generate.main(["--mode", mode, "--rows", "800", "--train", "60", "--test", "12",
                          "--out", str(out)]) == 0
    data, _ = qio.read_dataset(out / "data.csv")
    for name in ("train_queries.txt", "test_queries.txt", "permanent_queries.txt"):
        qs = qio.read_queries(out / name, clamp=False)
        assert qs.dims == data.shape[1]
        assert (qs.lo <= qs.hi).all() and qs.lo.min() >= 0 and qs.hi.max() <= 1
        assert ((qs.sel >= 0) & (qs.sel <= 1)).all()
        # Selectivity in the file = selectivity recomputed from data.csv.
        np.testing.assert_allclose(true_selectivity(data, qs.lo, qs.hi), qs.sel, atol=1e-9)
    assert qio.read_queries(out / "permanent_queries.txt").sel[0] == 1.0
    assert (out / "meta.json").exists()


def test_train_and_test_are_different(tmp_path):
    generate.main(["--mode", "overlapping", "--rows", "500", "--train", "100", "--test", "100",
                   "--out", str(tmp_path)])
    tr = qio.read_queries(tmp_path / "train_queries.txt")
    te = qio.read_queries(tmp_path / "test_queries.txt")
    assert not np.array_equal(tr.lo, te.lo)
