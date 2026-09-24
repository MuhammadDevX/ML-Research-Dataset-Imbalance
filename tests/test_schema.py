import pytest

from common import io


def _row(**over):
    row = {c: 0 for c in io.SCHEMA}
    row.update(config_id="c0_none", classifier="LR", weight_mode="none", ir_level="",
               best_params_json="{}", git_commit="x", package_versions="x", timestamp="x")
    row.update(over)
    return row


def test_append_rejects_wrong_columns(tmp_path):
    path = tmp_path / "r.csv"
    bad = _row()
    bad.pop("macro_f1")
    with pytest.raises(ValueError):
        io.append_row(path, bad)
    with pytest.raises(ValueError):
        io.append_row(path, _row(extra_col=1))


def test_resume_keys_round_trip_through_csv(tmp_path):
    path = tmp_path / "r.csv"
    mem = _row(repeat=2, fold=1, weight_power=1.0, sampling_ratio=0.0, C=1)
    io.append_row(path, mem)
    assert io.row_key(mem) in io.done_keys(path)
    assert io.row_key(_row(repeat=3, fold=1)) not in io.done_keys(path)


def test_read_results_checks_column_order(tmp_path):
    path = tmp_path / "r.csv"
    io.append_row(path, _row())
    assert list(io.read_results(path).columns) == io.SCHEMA
