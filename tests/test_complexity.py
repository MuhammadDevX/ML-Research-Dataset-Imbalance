import numpy as np

from common.complexity import MAX_ROWS, sample_rows


def test_meta_feature_sample_keeps_every_minority_row():
    y = np.zeros(100_000, dtype=int)
    y[np.random.default_rng(0).choice(len(y), 167, replace=False)] = 1   # Credit Card rate
    idx = sample_rows(y)
    assert len(idx) == MAX_ROWS and len(np.unique(idx)) == MAX_ROWS
    assert y[idx].sum() == 167
    assert np.array_equal(idx, sample_rows(y))   # seeded


def test_meta_feature_sample_small_data_used_whole():
    y = np.r_[np.ones(41), np.zeros(896)].astype(int)   # Oil Spill size
    assert np.array_equal(sample_rows(y), np.arange(len(y)))


def test_meta_feature_sample_falls_back_to_stratified_when_minority_is_large():
    y = np.r_[np.ones(4_000), np.zeros(6_000)].astype(int)
    idx = sample_rows(y)
    assert len(idx) == MAX_ROWS and abs(y[idx].mean() - 0.4) < 0.01
