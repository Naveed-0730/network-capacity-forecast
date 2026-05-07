"""Tests for the evaluation module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from network_forecast.evaluation import mae, mape, rmse, smape, walk_forward_eval
from network_forecast.forecasters import SeasonalNaiveForecaster


class TestMetrics:
    def test_mae_zero_for_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_rmse_zero_for_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmse(y, y) == 0.0

    def test_mae_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # errors: 1, 0, 1 → mean = 2/3
        assert mae(y_true, y_pred) == pytest.approx(2 / 3)

    def test_rmse_penalises_large_errors_more(self):
        y_true = np.array([0.0, 0.0])
        small_errors = np.array([1.0, 1.0])
        big_errors = np.array([0.0, 2.0])  # same MAE, bigger RMSE
        assert mae(y_true, small_errors) == mae(y_true, big_errors)
        assert rmse(y_true, big_errors) > rmse(y_true, small_errors)

    def test_mape_handles_near_zero(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([1.0, 110.0])
        # Should not blow up; eps protects the zero
        assert np.isfinite(mape(y_true, y_pred))

    def test_smape_bounded(self):
        # sMAPE is in [0, 200]; check we don't exceed this on extreme inputs
        y_true = np.array([0.0, 1e9])
        y_pred = np.array([1e9, 0.0])
        assert 0 <= smape(y_true, y_pred) <= 200

    def test_smape_zero_for_perfect_prediction(self):
        y = np.array([5.0, 10.0, 15.0])
        assert smape(y, y) == pytest.approx(0.0)


class TestWalkForward:
    @pytest.fixture
    def long_series(self):
        idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
        rng = np.random.default_rng(0)
        values = 100 + 20 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 1, len(idx))
        return pd.Series(values, index=idx)

    def test_returns_one_row_per_fold(self, long_series):
        result = walk_forward_eval(long_series, SeasonalNaiveForecaster, horizon=24, n_folds=3)
        assert len(result) == 3

    def test_fold_indices_are_distinct(self, long_series):
        result = walk_forward_eval(long_series, SeasonalNaiveForecaster, horizon=24, n_folds=3)
        assert sorted(result["fold"].tolist()) == [0, 1, 2]

    def test_metrics_columns_present(self, long_series):
        result = walk_forward_eval(long_series, SeasonalNaiveForecaster, horizon=24, n_folds=2)
        assert {"mae", "rmse", "mape", "smape"}.issubset(result.columns)

    def test_unsorted_index_raises(self, long_series):
        shuffled = long_series.sample(frac=1, random_state=0)
        with pytest.raises(ValueError, match="sorted"):
            walk_forward_eval(shuffled, SeasonalNaiveForecaster, horizon=24, n_folds=2)

    def test_initial_train_too_small_raises(self, long_series):
        with pytest.raises(ValueError, match="initial_train_size"):
            walk_forward_eval(
                long_series, SeasonalNaiveForecaster,
                horizon=24, n_folds=100,  # 100*24 > len(series)
            )
