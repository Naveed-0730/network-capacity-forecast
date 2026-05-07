"""Tests for the forecasting models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from network_forecast.forecasters import (
    LightGBMForecaster,
    NotFittedError,
    SeasonalNaiveForecaster,
    build_forecaster,
)


@pytest.fixture
def hourly_series():
    """Two weeks of synthetic hourly data with a clean weekly pattern."""
    idx = pd.date_range("2024-01-01", periods=336, freq="h", tz="UTC")
    # Weekly pattern: y_t = 100 + 30*sin(2π*hour/24) + 10*(weekend)
    rng = np.random.default_rng(0)
    values = (
        100
        + 30 * np.sin(2 * np.pi * idx.hour / 24)
        + 10 * (idx.dayofweek >= 5).astype(float)
        + rng.normal(0, 2, len(idx))
    )
    return pd.Series(values, index=idx, name="traffic")


class TestSeasonalNaive:
    def test_fit_returns_self(self, hourly_series):
        model = SeasonalNaiveForecaster()
        assert model.fit(hourly_series) is model

    def test_predict_horizon_length(self, hourly_series):
        model = SeasonalNaiveForecaster().fit(hourly_series)
        result = model.predict(horizon=48)
        assert len(result.point_forecast) == 48
        assert len(result.timestamps) == 48

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            SeasonalNaiveForecaster().predict(horizon=24)

    def test_predict_repeats_last_season(self, hourly_series):
        model = SeasonalNaiveForecaster(season_length=168).fit(hourly_series)
        result = model.predict(horizon=168)
        np.testing.assert_array_equal(result.point_forecast, hourly_series.values[-168:])

    def test_predict_horizon_longer_than_season(self, hourly_series):
        model = SeasonalNaiveForecaster(season_length=168).fit(hourly_series)
        result = model.predict(horizon=200)
        assert len(result.point_forecast) == 200
        # First 168 should match the last season
        np.testing.assert_array_equal(result.point_forecast[:168], hourly_series.values[-168:])

    def test_too_short_series_raises(self):
        short = pd.Series(
            np.arange(50.0),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        with pytest.raises(ValueError, match="≥168"):
            SeasonalNaiveForecaster().fit(short)

    def test_nan_in_series_raises(self, hourly_series):
        polluted = hourly_series.copy()
        polluted.iloc[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            SeasonalNaiveForecaster().fit(polluted)

    def test_non_datetime_index_raises(self):
        bad = pd.Series(np.arange(200.0))  # default RangeIndex
        with pytest.raises(TypeError, match="DatetimeIndex"):
            SeasonalNaiveForecaster().fit(bad)


class TestLightGBM:
    def test_fit_and_predict(self, hourly_series):
        model = LightGBMForecaster(num_boost_round=20).fit(hourly_series)
        result = model.predict(horizon=24)
        assert len(result.point_forecast) == 24
        # Forecast should be in roughly the right range
        assert 50 < result.point_forecast.mean() < 200

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            LightGBMForecaster().predict(horizon=24)


class TestForecasterFactory:
    @pytest.mark.parametrize("name", ["seasonal_naive", "sarima", "lightgbm"])
    def test_build_known(self, name):
        model = build_forecaster(name)
        assert model.name == name

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown forecaster"):
            build_forecaster("not_a_real_model")
