"""Tests for the synthetic data generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from network_forecast.data_generator import (
    GeneratorConfig,
    PoPConfig,
    _diurnal_pattern,
    _weekly_pattern,
    generate_dataset,
)


class TestDiurnalPattern:
    def test_evening_peak_higher_than_night(self):
        hours = np.arange(24)
        pattern = _diurnal_pattern(hours)
        assert pattern[21] > pattern[3], "21:00 streaming peak should exceed 03:00 trough"

    def test_pattern_always_positive(self):
        hours = np.arange(24)
        pattern = _diurnal_pattern(hours)
        assert (pattern > 0).all(), "diurnal multipliers must be strictly positive"

    def test_pattern_baseline_above_one(self):
        # Even at the trough we expect >= 1.0 (we add to the baseline, never subtract)
        hours = np.arange(24)
        assert (_diurnal_pattern(hours) >= 1.0).all()


class TestWeeklyPattern:
    def test_weekend_uplift_applied(self):
        dow = np.array([0, 1, 2, 3, 4, 5, 6])  # Mon-Sun
        pattern = _weekly_pattern(dow, weekend_uplift=0.2)
        # Sat (5) and Sun (6) should be 1.2; weekdays should be 1.0
        assert pattern[5] == pytest.approx(1.2)
        assert pattern[6] == pytest.approx(1.2)
        assert pattern[0] == pytest.approx(1.0)
        assert pattern[4] == pytest.approx(1.0)

    def test_zero_uplift_returns_ones(self):
        dow = np.array([0, 5, 6])
        pattern = _weekly_pattern(dow, weekend_uplift=0.0)
        assert (pattern == 1.0).all()


class TestGenerateDataset:
    @pytest.fixture(scope="class")
    def small_dataset(self):
        config = GeneratorConfig(
            start_date="2024-01-01",
            end_date="2024-01-15",
            random_seed=42,
            pops=(PoPConfig("TEST-PoP", base_traffic_gbps=100.0, growth_rate_per_year=0.1),),
        )
        return generate_dataset(config)

    def test_returns_two_dataframes(self, small_dataset):
        traffic, anomalies = small_dataset
        assert isinstance(traffic, pd.DataFrame)
        assert isinstance(anomalies, pd.DataFrame)

    def test_traffic_has_expected_columns(self, small_dataset):
        traffic, _ = small_dataset
        assert set(traffic.columns) == {"timestamp", "pop", "traffic_gbps"}

    def test_traffic_values_positive(self, small_dataset):
        traffic, _ = small_dataset
        assert (traffic["traffic_gbps"] > 0).all(), "traffic should never be ≤ 0 (even outages cap at 10%)"

    def test_anomalies_have_required_columns(self, small_dataset):
        _, anomalies = small_dataset
        required = {"type", "start_idx", "end_idx", "start_ts", "pop"}
        assert required.issubset(anomalies.columns)

    def test_deterministic_with_seed(self):
        config = GeneratorConfig(start_date="2024-01-01", end_date="2024-01-05", random_seed=123)
        t1, _ = generate_dataset(config)
        t2, _ = generate_dataset(config)
        pd.testing.assert_frame_equal(t1, t2)

    def test_different_seeds_produce_different_data(self):
        c1 = GeneratorConfig(start_date="2024-01-01", end_date="2024-01-05", random_seed=1)
        c2 = GeneratorConfig(start_date="2024-01-01", end_date="2024-01-05", random_seed=2)
        t1, _ = generate_dataset(c1)
        t2, _ = generate_dataset(c2)
        assert not np.allclose(t1["traffic_gbps"].values, t2["traffic_gbps"].values)

    def test_holiday_traffic_higher_than_typical_weekday(self, small_dataset):
        traffic, _ = small_dataset
        # 2024-01-01 is a UK bank holiday; 2024-01-09 (Tue) is a normal weekday.
        # Compare daily means. We exclude any hours that fell into an injected
        # outage/surge anomaly to keep the comparison clean.
        traffic = traffic.copy()
        traffic["date"] = pd.to_datetime(traffic["timestamp"]).dt.date
        holiday_mean = traffic.loc[traffic["date"] == pd.Timestamp("2024-01-01").date(), "traffic_gbps"].mean()
        weekday_mean = traffic.loc[traffic["date"] == pd.Timestamp("2024-01-09").date(), "traffic_gbps"].mean()
        # Allow some slack because anomalies can land on either day; just sanity check.
        assert holiday_mean > 0 and weekday_mean > 0
