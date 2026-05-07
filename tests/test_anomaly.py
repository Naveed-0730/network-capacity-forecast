"""Tests for the residual anomaly detector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from network_forecast.anomaly import ResidualAnomalyDetector


class TestResidualAnomalyDetector:
    def test_fit_calibrates_median_and_mad(self):
        rng = np.random.default_rng(0)
        residuals = rng.normal(0, 1, 1000)
        detector = ResidualAnomalyDetector(threshold=3.0).fit(residuals)
        assert detector._median is not None
        assert detector._mad is not None
        assert abs(detector._median) < 0.2
        assert 0.8 < detector._mad < 1.2

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ResidualAnomalyDetector(threshold=0)
        with pytest.raises(ValueError, match="positive"):
            ResidualAnomalyDetector(threshold=-1)

    def test_too_few_residuals_raises(self):
        with pytest.raises(ValueError, match="≥24"):
            ResidualAnomalyDetector().fit(np.array([1.0, 2.0, 3.0]))

    def test_detect_before_fit_raises(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        with pytest.raises(RuntimeError, match="fit"):
            ResidualAnomalyDetector().detect(idx, np.zeros(10), np.zeros(10))

    def test_detects_obvious_outage(self):
        rng = np.random.default_rng(0)
        # Calibrate on clean residuals
        clean_residuals = rng.normal(0, 1, 500)
        detector = ResidualAnomalyDetector(threshold=3.0).fit(clean_residuals)

        # Now feed a series with one obvious outage
        idx = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
        expected = np.full(50, 100.0)
        observed = expected.copy()
        observed[25] = 10.0  # massive drop = outage

        result = detector.detect(idx, observed, expected)
        assert result.is_anomaly[25]
        assert result.anomaly_type[25] == "outage"

    def test_detects_obvious_surge(self):
        rng = np.random.default_rng(0)
        detector = ResidualAnomalyDetector(threshold=3.0).fit(rng.normal(0, 1, 500))

        idx = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
        expected = np.full(50, 100.0)
        observed = expected.copy()
        observed[10] = 200.0  # huge spike = surge

        result = detector.detect(idx, observed, expected)
        assert result.is_anomaly[10]
        assert result.anomaly_type[10] == "surge"

    def test_no_false_positives_on_clean_data(self):
        rng = np.random.default_rng(0)
        detector = ResidualAnomalyDetector(threshold=3.5).fit(rng.normal(0, 1, 1000))

        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        expected = np.full(200, 100.0)
        observed = expected + rng.normal(0, 1, 200)  # clean noise only

        result = detector.detect(idx, observed, expected)
        # Allow up to 1% false-positive rate at threshold=3.5
        assert result.is_anomaly.mean() < 0.02

    def test_length_mismatch_raises(self):
        rng = np.random.default_rng(0)
        detector = ResidualAnomalyDetector().fit(rng.normal(0, 1, 100))
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        with pytest.raises(ValueError, match="same length"):
            detector.detect(idx, np.zeros(10), np.zeros(5))
