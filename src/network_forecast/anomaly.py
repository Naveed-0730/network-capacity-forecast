"""
Anomaly detection on network telemetry residuals.

Approach: fit a forecast model, compute residuals on a holdout, flag any
observation whose residual exceeds k * IQR-derived threshold from the
median. This is the classical "model + threshold" approach used in NOC
operations — interpretable, fast, no labelled data required.

We *also* support a simple isolation-style scoring as a secondary signal,
but residual-based is the primary detector here because operators trust
"the model expected 200Gbps, we saw 30Gbps" far more than an opaque score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionResult:
    """Per-timestamp anomaly flags with the underlying residual scores."""

    timestamps: pd.DatetimeIndex
    observed: np.ndarray
    expected: np.ndarray
    residual: np.ndarray
    z_score: np.ndarray
    is_anomaly: np.ndarray
    anomaly_type: np.ndarray  # "outage", "surge", or ""

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "observed": self.observed,
            "expected": self.expected,
            "residual": self.residual,
            "z_score": self.z_score,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
        })


class ResidualAnomalyDetector:
    """
    Flag points whose residual is ``threshold`` IQR-scaled MADs from zero.

    Uses median + MAD (median absolute deviation) instead of mean + std
    because residuals on real telemetry have heavy tails and we don't want
    a few past anomalies to inflate the threshold and hide future ones.
    """

    def __init__(self, threshold: float = 3.5) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.threshold = threshold
        self._median: float | None = None
        self._mad: float | None = None

    def fit(self, residuals: np.ndarray) -> "ResidualAnomalyDetector":
        """Calibrate the detector on a clean(ish) holdout of residuals."""
        if len(residuals) < 24:
            raise ValueError(f"need ≥24 residuals to calibrate, got {len(residuals)}")
        self._median = float(np.median(residuals))
        # 1.4826 makes MAD a consistent estimator of σ under normality.
        self._mad = float(np.median(np.abs(residuals - self._median)) * 1.4826)
        if self._mad == 0:
            logger.warning("MAD is zero; falling back to std")
            self._mad = float(np.std(residuals)) or 1.0
        logger.info("Calibrated detector: median=%.3f, MAD=%.3f", self._median, self._mad)
        return self

    def detect(
        self,
        timestamps: pd.DatetimeIndex,
        observed: np.ndarray,
        expected: np.ndarray,
    ) -> AnomalyDetectionResult:
        """Score each timestamp and label outages (low) vs surges (high)."""
        if self._median is None or self._mad is None:
            raise RuntimeError("Detector must be fit() before detect().")
        if not (len(timestamps) == len(observed) == len(expected)):
            raise ValueError("timestamps, observed, expected must be same length")

        residual = observed - expected
        z_score = (residual - self._median) / self._mad
        is_anomaly = np.abs(z_score) > self.threshold
        anomaly_type = np.where(
            is_anomaly & (z_score < 0), "outage",
            np.where(is_anomaly & (z_score > 0), "surge", ""),
        )

        n_outage = int((anomaly_type == "outage").sum())
        n_surge = int((anomaly_type == "surge").sum())
        logger.info("Detected %d outages, %d surges in %d points", n_outage, n_surge, len(observed))

        return AnomalyDetectionResult(
            timestamps=timestamps,
            observed=observed,
            expected=expected,
            residual=residual,
            z_score=z_score,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
        )


def detector_recall(
    detection: AnomalyDetectionResult, ground_truth: pd.DataFrame, tolerance_hours: int = 2
) -> dict[str, float]:
    """
    Score a detection result against injected ground-truth anomalies.

    A ground-truth anomaly counts as "caught" if any flagged anomaly falls
    within ``tolerance_hours`` of its window. Returns recall per type and
    overall — precision is harder to define here because real ops anomalies
    bleed into adjacent hours, so we report recall as the primary metric.
    """
    flagged_ts = pd.to_datetime(detection.timestamps[detection.is_anomaly])
    tol = pd.Timedelta(hours=tolerance_hours)

    caught = {"outage": 0, "surge": 0}
    totals = {"outage": 0, "surge": 0}
    for _, row in ground_truth.iterrows():
        kind = row["type"]
        if kind not in totals:
            continue
        totals[kind] += 1
        window_start = pd.to_datetime(row["start_ts"]) - tol
        # Anomalies in ground truth have implicit duration ~1-8h; widen the
        # match window generously so the "did we notice" question is fair.
        window_end = window_start + pd.Timedelta(hours=12) + tol
        if ((flagged_ts >= window_start) & (flagged_ts <= window_end)).any():
            caught[kind] += 1

    return {
        "outage_recall": caught["outage"] / totals["outage"] if totals["outage"] else float("nan"),
        "surge_recall": caught["surge"] / totals["surge"] if totals["surge"] else float("nan"),
        "overall_recall": (
            (caught["outage"] + caught["surge"]) / (totals["outage"] + totals["surge"])
            if (totals["outage"] + totals["surge"]) else float("nan")
        ),
    }
