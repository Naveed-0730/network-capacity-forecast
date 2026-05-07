"""
Synthetic broadband network telemetry generator.

Produces realistic hourly data for multiple Points of Presence (PoPs) with:
- Multiplicative trend (subscriber growth)
- Daily and weekly seasonality (work-from-home + evening streaming peaks)
- Holiday effects (UK bank holidays drive higher residential traffic)
- Injected anomalies (outages, traffic surges) for anomaly-detection testing

Designed to mimic the kind of telemetry a Network Data Scientist at a major
UK ISP / streaming operator would actually work with.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# UK bank holidays for the synthetic period (2023-2024). Hard-coded so the
# generator stays deterministic and offline — no calls to external APIs.
UK_BANK_HOLIDAYS_2023_2024: tuple[str, ...] = (
    "2023-01-02", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-08",
    "2023-05-29", "2023-08-28", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
    "2024-08-26", "2024-12-25", "2024-12-26",
)


@dataclass(frozen=True)
class PoPConfig:
    """Configuration for a single Point of Presence (network site)."""

    name: str
    base_traffic_gbps: float
    growth_rate_per_year: float
    weekend_uplift: float = 0.15
    noise_std: float = 0.04


@dataclass
class GeneratorConfig:
    """Top-level configuration for the synthetic dataset."""

    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    freq: str = "h"
    random_seed: int = 42
    pops: Sequence[PoPConfig] = field(
        default_factory=lambda: (
            PoPConfig("LON-CORE-01", base_traffic_gbps=420.0, growth_rate_per_year=0.18),
            PoPConfig("MAN-EDGE-02", base_traffic_gbps=180.0, growth_rate_per_year=0.22),
            PoPConfig("LDS-EDGE-03", base_traffic_gbps=140.0, growth_rate_per_year=0.20),
            PoPConfig("EDI-EDGE-04", base_traffic_gbps=95.0, growth_rate_per_year=0.16),
        )
    )


def _diurnal_pattern(hour_of_day: np.ndarray) -> np.ndarray:
    """Two daily peaks: morning work-from-home + evening streaming surge."""
    morning = 0.18 * np.exp(-((hour_of_day - 9) ** 2) / 8)
    evening = 0.42 * np.exp(-((hour_of_day - 21) ** 2) / 6)
    return 1.0 + morning + evening


def _weekly_pattern(day_of_week: np.ndarray, weekend_uplift: float) -> np.ndarray:
    """Weekend traffic is higher because people are home longer."""
    return np.where(day_of_week >= 5, 1.0 + weekend_uplift, 1.0)


def _holiday_pattern(timestamps: pd.DatetimeIndex, holidays: Sequence[str]) -> np.ndarray:
    """Bank holidays look like Sundays + ~10% extra residential demand."""
    holiday_dates = pd.to_datetime(holidays).date
    ts_dates = timestamps.date
    is_holiday = np.isin(ts_dates, holiday_dates)
    return np.where(is_holiday, 1.25, 1.0)


def _trend_component(
    timestamps: pd.DatetimeIndex, base: float, growth_per_year: float
) -> np.ndarray:
    """Compounding YoY subscriber growth, applied multiplicatively per hour."""
    seconds_elapsed = (timestamps - timestamps[0]).total_seconds().to_numpy()
    years_elapsed = seconds_elapsed / (365.25 * 24 * 3600)
    return base * (1.0 + growth_per_year) ** years_elapsed


def _inject_anomalies(
    series: np.ndarray,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    n_outages: int = 4,
    n_surges: int = 3,
) -> tuple[np.ndarray, list[dict]]:
    """
    Inject realistic operational anomalies and return their ground-truth labels.

    Outage: traffic collapses to ~10-30% of expected for 1-6 hours.
    Surge: traffic spikes to 150-200% of expected for 2-8 hours
    (e.g., live football, major game release, breaking news).
    """
    series = series.copy()
    labels: list[dict] = []
    n = len(series)

    for _ in range(n_outages):
        start = int(rng.integers(48, n - 24))
        duration = int(rng.integers(1, 6))
        severity = float(rng.uniform(0.10, 0.30))
        series[start : start + duration] *= severity
        labels.append({
            "type": "outage",
            "start_idx": start,
            "end_idx": start + duration,
            "start_ts": timestamps[start],
            "severity": severity,
        })

    for _ in range(n_surges):
        start = int(rng.integers(48, n - 24))
        duration = int(rng.integers(2, 8))
        magnitude = float(rng.uniform(1.5, 2.0))
        series[start : start + duration] *= magnitude
        labels.append({
            "type": "surge",
            "start_idx": start,
            "end_idx": start + duration,
            "start_ts": timestamps[start],
            "magnitude": magnitude,
        })

    return series, labels


def generate_pop_series(
    pop: PoPConfig, timestamps: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict]]:
    """Build a single PoP's traffic series with all components composed."""
    trend = _trend_component(timestamps, pop.base_traffic_gbps, pop.growth_rate_per_year)
    diurnal = _diurnal_pattern(timestamps.hour.values)
    weekly = _weekly_pattern(timestamps.dayofweek.values, pop.weekend_uplift)
    holiday = _holiday_pattern(timestamps, UK_BANK_HOLIDAYS_2023_2024)
    noise = rng.normal(loc=1.0, scale=pop.noise_std, size=len(timestamps))

    clean_series = trend * diurnal * weekly * holiday * noise
    series_with_anomalies, anomaly_labels = _inject_anomalies(clean_series, timestamps, rng)

    for label in anomaly_labels:
        label["pop"] = pop.name

    return series_with_anomalies, anomaly_labels


def generate_dataset(config: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the full multi-PoP dataset and the anomaly ground-truth table.

    Returns
    -------
    traffic_df : long-format DataFrame with columns [timestamp, pop, traffic_gbps]
    anomalies_df : DataFrame of injected anomalies with start, end, type, severity
    """
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.random_seed)

    timestamps = pd.date_range(
        start=config.start_date, end=config.end_date, freq=config.freq, tz="UTC"
    )
    logger.info("Generating %d hourly observations across %d PoPs", len(timestamps), len(config.pops))

    frames: list[pd.DataFrame] = []
    all_anomalies: list[dict] = []

    for pop in config.pops:
        series, anomalies = generate_pop_series(pop, timestamps, rng)
        frames.append(
            pd.DataFrame({
                "timestamp": timestamps,
                "pop": pop.name,
                "traffic_gbps": series,
            })
        )
        all_anomalies.extend(anomalies)

    traffic_df = pd.concat(frames, ignore_index=True)
    anomalies_df = pd.DataFrame(all_anomalies)

    logger.info(
        "Generated %d rows, %d anomalies (%d outages, %d surges)",
        len(traffic_df),
        len(anomalies_df),
        int((anomalies_df["type"] == "outage").sum()) if not anomalies_df.empty else 0,
        int((anomalies_df["type"] == "surge").sum()) if not anomalies_df.empty else 0,
    )

    return traffic_df, anomalies_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    traffic, anomalies = generate_dataset()
    traffic.to_parquet("data/network_traffic.parquet", index=False)
    anomalies.to_csv("data/anomalies_ground_truth.csv", index=False)
    print(f"Wrote {len(traffic):,} rows to data/network_traffic.parquet")
