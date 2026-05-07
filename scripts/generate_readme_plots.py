"""Generate static PNGs for the README — gives the GitHub page visual punch."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_forecast.anomaly import ResidualAnomalyDetector
from network_forecast.forecasters import LightGBMForecaster, SeasonalNaiveForecaster

DOCS = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})


def plot_traffic_sample(traffic: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    cutoff = traffic["timestamp"].max() - pd.Timedelta(days=21)
    recent = traffic[traffic["timestamp"] >= cutoff]
    for pop, group in recent.groupby("pop"):
        ax.plot(group["timestamp"], group["traffic_gbps"], label=pop, linewidth=1.0, alpha=0.85)
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic (Gbps)")
    ax.set_title("Synthetic broadband telemetry — last 3 weeks across 4 PoPs")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "01_telemetry_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_forecast_vs_actual(traffic: pd.DataFrame) -> None:
    series = (
        traffic[traffic["pop"] == "LON-CORE-01"]
        .set_index("timestamp")["traffic_gbps"]
        .sort_index()
    )
    horizon = 48
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    naive = SeasonalNaiveForecaster().fit(train).predict(horizon)
    lgbm = LightGBMForecaster(num_boost_round=150).fit(train).predict(horizon)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train.index[-168:], train.values[-168:], label="Training (last 7d)", color="#7f8c8d", linewidth=1.2)
    ax.plot(test.index, test.values, label="Actual", color="#2c3e50", linewidth=2.2)
    ax.plot(naive.timestamps, naive.point_forecast, label="Seasonal naive", color="#27ae60", linestyle="--", linewidth=1.8)
    ax.plot(lgbm.timestamps, lgbm.point_forecast, label="LightGBM", color="#e74c3c", linestyle="--", linewidth=1.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic (Gbps)")
    ax.set_title("48-hour forecast vs actual — LON-CORE-01")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "02_forecast_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_detection(traffic: pd.DataFrame) -> None:
    series = (
        traffic[traffic["pop"] == "LON-CORE-01"]
        .set_index("timestamp")["traffic_gbps"]
        .sort_index()
    )
    train = series.iloc[: -24 * 14]
    test = series.iloc[-24 * 14 :]
    expected = pd.Series(
        SeasonalNaiveForecaster(season_length=168).fit(train).predict(len(test)).point_forecast,
        index=test.index,
    )
    train_residuals = (train.iloc[168:] - train.shift(168).iloc[168:]).values
    detector = ResidualAnomalyDetector(threshold=3.5).fit(train_residuals)
    detection = detector.detect(test.index, test.values, expected.values)
    df = detection.to_dataframe()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["timestamp"], df["expected"], label="Expected (seasonal naive)", color="#7f8c8d", linestyle=":", linewidth=1.4)
    ax.plot(df["timestamp"], df["observed"], label="Observed", color="#2c3e50", linewidth=1.4)
    flagged = df[df["is_anomaly"]]
    outages = flagged[flagged["anomaly_type"] == "outage"]
    surges = flagged[flagged["anomaly_type"] == "surge"]
    if not outages.empty:
        ax.scatter(outages["timestamp"], outages["observed"], color="#e74c3c", s=60, marker="v", label=f"Outages ({len(outages)})", zorder=5)
    if not surges.empty:
        ax.scatter(surges["timestamp"], surges["observed"], color="#f39c12", s=60, marker="^", label=f"Surges ({len(surges)})", zorder=5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic (Gbps)")
    ax.set_title("Anomaly detection — LON-CORE-01, last 14 days")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "03_anomaly_detection.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    traffic = pd.read_parquet("data/network_traffic.parquet")
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], utc=True)
    plot_traffic_sample(traffic)
    plot_forecast_vs_actual(traffic)
    plot_anomaly_detection(traffic)
    print(f"Wrote 3 PNGs to {DOCS}/")


if __name__ == "__main__":
    main()
