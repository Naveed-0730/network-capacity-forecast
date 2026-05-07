"""
End-to-end anomaly detection across all PoPs.

Trains a seasonal-naive baseline, computes residuals on the last 14 days,
calibrates a MAD-based detector, and reports recall against the injected
ground-truth anomalies.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from network_forecast.anomaly import ResidualAnomalyDetector, detector_recall
from network_forecast.forecasters import SeasonalNaiveForecaster

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    data_dir = Path("data")
    traffic = pd.read_parquet(data_dir / "network_traffic.parquet")
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], utc=True)
    ground_truth = pd.read_csv(data_dir / "anomalies_ground_truth.csv")
    ground_truth["start_ts"] = pd.to_datetime(ground_truth["start_ts"], utc=True)

    all_recalls: list[dict] = []
    for pop in sorted(traffic["pop"].unique()):
        series = (
            traffic[traffic["pop"] == pop]
            .set_index("timestamp")["traffic_gbps"]
            .sort_index()
        )

        # Hold out the last 30 days for detection
        train = series.iloc[: -24 * 30]
        test = series.iloc[-24 * 30 :]

        # Generate "expected" via tiled seasonal naive
        history = train.copy()
        expected_values = []
        n_chunks = (len(test) // 168) + 1
        for _ in range(n_chunks):
            f = SeasonalNaiveForecaster(season_length=168).fit(history).predict(168)
            expected_values.extend(f.point_forecast)
            next_idx = pd.date_range(
                history.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h", tz="UTC",
            )
            history = pd.concat([history, pd.Series(f.point_forecast, index=next_idx)])
        expected = pd.Series(expected_values[: len(test)], index=test.index)

        residuals = (test - expected).values
        # Calibrate detector on train residuals (cleaner — fewer injected anomalies)
        train_residuals = (train.iloc[168:] - train.shift(168).iloc[168:]).values
        detector = ResidualAnomalyDetector(threshold=3.5).fit(train_residuals)
        detection = detector.detect(test.index, test.values, expected.values)

        pop_truth = ground_truth[
            (ground_truth["pop"] == pop)
            & (ground_truth["start_ts"] >= test.index[0])
        ]
        recall = detector_recall(detection, pop_truth)
        recall["pop"] = pop
        recall["n_flagged"] = int(detection.is_anomaly.sum())
        recall["n_truth"] = len(pop_truth)
        all_recalls.append(recall)
        logger.info("[%s] flagged=%d, truth=%d, recall=%.2f",
                    pop, recall["n_flagged"], recall["n_truth"], recall["overall_recall"])

    summary = pd.DataFrame(all_recalls)
    print("\n=== Anomaly detection summary ===")
    print(summary[["pop", "n_truth", "n_flagged", "outage_recall", "surge_recall", "overall_recall"]].to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/anomaly_detection_summary.csv", index=False)


if __name__ == "__main__":
    main()
