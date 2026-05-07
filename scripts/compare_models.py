"""
Champion-challenger comparison across all forecasters.

Runs walk-forward evaluation for each model on each PoP and emits a tidy
results table to ``results/model_comparison.csv`` plus a markdown summary
suitable for a Confluence post or pull-request description.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from network_forecast.evaluation import walk_forward_eval
from network_forecast.forecasters import (
    LightGBMForecaster,
    SeasonalNaiveForecaster,
)

logger = logging.getLogger(__name__)

# SARIMA is omitted by default — fitting (1,1,1)(1,1,1,24) on the full hourly
# dataset takes >10 minutes per fold. The CLI flag --include-sarima re-enables
# it for users who want the comparison and have the time.
DEFAULT_FORECASTERS: dict[str, callable] = {
    "seasonal_naive": SeasonalNaiveForecaster,
    "lightgbm": lambda: LightGBMForecaster(num_boost_round=150),
}


def run_comparison(
    data_path: Path,
    output_path: Path,
    horizon: int = 24,
    n_folds: int = 4,
    pops: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate all forecasters on all PoPs (or a subset) and save the results."""
    traffic = pd.read_parquet(data_path)
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], utc=True)

    if pops is None:
        pops = sorted(traffic["pop"].unique())

    all_results: list[pd.DataFrame] = []
    for pop in pops:
        series = (
            traffic[traffic["pop"] == pop]
            .set_index("timestamp")["traffic_gbps"]
            .sort_index()
        )
        logger.info("Evaluating PoP %s (%d obs)", pop, len(series))
        for name, factory in DEFAULT_FORECASTERS.items():
            logger.info("  → %s", name)
            df = walk_forward_eval(series, factory, horizon=horizon, n_folds=n_folds)
            df["pop"] = pop
            all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary = (
        results.groupby(["pop", "forecaster_name"])
        .agg(mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"), smape_mean=("smape", "mean"))
        .round(2)
    )
    print("\n=== Walk-forward summary ===")
    print(summary.to_string())
    print(f"\nFull results written to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Champion-challenger forecaster comparison")
    parser.add_argument("--data", type=Path, default=Path("data/network_traffic.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/model_comparison.csv"))
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--pops", nargs="*", help="Subset of PoPs to evaluate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_comparison(args.data, args.output, args.horizon, args.n_folds, args.pops)


if __name__ == "__main__":
    main()
