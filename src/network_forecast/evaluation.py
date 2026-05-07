"""
Forecast evaluation: metrics and walk-forward (rolling-origin) validation.

Walk-forward is the only honest way to score a time-series model — random
k-fold CV leaks future information into the past. This module implements
expanding-window walk-forward and the four metrics network ops actually
care about: MAE (Gbps you'll be off by), RMSE (penalises big misses,
matters for capacity planning), MAPE (interpretable percentage), and
sMAPE (handles zero/near-zero traffic during outages without blowing up).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecasters import Forecaster

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Per-fold metrics for a single forecaster."""

    forecaster_name: str
    fold: int
    mae: float
    rmse: float
    mape: float
    smape: float
    horizon: int


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in the units of the target (Gbps here)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error — penalises large errors more heavily."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean absolute percentage error.

    Guards against division-by-zero with a small ``eps``; if the true series
    contains genuine zeros (e.g. during an outage), prefer ``smape`` instead.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE — bounded in [0, 200], well-defined when y_true ≈ 0.

    Better choice than MAPE for network telemetry where outages produce
    near-zero observed values that would explode a vanilla MAPE.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return float(np.mean(diff) * 100)


def walk_forward_eval(
    series: pd.Series,
    forecaster_factory,
    horizon: int = 24,
    n_folds: int = 4,
    initial_train_size: int | None = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.

    Each fold trains on everything up to ``cutoff_i`` and forecasts the next
    ``horizon`` periods. Returns a tidy DataFrame of per-fold metrics that
    you can aggregate by ``.groupby('forecaster_name')`` for a final table.

    Parameters
    ----------
    series : the full historical series (must be sorted, DatetimeIndex).
    forecaster_factory : zero-arg callable returning a fresh ``Forecaster``.
        Passed as a factory rather than an instance because each fold needs
        a *new* unfitted model.
    horizon : forecast horizon in periods (24 hours = 1 day ahead by default).
    n_folds : number of rolling-origin folds.
    initial_train_size : observations in the first training window. Defaults
        to ``len(series) - n_folds * horizon``.
    """
    if not series.index.is_monotonic_increasing:
        raise ValueError("series index must be sorted ascending.")

    if initial_train_size is None:
        initial_train_size = len(series) - n_folds * horizon
    if initial_train_size <= 0:
        raise ValueError(
            f"initial_train_size resolved to {initial_train_size}; "
            f"reduce n_folds or horizon."
        )

    results: list[EvalResult] = []
    forecaster = forecaster_factory()
    name = forecaster.name

    for fold in range(n_folds):
        cutoff = initial_train_size + fold * horizon
        train = series.iloc[:cutoff]
        test = series.iloc[cutoff : cutoff + horizon]

        if len(test) < horizon:
            logger.warning("Fold %d truncated (only %d test points); skipping", fold, len(test))
            continue

        model = forecaster_factory()
        model.fit(train)
        forecast = model.predict(horizon)

        y_true = test.values
        y_pred = forecast.point_forecast

        results.append(EvalResult(
            forecaster_name=name,
            fold=fold,
            mae=mae(y_true, y_pred),
            rmse=rmse(y_true, y_pred),
            mape=mape(y_true, y_pred),
            smape=smape(y_true, y_pred),
            horizon=horizon,
        ))
        logger.info(
            "[%s] fold %d: MAE=%.2f RMSE=%.2f sMAPE=%.2f%%",
            name, fold, results[-1].mae, results[-1].rmse, results[-1].smape,
        )

    return pd.DataFrame([r.__dict__ for r in results])
