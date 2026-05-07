"""
Forecasting models for network capacity planning.

All models implement a common ``Forecaster`` interface, enabling
champion-challenger comparison and clean swapping in production pipelines.
Currently includes:

- SeasonalNaiveForecaster: strong baseline that copies the same hour from
  one week ago. Hard to beat on stable telemetry, surprisingly often wins.
- SARIMAForecaster: classical statistical model, captures local dynamics.
- LightGBMForecaster: gradient-boosted model on engineered calendar +
  lag features. Scales to many series and handles holidays cleanly.

The abstraction means the dashboard, evaluator, and CI tests don't care
which model is plugged in — important for long-term maintainability.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


class NotFittedError(RuntimeError):
    """Raised when ``predict`` is called before ``fit``."""


@dataclass
class ForecastResult:
    """Container for forecast output with optional uncertainty intervals."""

    timestamps: pd.DatetimeIndex
    point_forecast: np.ndarray
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a tidy DataFrame for plotting and stakeholder reports."""
        data = {
            "timestamp": self.timestamps,
            "forecast": self.point_forecast,
        }
        if self.lower_bound is not None:
            data["lower_80"] = self.lower_bound
        if self.upper_bound is not None:
            data["upper_80"] = self.upper_bound
        return pd.DataFrame(data)


class Forecaster(ABC):
    """Abstract base class for all forecasters in this project."""

    name: str = "abstract"

    def __init__(self) -> None:
        self._is_fitted = False

    @abstractmethod
    def fit(self, series: pd.Series) -> "Forecaster":
        """Fit the model. ``series`` must have a tz-aware DatetimeIndex."""

    @abstractmethod
    def predict(self, horizon: int) -> ForecastResult:
        """Produce a forecast for the next ``horizon`` periods."""

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(f"{self.name} must be fitted before predicting.")

    @staticmethod
    def _validate_series(series: pd.Series) -> None:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("series must have a DatetimeIndex.")
        if series.isna().any():
            raise ValueError("series contains NaN values; impute or drop before fitting.")
        if len(series) < 168:
            raise ValueError(f"series has {len(series)} obs; need ≥168 (one week of hourly data).")


class SeasonalNaiveForecaster(Forecaster):
    """
    Predicts y_t = y_{t - season_length}.

    Despite (or because of) its simplicity, this is the model to beat.
    If your fancy LightGBM can't outperform a 168-hour seasonal naive,
    you don't have a model — you have a confidently wrong baseline.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 168) -> None:
        super().__init__()
        self.season_length = season_length
        self._last_season: np.ndarray | None = None
        self._last_timestamp: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, series: pd.Series) -> "SeasonalNaiveForecaster":
        self._validate_series(series)
        if len(series) < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations, got {len(series)}."
            )
        self._last_season = series.values[-self.season_length :].copy()
        self._last_timestamp = series.index[-1]
        self._freq = pd.infer_freq(series.index) or "h"
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> ForecastResult:
        self._check_fitted()
        # Tile the last season as many times as needed, then slice to horizon.
        repeats = int(np.ceil(horizon / self.season_length))
        tiled = np.tile(self._last_season, repeats)[:horizon]
        future_ts = pd.date_range(
            start=self._last_timestamp + pd.Timedelta(1, unit=self._freq),
            periods=horizon,
            freq=self._freq,
        )
        return ForecastResult(timestamps=future_ts, point_forecast=tiled)


class SARIMAForecaster(Forecaster):
    """
    SARIMA(p,d,q)(P,D,Q,s) wrapper around statsmodels.

    Fitted with a small (1,1,1)(1,1,1,24) order by default — daily seasonality
    only, since fitting weekly seasonality (s=168) on years of hourly data is
    prohibitively slow. For weekly patterns we rely on LightGBM instead.
    """

    name = "sarima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted_model = None
        self._last_timestamp: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, series: pd.Series) -> "SARIMAForecaster":
        self._validate_series(series)
        logger.info("Fitting SARIMA%s%s on %d obs", self.order, self.seasonal_order, len(series))
        # SARIMAX is slow on long hourly series; downstream callers should
        # pass the tail of their training data (e.g. last 60 days).
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fitted_model = model.fit(disp=False, maxiter=50)
        self._last_timestamp = series.index[-1]
        self._freq = pd.infer_freq(series.index) or "h"
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> ForecastResult:
        self._check_fitted()
        forecast = self._fitted_model.get_forecast(steps=horizon)
        mean = forecast.predicted_mean.values
        conf_int = forecast.conf_int(alpha=0.2).values  # 80% interval
        future_ts = pd.date_range(
            start=self._last_timestamp + pd.Timedelta(1, unit=self._freq),
            periods=horizon,
            freq=self._freq,
        )
        return ForecastResult(
            timestamps=future_ts,
            point_forecast=mean,
            lower_bound=conf_int[:, 0],
            upper_bound=conf_int[:, 1],
        )


class LightGBMForecaster(Forecaster):
    """
    Direct multi-step LightGBM forecaster on engineered calendar + lag features.

    Uses a *recursive* prediction strategy: predicts one step ahead, appends
    the prediction to history, recomputes lags, predicts again. Slower than
    direct multi-output but produces well-behaved long forecasts and avoids
    training H separate models.
    """

    name = "lightgbm"

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 24, 168),
        rolling_windows: tuple[int, ...] = (24, 168),
        num_boost_round: int = 200,
    ) -> None:
        super().__init__()
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.num_boost_round = num_boost_round
        self._model: lgb.Booster | None = None
        self._history: pd.Series | None = None
        self._freq: str | None = None

    def _build_features(self, series: pd.Series) -> pd.DataFrame:
        """Calendar features + lag features + rolling-window summaries."""
        df = pd.DataFrame({"y": series.values}, index=series.index)
        df["hour"] = series.index.hour
        df["dow"] = series.index.dayofweek
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["month"] = series.index.month
        df["day_of_year"] = series.index.dayofyear

        for lag in self.lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        for window in self.rolling_windows:
            df[f"rmean_{window}"] = df["y"].shift(1).rolling(window).mean()
            df[f"rstd_{window}"] = df["y"].shift(1).rolling(window).std()

        return df.dropna()

    def fit(self, series: pd.Series) -> "LightGBMForecaster":
        self._validate_series(series)
        feature_df = self._build_features(series)
        X = feature_df.drop(columns=["y"])
        y = feature_df["y"]
        logger.info("Training LightGBM on %d rows × %d features", len(X), X.shape[1])

        train_set = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self._model = lgb.train(params, train_set, num_boost_round=self.num_boost_round)
        self._history = series.copy()
        self._freq = pd.infer_freq(series.index) or "h"
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> ForecastResult:
        self._check_fitted()
        history = self._history.copy()
        predictions: list[float] = []

        for _ in range(horizon):
            features = self._build_features(history).iloc[[-1]].drop(columns=["y"])
            yhat = float(self._model.predict(features)[0])
            predictions.append(yhat)
            next_ts = history.index[-1] + pd.Timedelta(1, unit=self._freq)
            history = pd.concat([history, pd.Series([yhat], index=[next_ts])])

        future_ts = history.index[-horizon:]
        return ForecastResult(timestamps=future_ts, point_forecast=np.array(predictions))


def build_forecaster(name: str) -> Forecaster:
    """Factory for instantiating forecasters by name (used by the CLI / dashboard)."""
    registry: dict[str, type[Forecaster]] = {
        "seasonal_naive": SeasonalNaiveForecaster,
        "sarima": SARIMAForecaster,
        "lightgbm": LightGBMForecaster,
    }
    if name not in registry:
        raise ValueError(f"Unknown forecaster '{name}'. Available: {sorted(registry)}")
    return registry[name]()
