"""Network capacity forecasting and anomaly detection."""
from .anomaly import AnomalyDetectionResult, ResidualAnomalyDetector, detector_recall
from .data_generator import GeneratorConfig, PoPConfig, generate_dataset
from .evaluation import mae, mape, rmse, smape, walk_forward_eval
from .forecasters import (
    Forecaster,
    ForecastResult,
    LightGBMForecaster,
    NotFittedError,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
    build_forecaster,
)

__version__ = "0.1.0"

__all__ = [
    "AnomalyDetectionResult",
    "Forecaster",
    "ForecastResult",
    "GeneratorConfig",
    "LightGBMForecaster",
    "NotFittedError",
    "PoPConfig",
    "ResidualAnomalyDetector",
    "SARIMAForecaster",
    "SeasonalNaiveForecaster",
    "build_forecaster",
    "detector_recall",
    "generate_dataset",
    "mae",
    "mape",
    "rmse",
    "smape",
    "walk_forward_eval",
]
