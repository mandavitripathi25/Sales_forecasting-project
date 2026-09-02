"""
src/forecasting_models.py
---------------------------
Model Library

A collection of individual time series forecasting models, each
wrapped behind the same standardized interface:

    model = SomeModel(**params)
    model.fit(train_series)
    forecast_df = model.predict(horizon)   # columns: ["yhat"]

This lets ensemble_model.py and decomposition_engine.py treat every
model identically without caring what's underneath.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__, config.LOG_LEVEL)


class BaseForecastModel(ABC):
    """Common interface every model in the library must implement."""

    name = "base"

    def __init__(self, **params):
        self.params = params
        self._model = None
        self._fitted = False
        self._history: pd.Series = None

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseForecastModel":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Return a DataFrame indexed by future dates with a 'yhat' column."""
        ...

    def _future_index(self, horizon: int) -> pd.DatetimeIndex:
        last_date = self._history.index[-1]
        return pd.date_range(
            start=last_date, periods=horizon + 1, freq=config.FREQUENCY
        )[1:]


class ArimaModel(BaseForecastModel):
    """SARIMA wrapper via statsmodels."""

    name = "arima"

    def fit(self, series: pd.Series) -> "ArimaModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._history = series.astype(float)
        order = self.params.get("order", (1, 1, 1))
        seasonal_order = self.params.get("seasonal_order", (0, 0, 0, 0))

        self._model = SARIMAX(
            self._history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        forecast = self._model.forecast(steps=horizon)
        idx = self._future_index(horizon)
        return pd.DataFrame({"yhat": forecast.values}, index=idx)


class EtsModel(BaseForecastModel):
    """Exponential smoothing (Holt-Winters) wrapper via statsmodels."""

    name = "ets"

    def fit(self, series: pd.Series) -> "EtsModel":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._history = series.astype(float)
        # ETS needs at least 2 full seasonal cycles for a seasonal fit;
        # fall back to non-seasonal if history is too short.
        seasonal_periods = self.params.get("seasonal_periods", 12)
        use_seasonal = len(self._history) >= 2 * seasonal_periods

        self._model = ExponentialSmoothing(
            self._history,
            trend=self.params.get("trend", "add"),
            seasonal=self.params.get("seasonal", "add") if use_seasonal else None,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            damped_trend=self.params.get("damped_trend", False),
        ).fit()
        self._fitted = True
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        forecast = self._model.forecast(horizon)
        idx = self._future_index(horizon)
        return pd.DataFrame({"yhat": forecast.values}, index=idx)


class ProphetModel(BaseForecastModel):
    """
    Facebook/Meta Prophet wrapper. Prophet is an optional dependency —
    if it isn't installed, fit() raises a clear ImportError rather than
    silently degrading, so callers can decide whether to drop it from
    the ensemble.
    """

    name = "prophet"

    def fit(self, series: pd.Series) -> "ProphetModel":
        try:
            from prophet import Prophet
        except ImportError as e:
            raise ImportError(
                "prophet is not installed. Run `pip install prophet` or remove "
                "'prophet' from config.ENSEMBLE_MODELS."
            ) from e

        self._history = series.astype(float)
        df = self._history.reset_index()
        df.columns = ["ds", "y"]

        self._model = Prophet(
            yearly_seasonality=self.params.get("yearly_seasonality", True),
            weekly_seasonality=self.params.get("weekly_seasonality", False),
            daily_seasonality=self.params.get("daily_seasonality", False),
            seasonality_mode=self.params.get("seasonality_mode", "additive"),
        )
        self._model.fit(df)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        future = self._model.make_future_dataframe(
            periods=horizon, freq=config.FREQUENCY
        )
        forecast = self._model.predict(future).set_index("ds")
        idx = self._future_index(horizon)
        return pd.DataFrame({"yhat": forecast.loc[idx, "yhat"].values}, index=idx)


class NaiveSeasonalModel(BaseForecastModel):
    """
    Simple, dependency-free fallback: repeats the last full seasonal
    cycle. Useful for series too short/sparse for the statistical
    models, and as a sanity baseline in backtests.
    """

    name = "naive_seasonal"

    def fit(self, series: pd.Series) -> "NaiveSeasonalModel":
        self._history = series.astype(float)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        seasonal_periods = self.params.get("seasonal_periods", 12)
        last_cycle = self._history.iloc[-seasonal_periods:].values
        reps = int(np.ceil(horizon / seasonal_periods))
        values = np.tile(last_cycle, reps)[:horizon]
        idx = self._future_index(horizon)
        return pd.DataFrame({"yhat": values}, index=idx)


MODEL_REGISTRY = {
    "arima": ArimaModel,
    "ets": EtsModel,
    "prophet": ProphetModel,
    "naive_seasonal": NaiveSeasonalModel,
}


def build_model(model_name: str) -> BaseForecastModel:
    """Factory: instantiate a model by name using its config.MODEL_PARAMS."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        )
    params = config.MODEL_PARAMS.get(model_name, {})
    return MODEL_REGISTRY[model_name](**params)
