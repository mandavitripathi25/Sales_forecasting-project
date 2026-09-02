"""
src/ensemble_model.py
------------------------
Ensemble Engine

Contains EnsembleForecaster, which fits several individual models
(from forecasting_models.py) on the same series, backtests them on a
held-out window to estimate each model's error, and combines their
forecasts into a single blended forecast — either by simple average
or by weighting each model inversely to its backtest error.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src import config
from src.forecasting_models import build_model, NaiveSeasonalModel
from src.utils import get_logger, rmse

logger = get_logger(__name__, config.LOG_LEVEL)


class EnsembleForecaster:
    """
    Parameters
    ----------
    model_names : list[str]
        Which models (by name, matching forecasting_models.MODEL_REGISTRY)
        to include in the ensemble. Defaults to config.ENSEMBLE_MODELS.
    strategy : str
        "equal" -> simple average across models
        "inverse_error" -> weight by 1 / backtest RMSE (better models count more)
    test_horizon : int
        Holdout window (in periods) used to score each model before
        computing ensemble weights.
    """

    def __init__(
        self,
        model_names: List[str] = None,
        strategy: str = None,
        test_horizon: int = None,
    ):
        self.model_names = model_names or config.ENSEMBLE_MODELS
        self.strategy = strategy or config.ENSEMBLE_STRATEGY
        self.test_horizon = test_horizon or config.TEST_HORIZON
        self.weights: Dict[str, float] = {}
        self.backtest_errors: Dict[str, float] = {}
        self._fitted_models: Dict[str, object] = {}

    def _backtest(self, series: pd.Series) -> Dict[str, float]:
        """Fit each model on train-minus-holdout, score on the holdout."""
        if len(series) <= self.test_horizon + config.MIN_HISTORY_PERIODS:
            logger.warning(
                "Series too short for a clean backtest; falling back to equal weights."
            )
            return {name: 1.0 for name in self.model_names}

        train = series.iloc[: -self.test_horizon]
        holdout = series.iloc[-self.test_horizon :]

        errors = {}
        for name in self.model_names:
            try:
                model = build_model(name)
                model.fit(train)
                preds = model.predict(self.test_horizon)["yhat"].values
                errors[name] = rmse(holdout.values, preds)
            except Exception as e:
                logger.warning(f"Model '{name}' failed during backtest ({e}); excluding.")
                errors[name] = np.inf
        return errors

    def _compute_weights(self, errors: Dict[str, float]) -> Dict[str, float]:
        usable = {k: v for k, v in errors.items() if np.isfinite(v)}
        if not usable:
            raise RuntimeError(
                "Every model failed during backtesting — check data quality/history length."
            )

        if self.strategy == "equal":
            w = {k: 1.0 for k in usable}
        else:  # inverse_error
            # add a small epsilon so a perfect (0-error) fit doesn't divide by zero
            w = {k: 1.0 / (v + 1e-6) for k, v in usable.items()}

        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def fit(self, series: pd.Series) -> "EnsembleForecaster":
        """Backtest models, derive weights, then refit each on the full series."""
        self.backtest_errors = self._backtest(series)
        self.weights = self._compute_weights(self.backtest_errors)

        logger.info(f"Ensemble weights: {self.weights}")

        for name in self.weights:
            model = build_model(name)
            try:
                model.fit(series)
                self._fitted_models[name] = model
            except Exception as e:
                logger.warning(f"Model '{name}' failed on full-history fit ({e}); using naive fallback.")
                fallback = NaiveSeasonalModel(seasonal_periods=12)
                fallback.fit(series)
                self._fitted_models[name] = fallback
        return self

    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Return a DataFrame indexed by future date with:
          - one column per component model's forecast
          - an 'ensemble' column: the weighted blend
        """
        horizon = horizon or config.FORECAST_HORIZON
        component_forecasts = {}
        for name, model in self._fitted_models.items():
            component_forecasts[name] = model.predict(horizon)["yhat"]

        result = pd.DataFrame(component_forecasts)
        result["ensemble"] = sum(
            result[name] * weight for name, weight in self.weights.items()
        )
        return result
