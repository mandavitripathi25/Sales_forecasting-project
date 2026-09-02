"""
src/config.py
--------------
Configuration Hub

Centralizes all project settings: file paths, model parameters, and
static variables. Every other module should import from here rather
than hard-coding paths or hyperparameters, so the whole pipeline can
be reconfigured from a single place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_FILE = DATA_DIR / "sales_data.xlsx"     # source Excel file
SHEET_NAME = 0                                    # sheet index or name

FORECAST_OUTPUT_FILE = OUTPUT_DIR / "final_forecast.csv"
MODEL_METRICS_FILE = OUTPUT_DIR / "model_metrics.csv"

# Make sure output dir exists at import time so downstream modules can
# write to it without each having to check.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column naming conventions (post data_preparation cleaning)
# ---------------------------------------------------------------------------
# NOTE: these match the uploaded sales_data.xlsx schema after
# clean_column_names runs (e.g. "order_value_EUR" -> "order_value_eur").
DATE_COL = "date"
TARGET_COL = "order_value_eur"

# Columns that define the granular hierarchy, ordered from most granular
# to least granular. Used by SparsityAnalyzer / DecompositionEngine to
# decide the "best" level to forecast at and how to disaggregate back down.
# The dataset has country, product category, and device type as its
# categorical dimensions — sku/store-level detail isn't present here.
HIERARCHY_COLS = ["country", "category", "device_type"]

# ---------------------------------------------------------------------------
# Data preparation / sparsity settings
# ---------------------------------------------------------------------------
# Minimum fraction of non-zero periods required for a series to be
# considered "forecastable" at a given hierarchy level.
MIN_NON_ZERO_RATIO = 0.30

# Minimum number of historical periods required to fit a model.
# The dataset only spans 2 years of monthly data (24 periods total), so
# this is set well below the general-purpose 24-period default.
MIN_HISTORY_PERIODS = 12

# ---------------------------------------------------------------------------
# Train / test split & forecast horizon
# ---------------------------------------------------------------------------
# Only ~24 months of history exist in the dataset, so horizon/test/history
# are scaled down from typical multi-year-history defaults.
FORECAST_HORIZON = 6           # periods to forecast into the future
TEST_HORIZON = 3               # periods held out for backtesting
FREQUENCY = "MS"               # pandas offset alias — data is aggregated to monthly

# ---------------------------------------------------------------------------
# Individual model parameters
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "arima": {
        # (p, d, q) — set to None to let auto_arima-style search decide
        "order": (1, 1, 1),
        # Seasonal component turned off by default: with only ~24 months
        # of history, a 12-period seasonal SARIMA rarely has enough data
        # to fit reliably after the train/test split. Turn it back on
        # (e.g. (1, 1, 1, 12)) once you have 3+ years of history.
        "seasonal_order": (0, 0, 0, 0),
    },
    "ets": {
        "trend": "add",
        "seasonal": "add",
        "seasonal_periods": 12,
        "damped_trend": False,
    },
    "prophet": {
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "seasonality_mode": "additive",
    },
}

# ---------------------------------------------------------------------------
# Ensemble settings
# ---------------------------------------------------------------------------
# Which individual models feed the ensemble.
ENSEMBLE_MODELS = ["arima", "ets", "prophet"]

# "equal"   -> simple average of all model forecasts
# "inverse_error" -> weight each model by 1 / backtest error
ENSEMBLE_STRATEGY = "inverse_error"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
LOG_LEVEL = "INFO"
