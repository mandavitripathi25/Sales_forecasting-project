"""
src/utils.py
------------
Utility Functions

Small, reusable helper functions shared across the pipeline. Keep this
module dependency-light — it should never import from other src/
modules, so any module can safely import from it without circular
imports.
"""

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger so every module logs consistently."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, strip whitespace, replace
    spaces/special characters with underscores.

    'Store  ID ' -> 'store_id'
    'Product-Category' -> 'product_category'
    """
    df = df.copy()
    new_cols = []
    for col in df.columns:
        col = str(col).strip().lower()
        col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
        col = re.sub(r"_+", "_", col).strip("_")
        new_cols.append(col)
    df.columns = new_cols
    return df


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Percentage Error, ignoring zero-actual periods."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Symmetric MAPE — more stable than MAPE when actuals are near zero."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted))
    mask = denom != 0
    if not mask.any():
        return np.nan
    return float(np.mean(2 * np.abs(predicted[mask] - actual[mask]) / denom[mask]) * 100)


def ensure_datetime_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Set `date_col` as a sorted DatetimeIndex."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df
