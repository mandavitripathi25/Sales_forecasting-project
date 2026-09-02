"""
src/data_preparation.py
-------------------------
Data Transformer

- DataPreparer: cleans the raw ingested data and builds the "master
  table" (a fully-populated date x hierarchy grid with the target
  variable, ready for modeling).
- SparsityAnalyzer: walks the hierarchy defined in config.HIERARCHY_COLS
  and finds the most granular level at which series are still dense
  enough to forecast reliably.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config
from src.utils import ensure_datetime_index, get_logger

logger = get_logger(__name__, config.LOG_LEVEL)


class DataPreparer:
    """
    Cleans raw data and builds the master table used by every model.

    The master table is a DataFrame with:
      - one row per (date, *hierarchy level combination)
      - a complete date range per group (no missing periods — gaps are
        filled with 0, since a missing sales row usually means "no sale",
        not "no data")
    """

    def __init__(self, hierarchy_cols: List[str] = None):
        self.hierarchy_cols = hierarchy_cols or config.HIERARCHY_COLS

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic hygiene: dedupe, drop rows with no date/target, coerce types."""
        df = df.copy()
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")
        df[config.TARGET_COL] = pd.to_numeric(df[config.TARGET_COL], errors="coerce")

        before = len(df)
        df = df.dropna(subset=[config.DATE_COL, config.TARGET_COL])
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with invalid date/target values.")

        df = df.drop_duplicates()
        return df

    def build_master_table(
        self, df: pd.DataFrame, group_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate to (date, group_cols) grain and densify each group's
        date range so every group has one row per period, zero-filled.
        """
        group_cols = group_cols if group_cols is not None else self.hierarchy_cols
        group_cols = [c for c in group_cols if c in df.columns]

        # pandas groupby drops rows with NaN keys by default — fill hierarchy
        # NaNs with an explicit "Unknown" label instead of silently losing data.
        df = df.copy()
        if group_cols:
            df[group_cols] = df[group_cols].fillna("Unknown")

        # Source dates are usually daily-resolution (e.g. individual order
        # timestamps), not pre-aligned to the forecasting period. Bucket
        # each date into its period (pd.Grouper handles this correctly,
        # e.g. binning every day in a month to that month's start under
        # FREQUENCY="MS") *before* summing — matching on exact dates
        # against a period grid would silently drop almost everything.
        agg = (
            df.groupby(
                [pd.Grouper(key=config.DATE_COL, freq=config.FREQUENCY)] + group_cols,
                as_index=False,
            )[config.TARGET_COL]
            .sum()
        )

        full_range = pd.date_range(
            agg[config.DATE_COL].min(), agg[config.DATE_COL].max(), freq=config.FREQUENCY
        )

        if group_cols:
            groups = agg[group_cols].drop_duplicates()
            groups["_key"] = 1
            dates = pd.DataFrame({config.DATE_COL: full_range, "_key": 1})
            skeleton = groups.merge(dates, on="_key").drop(columns="_key")
            master = skeleton.merge(agg, on=[config.DATE_COL] + group_cols, how="left")
        else:
            master = pd.DataFrame({config.DATE_COL: full_range}).merge(
                agg, on=config.DATE_COL, how="left"
            )

        master[config.TARGET_COL] = master[config.TARGET_COL].fillna(0)
        master = master.sort_values([config.DATE_COL] + group_cols).reset_index(drop=True)

        logger.info(
            f"Master table built: {len(master):,} rows across "
            f"{master[group_cols].drop_duplicates().shape[0] if group_cols else 1} series."
        )
        return master


class SparsityAnalyzer:
    """
    Determines the most granular level in the hierarchy that still has
    enough non-zero history to forecast reliably. Forecasting SKU x
    store daily sales directly is often too sparse; this walks the
    hierarchy from most to least granular and stops at the first level
    that clears the density/history thresholds.
    """

    def __init__(
        self,
        hierarchy_cols: List[str] = None,
        min_non_zero_ratio: float = None,
        min_history_periods: int = None,
    ):
        self.hierarchy_cols = hierarchy_cols or config.HIERARCHY_COLS
        self.min_non_zero_ratio = min_non_zero_ratio or config.MIN_NON_ZERO_RATIO
        self.min_history_periods = min_history_periods or config.MIN_HISTORY_PERIODS

    def _series_quality(self, group_df: pd.DataFrame) -> Tuple[float, int]:
        n_periods = len(group_df)
        non_zero_ratio = (
            (group_df[config.TARGET_COL] != 0).sum() / n_periods if n_periods else 0
        )
        return non_zero_ratio, n_periods

    def evaluate_level(self, master_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Return per-series quality stats at a given hierarchy level.

        master_df is built at the *finest* hierarchy level, so when
        group_cols is a coarser subset (or empty, for TOTAL), we first
        collapse it down by summing the target across the dropped
        dimensions — otherwise every row of the finer table would get
        miscounted as its own "period" instead of being combined per date.
        """
        group_cols = [c for c in group_cols if c in master_df.columns]
        collapsed = (
            master_df.groupby([config.DATE_COL] + group_cols, as_index=False)[
                config.TARGET_COL
            ].sum()
        )

        if not group_cols:
            ratio, periods = self._series_quality(collapsed)
            return pd.DataFrame(
                [{"non_zero_ratio": ratio, "n_periods": periods, "passes": (
                    ratio >= self.min_non_zero_ratio and periods >= self.min_history_periods
                )}]
            )

        rows = []
        for keys, g in collapsed.groupby(group_cols):
            ratio, periods = self._series_quality(g)
            row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row["non_zero_ratio"] = ratio
            row["n_periods"] = periods
            row["passes"] = (
                ratio >= self.min_non_zero_ratio and periods >= self.min_history_periods
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def find_best_level(self, master_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Walk hierarchy_cols from most granular (full list) to least
        granular (empty list == totally aggregated), returning the
        first level where the majority of series pass quality checks.
        """
        for i in range(len(self.hierarchy_cols), -1, -1):
            candidate_cols = self.hierarchy_cols[:i]
            stats = self.evaluate_level(master_df, candidate_cols)
            pass_rate = stats["passes"].mean() if len(stats) else 0

            logger.info(
                f"Level {candidate_cols or ['TOTAL']}: "
                f"{len(stats)} series, {pass_rate:.0%} pass quality thresholds."
            )
            if pass_rate >= 0.8:  # majority of series are forecastable at this level
                logger.info(f"Selected forecast level: {candidate_cols or ['TOTAL']}")
                return candidate_cols, stats

        logger.warning("No level cleanly passed thresholds; falling back to TOTAL level.")
        return [], self.evaluate_level(master_df, [])
