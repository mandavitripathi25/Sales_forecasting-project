"""
src/decomposition_engine.py
------------------------------
Forecast Allocator

Contains DecompositionEngine, which disaggregates a single high-level
ensemble forecast down to a granular level (e.g. total -> region ->
store -> SKU) using historical proportions. This is the standard
"top-down" reconciliation approach: forecast where the signal is
dense and stable, then split the total back down using each group's
historical share, optionally trending that share over time.
"""

from typing import List

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__, config.LOG_LEVEL)


class DecompositionEngine:
    """
    Parameters
    ----------
    hierarchy_cols : list[str]
        Granular columns to disaggregate down to.
    method : str
        "average_share" -> each group's share is its average share of
            history-wide totals (stable, ignores trend).
        "recent_share"  -> each group's share is computed from only the
            most recent `recent_periods` periods (reacts to trend/mix shift).
    recent_periods : int
        Window size used when method="recent_share".
    """

    def __init__(
        self,
        hierarchy_cols: List[str] = None,
        method: str = "recent_share",
        recent_periods: int = 6,
    ):
        self.hierarchy_cols = hierarchy_cols or config.HIERARCHY_COLS
        self.method = method
        self.recent_periods = recent_periods
        self.shares_: pd.DataFrame = None

    def compute_shares(
        self, master_df: pd.DataFrame, group_cols: List[str]
    ) -> pd.DataFrame:
        """
        Compute each group's historical share of the total, at the
        granular `group_cols` level.
        """
        df = master_df.copy()

        if self.method == "recent_share":
            cutoff_dates = sorted(df[config.DATE_COL].unique())[-self.recent_periods :]
            df = df[df[config.DATE_COL].isin(cutoff_dates)]

        group_totals = df.groupby(group_cols)[config.TARGET_COL].sum().reset_index()
        grand_total = group_totals[config.TARGET_COL].sum()

        if grand_total == 0:
            logger.warning("Grand total across history is 0; falling back to equal shares.")
            n = len(group_totals)
            group_totals["share"] = 1.0 / n if n else 0
        else:
            group_totals["share"] = group_totals[config.TARGET_COL] / grand_total

        self.shares_ = group_totals[group_cols + ["share"]]
        return self.shares_

    def disaggregate(
        self, top_level_forecast: pd.DataFrame, group_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Split a top-level forecast (DataFrame indexed by date with an
        'ensemble' column) down to the granular level using computed
        shares. Returns one row per (date, *group_cols) with an
        allocated 'forecast' column that sums back to the top-level
        total for every date.
        """
        group_cols = group_cols or self.hierarchy_cols
        if self.shares_ is None:
            raise RuntimeError("Call .compute_shares() before .disaggregate().")

        top = top_level_forecast.reset_index().rename(columns={"index": config.DATE_COL})
        top["_key"] = 1
        shares = self.shares_.copy()
        shares["_key"] = 1

        allocated = top.merge(shares, on="_key").drop(columns="_key")
        allocated["forecast"] = allocated["ensemble"] * allocated["share"]

        cols = [config.DATE_COL] + group_cols + ["forecast", "share"]
        allocated = allocated[cols].sort_values([config.DATE_COL] + group_cols)

        # sanity check: allocations should sum back to the top-level total
        check = allocated.groupby(config.DATE_COL)["forecast"].sum()
        top_indexed = top.set_index(config.DATE_COL)["ensemble"]
        max_diff = (check - top_indexed).abs().max()
        if max_diff > 1e-6:
            logger.warning(f"Disaggregated totals drift from top-level forecast by {max_diff:.4f}.")

        return allocated.reset_index(drop=True)
