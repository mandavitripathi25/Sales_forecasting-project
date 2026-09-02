"""
src/data_ingestion.py
----------------------
Data Loader

Contains the DataIngestor class, responsible for loading raw data from
the Excel file in data/ and handing back a clean, validated DataFrame
ready for src/data_preparation.py.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src import config
from src.utils import clean_column_names, get_logger

logger = get_logger(__name__, config.LOG_LEVEL)


class DataIngestor:
    """
    Loads raw sales data from an Excel file and performs light,
    load-time validation (file exists, required columns present,
    no fully-empty sheet).

    Parameters
    ----------
    file_path : Path, optional
        Overrides config.RAW_DATA_FILE. Useful for tests or ad-hoc runs.
    sheet_name : str | int, optional
        Overrides config.SHEET_NAME.
    """

    def __init__(
        self,
        file_path: Optional[Path] = None,
        sheet_name=None,
        required_columns: Optional[list] = None,
    ):
        self.file_path = Path(file_path) if file_path else config.RAW_DATA_FILE
        self.sheet_name = sheet_name if sheet_name is not None else config.SHEET_NAME
        # Sensible defaults; adjust to match your real source schema.
        self.required_columns = required_columns or [
            config.DATE_COL,
            config.TARGET_COL,
        ]

    def load(self) -> pd.DataFrame:
        """
        Read the source file and return a cleaned-column DataFrame.

        Detects the *real* file format from its content (not just the
        file extension) and dispatches to the right pandas reader, since
        files named ".xlsx" are sometimes actually CSVs or old-style
        ".xls" files under the hood.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Expected source data at {self.file_path}, but it was not found. "
                f"Place your file there or point DataIngestor to the right path."
            )

        detected_format = self._detect_format()
        logger.info(
            f"Loading data from {self.file_path} "
            f"(detected format={detected_format}, sheet={self.sheet_name})"
        )
        raw_df = self._read(detected_format)

        if raw_df.empty:
            raise ValueError(f"{self.file_path} loaded but contains no rows.")

        df = clean_column_names(raw_df)
        self._validate(df)

        logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
        return df

    def _detect_format(self) -> str:
        """
        Sniff the file's actual bytes to determine its real format,
        regardless of what the file extension claims.

        Returns one of: "xlsx", "xls", "csv".
        """
        with open(self.file_path, "rb") as f:
            header = f.read(8)

        # .xlsx (and .xlsm/.docx/etc.) are zip archives -> start with PK\x03\x04
        if header.startswith(b"PK\x03\x04"):
            return "xlsx"
        # legacy .xls (OLE2 compound file) signature
        if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return "xls"
        # otherwise, assume it's actually plain text -> csv/tsv
        return "csv"

    def _read(self, detected_format: str) -> pd.DataFrame:
        if detected_format == "xlsx":
            return pd.read_excel(self.file_path, sheet_name=self.sheet_name, engine="openpyxl")
        if detected_format == "xls":
            return pd.read_excel(self.file_path, sheet_name=self.sheet_name, engine="xlrd")

        # csv fallback — also handles semicolon/tab-separated exports
        logger.warning(
            f"{self.file_path.name} isn't actually an Excel file (it's plain text) — "
            f"reading it as CSV instead. Consider re-saving it as a real .xlsx to avoid this."
        )
        try:
            return pd.read_csv(self.file_path, sep=None, engine="python")
        except Exception as e:
            raise ValueError(
                f"Could not read {self.file_path} as Excel or CSV. "
                f"Open it and confirm it's a real, non-corrupted data file. Original error: {e}"
            ) from e

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Source data is missing required columns {missing}. "
                f"Found columns: {list(df.columns)}. "
                f"Update src/config.py DATE_COL/TARGET_COL or the Excel headers to match."
            )


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.data_ingestion`
    ingestor = DataIngestor()
    data = ingestor.load()
    print(data.head())
