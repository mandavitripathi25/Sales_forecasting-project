# Sales Forecasting Pipeline

A modular ensemble forecasting pipeline: ingest sales data, find the
right level of the hierarchy to forecast at, fit multiple time series
models, blend them into an ensemble, and disaggregate the blended
forecast back down to a granular (e.g. SKU x store) level.

## Structure

```
main.ipynb                     # orchestrator — run this end to end
src/
  __init__.py                  # package initializer
  config.py                    # paths, columns, model params — edit this first
  utils.py                     # clean_column_names, error metrics, logging
  data_ingestion.py            # DataIngestor — loads data/sales_data.xlsx
  data_preparation.py          # DataPreparer, SparsityAnalyzer — builds master table, picks forecast level
  forecasting_models.py        # ArimaModel, EtsModel, ProphetModel, NaiveSeasonalModel
  ensemble_model.py            # EnsembleForecaster — backtests + blends models
  decomposition_engine.py      # DecompositionEngine — allocates total forecast down to granular level
data/
  sales_data.xlsx              # <- put your source Excel file here
outputs/
  final_forecast.csv           # written by main.ipynb
  model_metrics.csv            # written by main.ipynb
```

## Setup

```bash
pip install -r requirements.txt
```

`prophet` can be slow/finicky to install on some platforms. If you'd
rather skip it, remove `"prophet"` from `ENSEMBLE_MODELS` in
`src/config.py` — the pipeline runs fine on ARIMA + ETS alone.

## Getting your data in

1. Drop your Excel file at `data/sales_data.xlsx` (or point
   `config.RAW_DATA_FILE` elsewhere).
2. Edit `src/config.py`:
   - `DATE_COL`, `TARGET_COL` — must match your cleaned column names
     (columns get lowercased/underscored by `clean_column_names`).
   - `HIERARCHY_COLS` — the granular dimensions you want to forecast
     and later disaggregate to (e.g. `region`, `store_id`, `sku`).
3. Run `main.ipynb` top to bottom.

## What main.ipynb does

1. `DataIngestor` loads and validates the raw Excel data.
2. `DataPreparer` cleans it and builds a dense master table (every
   period present per series, zero-filled gaps).
3. `SparsityAnalyzer` walks the hierarchy to find the most granular
   level that still has enough history/density to forecast reliably.
4. `EnsembleForecaster` backtests ARIMA / ETS / Prophet at that level,
   derives error-weighted blend weights, and produces the final
   ensemble forecast.
5. `DecompositionEngine` splits the ensemble forecast back down to the
   full granular hierarchy using historical shares.
6. Results are written to `outputs/final_forecast.csv` and
   `outputs/model_metrics.csv`.

## Notes / things you'll likely need to adjust

- The schema assumed here (`date`, `sales`, `region`, `store_id`,
  `product_category`, `sku`) is a placeholder — update
  `config.HIERARCHY_COLS` / `DATE_COL` / `TARGET_COL` to match your
  real Excel headers.
- `FREQUENCY` in `config.py` assumes monthly data (`"MS"`). Change to
  `"W"`, `"D"`, etc. if your data is weekly/daily.
- The decomposition step currently does simple top-down proportional
  allocation. If your business has structural breaks (e.g. a store
  opened last quarter), consider switching `DecompositionEngine`'s
  `method` to `"recent_share"` (default) rather than `"average_share"`.
