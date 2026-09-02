
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
=======
# Sales_forecasting-project
End-to-end Sales Forecasting Pipeline using Python, ARIMA, XGBoost, and Ensemble Models with hierarchical forecasting, real-world data validation, and automated preprocessing.

# 📈 Sales Forecasting Pipeline

An end-to-end Sales Forecasting Pipeline built with Python for forecasting future sales using classical time series models and machine learning. The project supports hierarchical forecasting, automated preprocessing, ensemble modeling, model evaluation, and visualization.

---

## 🚀 Features

- Automated data ingestion from Excel
- Data cleaning and preprocessing
- Monthly time-series aggregation
- Hierarchical forecasting
- Multiple forecasting models
  - Moving Average
  - Linear Regression
  - ARIMA
  - SARIMA
  - Holt-Winters (ETS)
  - Random Forest
  - XGBoost
- Ensemble Forecasting
- Forecast visualization
- Model comparison using MAE, RMSE, and MAPE
- Export forecasts to CSV

---

## 📂 Project Structure

```
sales_forecasting_pipeline/
│
├── data/
│   └── sales_data.xlsx
│
├── outputs/
│   ├── final_forecast.csv
│   └── model_metrics.csv
│
├── src/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_preparation.py
│   ├── forecasting_models.py
│   ├── ensemble_model.py
│   ├── decomposition_engine.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
│
├── main.ipynb
├── requirements.txt
└── README.md
```

---

# Dataset

The pipeline was tested using a real-world order-level sales dataset containing:

- **1,001 rows**
- **Period:** January 2019 – December 2020
- **Columns**
  - country
  - order_value_EUR
  - cost
  - date
  - category
  - customer_name
  - sales_manager
  - sales_rep
  - device_type
  - order_id

---

# Configuration

The project was configured specifically for the dataset:

```python
TARGET_COL = "order_value_eur"

DATE_COL = "date"

HIERARCHY_COLS = [
    "country",
    "category",
    "device_type"
]

FORECAST_HORIZON = 6

TEST_HORIZON = 3

MIN_HISTORY_PERIODS = 12
```

ARIMA seasonal components are disabled by default because two years of historical data is insufficient for reliable yearly seasonality estimation.

---

# Bugs Fixed

While testing the pipeline on the actual dataset, two important data-processing bugs were identified and fixed.

## 1. Hierarchical Sparsity Bug

### Problem

The sparsity checker incorrectly counted historical periods after collapsing hierarchy levels, causing the forecasting level selection to always fall back to the **TOTAL** level.

### Solution

Updated the sparsity evaluation logic to correctly calculate historical periods for every hierarchy level.

### Result

The pipeline now correctly selects the most suitable hierarchy level for forecasting.

---

## 2. Monthly Aggregation Bug

### Problem

The dataset contained daily order dates, but the aggregation process matched records against month-start dates using exact date equality.

As a result, most transactions were treated as missing and incorrectly filled with zeros.

### Solution

Reworked the aggregation logic to bucket daily transactions into their corresponding calendar month before aggregation.

### Result

Monthly sales totals are now calculated accurately and used for forecasting.

---

# Validation Results

After applying the fixes:

- Monthly aggregation is accurate
- Hierarchical forecasting works correctly
- Forecasts reconcile back to the top-level totals
- The pipeline automatically selects **Country** as the forecasting level because approximately **87%** of countries contain sufficient historical data.

---

# Models Included

- Moving Average
- Linear Regression
- ARIMA
- SARIMA
- Holt-Winters
- Random Forest
- XGBoost
- Ensemble Forecast

---

# Output Files

Running the notebook generates:

```
outputs/
├── final_forecast.csv
└── model_metrics.csv
```

**final_forecast.csv**

Contains future sales predictions.

**model_metrics.csv**

Contains evaluation metrics including:

- MAE
- RMSE
- MAPE

---

# Installation

```bash
git clone https://github.com/yourusername/sales_forecasting_pipeline.git

cd sales_forecasting_pipeline

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

---

# Run

Open

```
main.ipynb
```

Select the project Python environment as the notebook kernel.

Run all notebook cells.

Generated outputs will be available in the **outputs/** directory.

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- XGBoost
- Matplotlib
- Jupyter Notebook

---

# Future Improvements

- Prophet integration
- LightGBM forecasting
- AutoML model selection
- Hyperparameter optimization
- Interactive dashboard using Streamlit
- Cloud deployment

---

# Key Achievements

- Built a complete end-to-end forecasting pipeline.
- Implemented hierarchical forecasting with automatic level selection.
- Fixed real-world aggregation and hierarchy bugs discovered during testing.
- Validated the pipeline on an actual sales dataset.
- Generated reconciled forecasts and evaluation metrics for multiple forecasting models.

---

## Author

**Mandavi Tripathi**

AI Engineer | Data Science | Machine Learning | Time Series Forecasting
>>>>>>> b12e2c69998b2ba672f12d959e066270516018e4
