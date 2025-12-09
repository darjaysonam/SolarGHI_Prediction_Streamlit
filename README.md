# Solar Irradiation Forecast Dashboard

This Streamlit app trains XGBoost models to forecast Global Horizontal Irradiance (GHI) at t+3h, t+24h and t+7d from a CSV of historical observations.

Main file: `app.py`

---

## Features

- Forecast horizons: 3-hour, 24-hour and 7-day ahead GHI forecasts.
- Models: XGBoost regressors trained per horizon.
- Diagnostics: Validation metrics (MAE, RMSE, R²), actual vs predicted plots, feature importance and residual analysis.
- Output: Downloadable CSV with prediction columns and timestamps.

## Requirements

- Python 3.8+ recommended
- Install dependencies from `requirements.txt` (core packages used: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`)

## Install

Open PowerShell and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Open the URL provided by Streamlit (usually `http://localhost:8501`).

## Input CSV format

- Required time columns: `Year`, `Month`, `Day`, `Hour`, `Minute` (these must exist and be spelled exactly).
- Required target column: `GHI` (preferred) or `Clearsky GHI` (fallback).
- Recommended optional columns: `Temperature`, `Relative Humidity`, `Pressure`, `Wind Speed`, `Dew Point`, `Solar Zenith` (or `Solar Zenith Angle`), `Surface Albedo`, `DHI`, `DNI`, `Clearsky DHI`, `Clearsky DNI`.

Example header (minimum):

```
Year,Month,Day,Hour,Minute,GHI
```

Notes:
- The app normalizes some header variants (e.g., `Relative H` → `Relative Humidity`).
- The app creates lag features, rolling statistics and calendar features automatically.

## What the app does with your data

- Parses the time columns into a DateTime index.
- Builds selective lag features and rolling statistics to avoid excessive row loss from NaN values.
- Creates per-horizon target columns (`GHI_tplus3`, `GHI_tplus24`, `GHI_tplus168`) by shifting the target forward.
- Drops rows missing required predictor features, but allows horizon targets to be NaN (each horizon is trained on available samples).

## Output

- Predictions CSV: download `predictions.csv` containing the timestamp index and prediction columns such as `GHI_pred_tplus3`, `GHI_pred_tplus24`, `GHI_pred_tplus168`.
- In-app plots and a metrics table are shown for validation.

## Common issues & fixes

- "Missing required time column" — make sure `Year, Month, Day, Hour, Minute` are present.
- "Insufficient data: 0 rows after feature engineering" — usually caused by many NaNs in predictor columns or large gaps in timestamps. Fix by:
  - Ensuring `GHI` has numeric values.
  - Filling or interpolating missing predictor values before upload.
  - Providing continuous timestamped data (no large gaps) or reducing lags in `build_features`.
- If the detected columns still render as multi-line in your browser, refresh (Ctrl+F5). The app now prints columns as a single comma-separated line.

## Customization

- Adjust selected lags and horizon definitions in `app.py` (`build_features` and `train_and_predict`).
- Persist models using `joblib` if you want to save trained models to disk.
- Add hyperparameter tuning or early stopping to improve performance.

## Example CSV

A minimal example file is included as `example_minimal.csv`. It contains the required datetime columns and a small sample of `GHI` and `Temperature` values for quick testing.

## Acknowledgement

Parts of this code were adapted and modified from a Kaggle Notebook:
"XGBoost Regression model" by Neeraj Mohan, licensed under the Apache 2.0 License.
