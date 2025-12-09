import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Solar Irradiation Forecast", layout="wide")
st.title("Solar Irradiation Forecast Dashboard")
st.write("Upload a CSV with columns Year, Month, Day, Hour, Minute and irradiance/weather variables. The app will forecast GHI at t+3h, t+24h, t+7d.")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Relative H": "Relative Humidity",
        "Solar Zenit": "Solar Zenith",
        "Surface All": "Surface Albedo",
        "Clearsky D": "Clearsky DHI",
        "Clearsky G": "Clearsky GHI",
        "Clearsky D 2": "Clearsky DNI",
    }
    return df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Year", "Month", "Day", "Hour", "Minute"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required time column: {c}")
    
    df = df.copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Day"] = df["Day"].astype(int)
    df["Hour"] = df["Hour"].astype(int)
    df["Minute"] = df["Minute"].astype(int)
    
    dt = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]]
        .rename(columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour", "Minute": "minute"}),
        errors="coerce",
    )
    df.index = dt
    df = df[df.index.notna()].sort_index()
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(["N/A", "NA", ""], pd.NA)
    for col in df.columns:
        if col not in ["Year", "Month", "Day", "Hour", "Minute"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_features(df: pd.DataFrame):
    # Determine target column
    if "GHI" in df.columns:
        target = "GHI"
    elif "Clearsky GHI" in df.columns:
        target = "Clearsky GHI"
    else:
        raise ValueError("Neither GHI nor Clearsky GHI found. Add one of these columns.")

    # Check for non-null target values
    if df[target].isna().sum() == len(df):
        raise ValueError(f"Target column '{target}' has no valid values.")
    
    horizons = {"tplus3": 3, "tplus24": 24, "tplus168": 168}

    # Future targets - only create where valid data exists
    for name, h in horizons.items():
        df[f"{target}_{name}"] = df[target].shift(-h)

    # Target lags (reduced to avoid losing too many rows)
    for l in [1, 2, 3, 6, 12, 24, 48]:
        df[f"{target}_lag{l}"] = df[target].shift(l)

    # Exogenous candidates
    exog_candidates = [
        "Temperature",
        "Relative Humidity",
        "Pressure",
        "Wind Speed",
        "Dew Point",
        "Solar Zenith",
        "Surface Albedo",
        "DHI",
        "DNI",
        "Clearsky DHI",
        "Clearsky DNI",
        "Clearsky GHI",
    ]
    exog_cols = [c for c in exog_candidates if c in df.columns and c != target]
    for c in exog_cols:
        for l in [1, 6, 12]:
            df[f"{c}_lag{l}"] = df[c].shift(l)

    # Rolling stats on target
    for w in [3, 6, 24]:
        df[f"{target}_roll{w}_mean"] = df[target].rolling(w).mean()
        df[f"{target}_roll{w}_std"] = df[target].rolling(w).std()

    # Calendar features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Identify feature columns (exclude target and shifted targets)
    feature_cols = [
        c
        for c in df.columns
        if c not in ([target] + [f"{target}_{k}" for k in horizons.keys()])
    ]
    
    if len(feature_cols) == 0:
        raise ValueError("No valid features available after processing. Ensure input CSV has predictor columns.")
    
    # Drop rows with NaN only in feature columns, allowing NaN in target horizons
    df_model = df.dropna(subset=feature_cols).copy()
    
    # Also drop rows where ALL target horizons are NaN
    target_horizon_cols = [f"{target}_{k}" for k in horizons.keys()]
    df_model = df_model.dropna(subset=target_horizon_cols, how='all')
    
    if len(df_model) < 50:
        raise ValueError(f"Insufficient data: {len(df_model)} rows after feature engineering. Started with {len(df)} rows. Need at least 50 rows to train models.")
    
    st.info(f"Feature engineering: {len(df)} → {len(df_model)} rows ({len(feature_cols)} features)")
    
    return df_model, feature_cols, target, horizons


def train_and_predict(df_model: pd.DataFrame, feature_cols, target, horizons):
    split_idx = int(len(df_model) * 0.8) if len(df_model) > 200 else int(len(df_model) * 0.7)
    train_df = df_model.iloc[:split_idx]
    valid_df = df_model.iloc[split_idx:]

    results = {}
    preds = {}
    models = {}
    importances = {}

    def train_one(hname: str):
        y_col = f"{target}_{hname}"
        X_train_all = train_df[feature_cols].copy()
        y_train_all = train_df[y_col].copy()
        X_valid_all = valid_df[feature_cols].copy()
        y_valid_all = valid_df[y_col].copy()
        
        # Remove rows with NaN targets for this specific horizon
        valid_mask_train = y_train_all.notna()
        valid_mask_valid = y_valid_all.notna()
        
        X_train = X_train_all[valid_mask_train]
        y_train = y_train_all[valid_mask_train]
        X_valid = X_valid_all[valid_mask_valid]
        y_valid = y_valid_all[valid_mask_valid]
        
        if len(y_train) == 0 or len(y_valid) == 0:
            st.warning(f"Skipping horizon {hname}: insufficient valid samples")
            return False
        
        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            eval_metric="rmse",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        pred_valid = model.predict(X_valid)
        mae = float(np.mean(np.abs(pred_valid - y_valid)))
        rmse = float(np.sqrt(np.mean((pred_valid - y_valid) ** 2)))
        r2 = float(1 - (np.sum((pred_valid - y_valid) ** 2) / np.sum((y_valid - y_valid.mean()) ** 2)))
        results[hname] = {"mae": mae, "rmse": rmse, "r2": r2}
        preds[hname] = pd.Series(pred_valid, index=X_valid.index)
        models[hname] = model
        
        # Get feature importances
        importances[hname] = dict(zip(feature_cols, model.feature_importances_))
        return True

    for hname in ["tplus3", "tplus24", "tplus168"]:
        train_one(hname)

    return results, preds, models, importances, train_df, valid_df


uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_raw)} rows.")
        # Show columns on a single line to avoid large vertical output
        st.write("Detected columns:", ", ".join(map(str, df_raw.columns)))

        df = normalize_headers(df_raw)
        df = clean_numeric(df)
        df = to_datetime_index(df)

        st.write("Preview (first 10 rows):")
        st.dataframe(df.head(10))

        df_model, feature_cols, target, horizons = build_features(df)
        st.info(f"Target column: {target}. Training rows: {len(df_model)}")

        results, preds, models, importances, train_df, valid_df = train_and_predict(
            df_model, feature_cols, target, horizons
        )

        st.subheader("Validation metrics")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df, use_container_width=True)

        # Attach predictions back to full df
        out = df.copy()
        for hname, s in preds.items():
            out.loc[s.index, f"{target}_pred_{hname}"] = s

        # Download button for CSV
        out_to_save = out.reset_index(names="timestamp")
        buf = io.StringIO()
        out_to_save.to_csv(buf, index=False)
        st.download_button(
            "Download predictions CSV",
            data=buf.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )

        # Plots
        st.subheader("Validation: Actual vs Predicted")
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        titles = ["Next 3 hours", "Next 24 hours", "Next 7 days"]
        for ax, hname, ttl in zip(
            axes, ["tplus3", "tplus24", "tplus168"], titles
        ):
            idx = preds[hname].index
            actual = df.loc[idx, f"{target}_{hname}"]
            predicted = out.loc[idx, f"{target}_pred_{hname}"]
            
            ax.plot(actual.index, actual.values, label="Actual", lw=1.5, alpha=0.8)
            ax.plot(predicted.index, predicted.values, label="Predicted", lw=1.5, alpha=0.8)
            ax.set_title(f"{ttl} (MAE: {results[hname]['mae']:.2f}, RMSE: {results[hname]['rmse']:.2f})")
            ax.set_ylabel(f"{target} (W/m²)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # Feature importance plots
        st.subheader("Feature Importance by Horizon")
        fig_imp, axes_imp = plt.subplots(1, 3, figsize=(16, 5))
        for ax_i, hname, ttl in zip(
            axes_imp, ["tplus3", "tplus24", "tplus168"], 
            ["Next 3 hours", "Next 24 hours", "Next 7 days"]
        ):
            top_n = 15
            imp_dict = importances[hname]
            top_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            feat_names, feat_vals = zip(*top_features)
            
            ax_i.barh(feat_names, feat_vals, color="steelblue")
            ax_i.set_xlabel("Importance")
            ax_i.set_title(ttl)
            ax_i.invert_yaxis()
        
        fig_imp.tight_layout()
        st.pyplot(fig_imp)
        
        # Residual analysis
        st.subheader("Residual Analysis")
        fig_res, axes_res = plt.subplots(1, 3, figsize=(16, 5))
        for ax_r, hname, ttl in zip(
            axes_res, ["tplus3", "tplus24", "tplus168"],
            ["Next 3 hours", "Next 24 hours", "Next 7 days"]
        ):
            idx = preds[hname].index
            actual = df.loc[idx, f"{target}_{hname}"].values
            predicted = out.loc[idx, f"{target}_pred_{hname}"].values
            residuals = actual - predicted
            
            ax_r.scatter(predicted, residuals, alpha=0.5, s=20)
            ax_r.axhline(y=0, color='r', linestyle='--', lw=1)
            ax_r.set_xlabel("Predicted")
            ax_r.set_ylabel("Residuals (W/m²)")
            ax_r.set_title(ttl)
            ax_r.grid(True, alpha=0.3)
        
        fig_res.tight_layout()
        st.pyplot(fig_res)

    except Exception as e:
        st.error(f"Error: {e}")
