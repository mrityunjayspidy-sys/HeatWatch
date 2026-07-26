# ============================================================
# India Real Climate & Satellite Data Processor & ML Model Trainer
# Processes raw datasets in data/:
#  - india_climate_hourly_2014_2023_v20260321_223327.parquet (4.12M hourly climate records)
#  - india_cities_dataset_2021_2025.csv (5,587 satellite LST, NDVI, NDBI records)
#  - final_dataset.csv (105,980 regional climate records)
# ============================================================

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_raw_datasets():
    print("=" * 60)
    print("STEP 1: Processing Raw Datasets in data/ Directory")
    print("=" * 60)

    # 1. Load Hourly Climate Parquet Data (4.12M records)
    parquet_path = os.path.join(DATA_DIR, "india_climate_hourly_2014_2023_v20260321_223327.parquet")
    print(f"Loading hourly climate dataset: {parquet_path}")
    df_hourly = pd.read_parquet(parquet_path)
    print(f" -> Ingested {len(df_hourly):,} hourly records across {df_hourly['city'].nunique()} Indian cities.")

    # Feature extraction from datetime
    df_hourly["datetime"] = pd.to_datetime(df_hourly["datetime"])
    df_hourly["hour"] = df_hourly["datetime"].dt.hour
    df_hourly["month"] = df_hourly["datetime"].dt.month
    df_hourly["year"] = df_hourly["datetime"].dt.year
    df_hourly["is_daytime"] = df_hourly["hour"].between(6, 18).astype(int)

    # 2. Load Satellite Land-Use CSV Dataset (5,587 records)
    cities_csv_path = os.path.join(DATA_DIR, "india_cities_dataset_2021_2025.csv")
    print(f"Loading satellite land-use dataset: {cities_csv_path}")
    df_sat = pd.read_csv(cities_csv_path, encoding="latin1")
    
    # Standardize column names
    col_map = {}
    for c in df_sat.columns:
        if "Elevation" in c:
            col_map[c] = "elevation_m"
        elif "LST" in c:
            col_map[c] = "lst_celsius"
        elif "City" in c:
            col_map[c] = "city"
        elif "State" in c:
            col_map[c] = "state"
        elif "Latitude" in c:
            col_map[c] = "lat"
        elif "Longitude" in c:
            col_map[c] = "lon"
        elif "Year" in c:
            col_map[c] = "year"
        elif "Month" in c:
            col_map[c] = "month"
    df_sat.rename(columns=col_map, inplace=True)
    print(f" -> Ingested {len(df_sat):,} satellite records with NDVI, NDBI, NDWI, Elevation.")

    # 3. Load Regional Heat CSV Dataset (105,980 records)
    final_csv_path = os.path.join(DATA_DIR, "final_dataset.csv")
    print(f"Loading regional climate dataset: {final_csv_path}")
    df_final = pd.read_csv(final_csv_path, encoding="latin1")
    print(f" -> Ingested {len(df_final):,} regional climate records.")

    # 4. Merge & Feature Engineering Matrix
    print("\nMerging climate parameters with satellite land-use features...")
    
    # Aggregated monthly climate stats per city
    climate_monthly = df_hourly.groupby(["city", "month"]).agg({
        "temperature_2m_c": ["mean", "max", "min"],
        "relative_humidity_pct": "mean",
        "wind_speed_10m_kmh": "mean",
        "precipitation_mm": "mean",
        "lat": "first",
        "lon": "first"
    }).reset_index()

    climate_monthly.columns = [
        "city", "month",
        "temp_mean", "temp_max", "temp_min",
        "humidity_mean", "wind_mean", "precip_mean",
        "lat", "lon"
    ]

    # Combine satellite dataset with monthly climate averages
    df_sat["month"] = pd.to_numeric(df_sat["month"], errors="coerce").fillna(1).astype(int)
    df_sat["city"] = df_sat["city"].astype(str).str.strip().str.title()
    climate_monthly["city"] = climate_monthly["city"].astype(str).str.strip().str.title()
    climate_monthly["month"] = climate_monthly["month"].astype(int)

    merged = pd.merge(df_sat, climate_monthly, on=["city", "month"], how="left", suffixes=("", "_clim"))

    
    # Fill missing values with city or global averages
    merged["lat"] = merged["lat"].fillna(merged["lat_clim"]).fillna(20.5937)
    merged["lon"] = merged["lon"].fillna(merged["lon_clim"]).fillna(78.9629)
    merged["temp_mean"] = merged["temp_mean"].fillna(merged["lst_celsius"] - 2.5)
    merged["temp_max"] = merged["temp_max"].fillna(merged["lst_celsius"] + 3.0)
    merged["humidity_mean"] = merged["humidity_mean"].fillna(55.0)
    merged["wind_mean"] = merged["wind_mean"].fillna(7.5)
    merged["elevation_m"] = merged["elevation_m"].fillna(150.0)
    merged["NDVI"] = merged["NDVI"].fillna(0.15)
    merged["NDBI"] = merged["NDBI"].fillna(0.10)
    merged["NDWI"] = merged["NDWI"].fillna(0.0)

    # Compute UHI intensity differential (Urban LST - Ambient Mean)
    merged["uhi_intensity"] = (merged["lst_celsius"] - merged["temp_mean"]).clip(lower=0.2, upper=6.5)

    # Save processed matrix
    processed_csv = os.path.join(PROCESSED_DIR, "processed_india_uhi_dataset.csv")
    merged.to_csv(processed_csv, index=False)
    print(f" -> Saved processed training dataset to: {processed_csv} ({len(merged):,} rows)")
    return merged

def train_lst_model():
    df = process_raw_datasets()

    print("\n" + "=" * 60)
    print("STEP 2: Training ML Model Ensemble on Real Processed Data")
    print("=" * 60)

    feature_cols = [
        "lat", "lon", "month", "elevation_m",
        "NDVI", "NDBI", "NDWI",
        "temp_mean", "temp_max", "humidity_mean", "wind_mean"
    ]

    target_col = "lst_celsius"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1)
    gb = HistGradientBoostingRegressor(max_iter=150, max_depth=10, learning_rate=0.08, random_state=42)

    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)

    rf_preds = rf.predict(X_test_scaled)
    gb_preds = gb.predict(X_test_scaled)
    ensemble_preds = 0.55 * rf_preds + 0.45 * gb_preds

    mae = mean_absolute_error(y_test, ensemble_preds)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    r2 = r2_score(y_test, ensemble_preds)

    print(f" -> Model Metrics on Real Test Split:")
    print(f"    R² Score:  {r2:.4f}")
    print(f"    MAE:       {mae:.3f} °C")
    print(f"    RMSE:      {rmse:.3f} °C")

    # City stats dictionary
    city_stats = {}
    for city, grp in df.groupby("city"):
        city_stats[city] = {
            "lat": float(grp["lat"].mean()),
            "lon": float(grp["lon"].mean()),
            "avg_lst": float(grp["lst_celsius"].mean()),
            "avg_ndvi": float(grp["NDVI"].mean()),
            "avg_ndbi": float(grp["NDBI"].mean()),
            "avg_tree_cover": 0.18,
            "total_cells": 100
        }

    # Save trained models payload
    model_payload = {
        "rf_model": rf,
        "gb_model": gb,
        "model": rf,
        "scaler": scaler,
        "features": feature_cols,
        "feature_names": feature_cols,
        "city_stats": city_stats,
        "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse), "samples": len(X_train)},
        "dataset_source": "REAL_INDIA_CLIMATE_DATASETS"
    }

    model_path1 = os.path.join(ROOT_DIR, "backend", "lst_model.joblib")
    model_path2 = os.path.join(ROOT_DIR, "uhi_intensity_model.pkl")


    joblib.dump(model_payload, model_path1)
    joblib.dump(model_payload, model_path2)

    print(f" -> Successfully saved trained models to:\n    - {model_path1}\n    - {model_path2}")
    return model_payload

if __name__ == "__main__":
    train_lst_model()
