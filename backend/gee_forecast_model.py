"""
AI Future Temperature Forecasting Model
=========================================
Predicts future land surface temperatures (7-day and 30-day horizons)
for each grid cell using historical data + current satellite observations.

Training data: Open-Meteo Archive API historical hourly temperatures.
Features: temporal (month, day-of-year, solar declination), spatial (NDVI, NDBI),
           and thermal inertia (built-up heat retention vs vegetation cooling).
"""

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.joblib")

# ─── In-Memory Cache for Forecasts ───
_forecast_cache = {}
FORECAST_CACHE_TTL = 600  # 10 minutes


class TemperatureForecastModel:
    """
    AI model that predicts future temperatures at grid cell level.
    Uses historical Open-Meteo data + current satellite observations.
    """

    FEATURE_COLS = [
        "current_lst", "ndvi", "ndbi", "ndwi",
        "month", "day_of_year", "solar_declination",
        "thermal_inertia", "vegetation_cooling_buffer",
        "hour_sin", "hour_cos",
        "humidity", "wind_speed"
    ]

    def __init__(self):
        self.model_7d = None
        self.model_30d = None
        self.is_trained = False
        self.metrics = {}
        self._load_model()

    def _load_model(self):
        """Load pre-trained forecast model if available."""
        if os.path.exists(FORECAST_MODEL_PATH):
            try:
                try:
                    import sklearn._loss._loss
                    sys.modules['_loss'] = sklearn._loss._loss
                except Exception:
                    pass
                payload = joblib.load(FORECAST_MODEL_PATH)
                self.model_7d = payload.get("model_7d")
                self.model_30d = payload.get("model_30d")
                self.metrics = payload.get("metrics", {})
                self.is_trained = True
                print(f"[Forecast] ✓ Loaded trained forecast model from {FORECAST_MODEL_PATH}")
            except Exception as e:
                print(f"[Forecast] Failed to load model: {e}")
                self.is_trained = False

    def train_from_openmeteo_archive(self, lat: float, lon: float,
                                      start_date: str = "2023-01-01",
                                      end_date: str = "2024-06-30") -> dict:
        """
        Train the forecast model using Open-Meteo Archive API historical data.
        Fetches hourly temperature data and derives temporal + trend features.
        """
        print(f"[Forecast] Fetching historical data from Open-Meteo Archive...")
        print(f"[Forecast]   Location: ({lat}, {lon}), Period: {start_date} to {end_date}")

        # Fetch historical hourly data
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,soil_temperature_0cm,"
                      "shortwave_radiation,cloud_cover",
            "timezone": "auto"
        }

        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        hourly_data = resp.json().get("hourly", {})

        df = pd.DataFrame(hourly_data)
        if df.empty or "time" not in df.columns:
            raise ValueError("No historical data returned from Open-Meteo Archive")

        df["time"] = pd.to_datetime(df["time"])
        df = df.dropna(subset=["temperature_2m"])

        print(f"[Forecast]   Fetched {len(df):,} hourly records")

        # ── Feature Engineering ──
        df["hour"] = df["time"].dt.hour
        df["month"] = df["time"].dt.month
        df["day_of_year"] = df["time"].dt.dayofyear
        df["is_daytime"] = df["hour"].between(6, 18).astype(int)

        # Solar declination angle (seasonality)
        df["solar_declination"] = np.sin((df["day_of_year"] - 80) * 2 * np.pi / 365.0) * 23.44

        # Circular hour encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

        # Surface temperature estimate (soil temp or adjusted air temp)
        df["surface_temp"] = df["soil_temperature_0cm"].fillna(df["temperature_2m"] + 2.0)

        # Daily aggregation for training targets
        daily = df.groupby(df["time"].dt.date).agg({
            "temperature_2m": "max",
            "surface_temp": "max",
            "relative_humidity_2m": "mean",
            "wind_speed_10m": "mean",
            "shortwave_radiation": "mean",
            "cloud_cover": "mean",
            "month": "first",
            "day_of_year": "first",
            "solar_declination": "first"
        }).reset_index()
        daily.columns = [
            "date", "air_temp_max", "surface_temp_max",
            "humidity_mean", "wind_mean", "radiation_mean", "cloud_mean",
            "month", "day_of_year", "solar_declination"
        ]

        # Rolling averages for trend features
        daily["temp_7d_rolling"] = daily["surface_temp_max"].rolling(7, min_periods=1).mean()
        daily["temp_30d_rolling"] = daily["surface_temp_max"].rolling(30, min_periods=1).mean()
        daily["temp_trend_7d"] = daily["surface_temp_max"] - daily["temp_7d_rolling"]

        # Create forecast targets: what is the temperature 7 days and 30 days ahead?
        daily["target_7d"] = daily["surface_temp_max"].shift(-7)
        daily["target_30d"] = daily["surface_temp_max"].shift(-30)

        # Simulated spatial features (these will be replaced by real satellite data at prediction time)
        daily["ndvi"] = 0.15  # Will be overridden during prediction
        daily["ndbi"] = 0.10
        daily["ndwi"] = -0.05
        daily["thermal_inertia"] = daily["ndbi"] * 0.4 - daily["ndvi"] * 0.35
        daily["vegetation_cooling_buffer"] = daily["ndvi"] * 2.5

        # Feature matrix
        feature_cols_train = [
            "surface_temp_max", "ndvi", "ndbi", "ndwi",
            "month", "day_of_year", "solar_declination",
            "thermal_inertia", "vegetation_cooling_buffer",
            "humidity_mean", "wind_mean", "radiation_mean", "cloud_mean",
            "temp_7d_rolling", "temp_trend_7d"
        ]

        # Drop rows with NaN targets
        train_7d = daily.dropna(subset=["target_7d"]).copy()
        train_30d = daily.dropna(subset=["target_30d"]).copy()

        print(f"[Forecast]   Training samples: {len(train_7d)} (7-day), {len(train_30d)} (30-day)")

        # ── Train 7-day forecast model ──
        X_7d = train_7d[feature_cols_train].fillna(0)
        y_7d = train_7d["target_7d"]

        self.model_7d = HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, learning_rate=0.08,
            min_samples_leaf=10, random_state=42
        )
        self.model_7d.fit(X_7d, y_7d)

        # Evaluate 7-day model
        pred_7d = self.model_7d.predict(X_7d)
        mae_7d = mean_absolute_error(y_7d, pred_7d)
        r2_7d = r2_score(y_7d, pred_7d)

        # ── Train 30-day forecast model ──
        X_30d = train_30d[feature_cols_train].fillna(0)
        y_30d = train_30d["target_30d"]

        self.model_30d = HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, learning_rate=0.06,
            min_samples_leaf=15, random_state=42
        )
        self.model_30d.fit(X_30d, y_30d)

        pred_30d = self.model_30d.predict(X_30d)
        mae_30d = mean_absolute_error(y_30d, pred_30d)
        r2_30d = r2_score(y_30d, pred_30d)

        self.metrics = {
            "7d": {"r2": round(r2_7d, 4), "mae": round(mae_7d, 2)},
            "30d": {"r2": round(r2_30d, 4), "mae": round(mae_30d, 2)},
            "training_samples": len(daily),
            "training_period": f"{start_date} to {end_date}",
            "trained_at": datetime.datetime.now().isoformat()
        }

        # Save model
        payload = {
            "model_7d": self.model_7d,
            "model_30d": self.model_30d,
            "feature_cols": feature_cols_train,
            "metrics": self.metrics,
            "training_location": {"lat": lat, "lon": lon}
        }
        joblib.dump(payload, FORECAST_MODEL_PATH)
        self.is_trained = True

        print(f"[Forecast] ✓ Models trained and saved!")
        print(f"[Forecast]   7-day: R²={r2_7d:.4f}, MAE={mae_7d:.2f}°C")
        print(f"[Forecast]   30-day: R²={r2_30d:.4f}, MAE={mae_30d:.2f}°C")

        return self.metrics

    def predict_grid_forecast(self, grid_cells: list, horizon: int = 7) -> list:
        """
        Predict future temperatures for each grid cell.

        Args:
            grid_cells: List of dicts with current grid cell data
                        (lst_mean, ndvi_mean, ndbi_mean, etc.)
            horizon: Forecast horizon in days (7 or 30)

        Returns:
            List of dicts with current + predicted future temperatures
        """
        now = datetime.datetime.now()
        current_month = now.month
        day_of_year = now.timetuple().tm_yday
        solar_declination = np.sin((day_of_year - 80) * 2 * np.pi / 365.0) * 23.44

        results = []

        for cell in grid_cells:
            current_lst = float(cell.get("lst_mean", 32.0))
            ndvi = float(cell.get("ndvi_mean", 0.12))
            ndbi = float(cell.get("ndbi_mean", 0.08))
            ndwi = float(cell.get("ndwi_mean", 0.0))
            humidity = float(cell.get("openmeteo_humidity", 55.0))
            wind = float(cell.get("openmeteo_wind_kmh", 6.0))

            # Derived features
            thermal_inertia = ndbi * 0.4 - ndvi * 0.35
            veg_cooling = ndvi * 2.5

            if self.is_trained:
                # Use trained ML model
                features = np.array([[
                    current_lst, ndvi, ndbi, ndwi,
                    current_month, day_of_year, solar_declination,
                    thermal_inertia, veg_cooling,
                    humidity, wind,
                    0.0,  # radiation (placeholder)
                    0.0,  # cloud cover (placeholder)
                    current_lst,  # rolling avg approximated as current
                    0.0   # trend approximated as 0
                ]])

                if horizon <= 7 and self.model_7d is not None:
                    forecast_lst = float(self.model_7d.predict(features)[0])
                elif self.model_30d is not None:
                    forecast_lst = float(self.model_30d.predict(features)[0])
                else:
                    forecast_lst = self._analytical_forecast(current_lst, ndvi, ndbi, horizon)
            else:
                # Analytical fallback
                forecast_lst = self._analytical_forecast(current_lst, ndvi, ndbi, horizon)

            forecast_lst = round(np.clip(forecast_lst, 10.0, 55.0), 2)
            delta = round(forecast_lst - current_lst, 2)

            # Confidence interval (wider for longer horizons)
            ci_half = 1.2 if horizon <= 7 else 2.5
            if not self.is_trained:
                ci_half *= 1.5  # Wider uncertainty for analytical model

            results.append({
                **cell,
                "current_lst": current_lst,
                f"forecast_lst_{horizon}d": forecast_lst,
                "forecast_delta": delta,
                "forecast_ci_low": round(forecast_lst - ci_half, 2),
                "forecast_ci_high": round(forecast_lst + ci_half, 2),
                "forecast_horizon_days": horizon,
                "forecast_model": "TRAINED_ML" if self.is_trained else "ANALYTICAL"
            })

        return results

    @staticmethod
    def _analytical_forecast(current_lst: float, ndvi: float, ndbi: float,
                              horizon: int) -> float:
        """
        Physics-based analytical forecast when no trained model is available.
        Uses seasonal trends, built-up heat retention, and vegetation cooling.
        """
        now = datetime.datetime.now()
        current_month = now.month

        # Seasonal solar declination warming factor
        seasonal_factor = np.sin((current_month - 3) * np.pi / 6.0) * 1.5

        # Built-up thermal inertia vs vegetation cooling
        heat_retention = (ndbi * 0.4 - ndvi * 0.35)
        heat_retention = np.clip(heat_retention, -0.5, 0.5)

        if horizon <= 7:
            delta = 0.35 + seasonal_factor * 0.2 + heat_retention * 0.8
        else:
            delta = 1.10 + seasonal_factor * 0.5 + heat_retention * 1.8

        return current_lst + delta

    def get_status(self) -> dict:
        """Return forecast model status."""
        return {
            "is_trained": self.is_trained,
            "model_path": FORECAST_MODEL_PATH,
            "model_exists": os.path.exists(FORECAST_MODEL_PATH),
            "metrics": self.metrics
        }


# ─── Module-level singleton ───
forecast_model = TemperatureForecastModel()
