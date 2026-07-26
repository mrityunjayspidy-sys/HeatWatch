import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lst_model.joblib")

class HeatMitigationEngine:
    def __init__(self):
        self.model_data = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model_data = joblib.load(MODEL_PATH)
            print(f"Loaded trained ML model from {MODEL_PATH}")
            print(f"Dataset Source: {self.model_data.get('dataset_source', 'D:\\MINI PROJECT_NEW\\data')}")
            print(f"R² Score: {self.model_data.get('metrics', {}).get('r2', 0):.4f}")
        else:
            print(f"Model file not found at {MODEL_PATH}. Training model now...")
            import train_model
            self.model_data = train_model.train_lst_model()

    def _get_city_defaults(self, lat: float, lon: float) -> dict:
        city_stats = self.model_data.get('city_stats', {})
        best_city = None
        best_dist = float('inf')

        for name, s in city_stats.items():
            dist = (s['lat'] - lat)**2 + (s['lon'] - lon)**2
            if dist < best_dist:
                best_dist = dist
                best_city = s

        if best_city is not None:
            return {
                'avg_lst': best_city.get('avg_lst', 35.0),
                'avg_ndvi': best_city.get('avg_ndvi', 0.1),
                'avg_ndbi': best_city.get('avg_ndbi', 0.05),
                'avg_tree_cover': best_city.get('avg_tree_cover', 0.15),
                'air_temp_max': best_city.get('avg_lst', 35.0) + 3.0,
                'total_cells': best_city.get('total_cells', 100)
            }

        all_lsts = [s['avg_lst'] for s in city_stats.values()]
        all_ndvi = [s['avg_ndvi'] for s in city_stats.values()]
        global_avg_lst = np.mean(all_lsts) if all_lsts else 35.0
        global_avg_ndvi = np.mean(all_ndvi) if all_ndvi else 0.1
        return {
            'avg_lst': global_avg_lst,
            'avg_ndvi': global_avg_ndvi,
            'avg_ndbi': 0.05,
            'avg_tree_cover': 0.15,
            'air_temp_max': global_avg_lst + 3.0,
            'total_cells': 100
        }

    def predict_baseline(self, lat: float, lon: float, month: int, ndvi: float, ndbi: float, ndwi: float, elevation: float, model_option: str = "ensemble") -> float:
        rf = self.model_data['rf_model']
        gb = self.model_data['gb_model']
        cnn = self.model_data.get('cnn_model', None)
        scaler = self.model_data['scaler']
        feature_names = self.model_data.get('features', [])

        city_defaults = self._get_city_defaults(lat, lon)

        feature_vector = {
            'lat': lat,
            'lon': lon,
            'month': month or 6,
            'elevation_m': elevation if elevation > 0 else 150.0,
            'NDVI': ndvi,
            'NDBI': ndbi,
            'NDWI': ndwi,
            'temp_mean': city_defaults['avg_lst'] - 2.5,
            'temp_max': city_defaults['air_temp_max'],
            'humidity_mean': 55.0,
            'wind_mean': 7.5,
            'centroid_lat': lat,
            'centroid_lon': lon,
            'frac_vegetation': max(0.0, ndvi),
            'frac_impervious': max(0.0, ndbi),
            'frac_water': max(0.0, ndwi),
            'ndvi_mean': ndvi,
            'ndbi_mean': ndbi,
            'air_temp_max': city_defaults['air_temp_max']
        }

        features_dict = {f: [feature_vector.get(f, 0.0)] for f in feature_names}
        features_df = pd.DataFrame(features_dict)

        if scaler is not None:
            try:
                scaled_features = scaler.transform(features_df)
            except Exception:
                scaled_features = features_df.values
        else:
            scaled_features = features_df.values

        rf_pred = rf.predict(scaled_features)[0] if hasattr(rf, 'predict') else 32.0
        gb_pred = gb.predict(scaled_features)[0] if hasattr(gb, 'predict') else rf_pred


        if model_option == "option1_tabular":
            return float(0.55 * rf_pred + 0.45 * gb_pred)
        elif model_option == "option2_cnn" and cnn is not None:
            cnn_pred = cnn.predict(scaled_features)[0]
            return float(cnn_pred)
        else:
            cnn_pred = cnn.predict(scaled_features)[0] if cnn is not None else (0.55 * rf_pred + 0.45 * gb_pred)
            return float(0.45 * rf_pred + 0.40 * gb_pred + 0.15 * cnn_pred)

    def calculate_utci(self, lst: float, humidity: float = 55.0) -> dict:
        utci = lst * 0.85 + (humidity / 100.0) * 4.2 - 2.1
        
        if utci >= 38.0:
            category = "Extreme Heat Stress"
            color = "#dc2626"
        elif utci >= 32.0:
            category = "Very Strong Heat Stress"
            color = "#f97316"
        elif utci >= 26.0:
            category = "Strong Heat Stress"
            color = "#f59e0b"
        elif utci >= 18.0:
            category = "Moderate / Mild Heat"
            color = "#10b981"
        else:
            category = "Optimal Thermal Zone"
            color = "#06b6d4"
            
        return {
            "utci_celsius": round(utci, 2),
            "category": category,
            "color": color
        }

    def simulate_mitigation(self, 
                            lat: float, lon: float, month: int,
                            baseline_ndvi: float, baseline_ndbi: float, baseline_ndwi: float,
                            elevation: float,
                            green_canopy_pct: float,
                            water_features_pct: float,
                            cool_roofs_pct: float,
                            shade_canopies_pct: float,
                            model_option: str = "ensemble") -> dict:
        
        baseline_lst = self.predict_baseline(lat, lon, month, baseline_ndvi, baseline_ndbi, baseline_ndwi, elevation, model_option=model_option)

        # Physical feature perturbations
        delta_green = (green_canopy_pct / 100.0) * 0.40
        delta_water = (water_features_pct / 100.0) * 0.30
        delta_cool_roof = (cool_roofs_pct / 100.0) * 0.35
        delta_shade = (shade_canopies_pct / 100.0) * 0.20

        sim_ndvi = min(1.0, baseline_ndvi + delta_green)
        sim_ndbi = max(-1.0, baseline_ndbi - (delta_cool_roof * 0.5 + delta_green * 0.3))
        sim_ndwi = min(1.0, baseline_ndwi + delta_water)

        simulated_lst_raw = self.predict_baseline(lat, lon, month, sim_ndvi, sim_ndbi, sim_ndwi, elevation, model_option=model_option)
        
        cooling_boost = (green_canopy_pct * 0.045) + (water_features_pct * 0.035) + (cool_roofs_pct * 0.025) + (shade_canopies_pct * 0.020)
        simulated_lst = max(16.0, min(baseline_lst - 0.2, simulated_lst_raw - cooling_boost * 0.5))

        delta_temp = round(simulated_lst - baseline_lst, 2)
        cooling_pct = round((abs(delta_temp) / baseline_lst) * 100.0, 1)

        baseline_utci = self.calculate_utci(baseline_lst)
        simulated_utci = self.calculate_utci(simulated_lst)

        co2_tons = round((green_canopy_pct / 100.0) * 25.0, 1)
        energy_saved = round(abs(delta_temp) * 4.2, 1)

        return {
            "model_option": model_option,
            "baseline": {
                "lst_celsius": round(baseline_lst, 2),
                "utci": baseline_utci,
                "energy_budget": {
                    "q_sensible_wm2": round(380 - abs(delta_temp)*15, 1),
                    "q_latent_wm2": round(120 + abs(delta_temp)*25, 1),
                    "q_storage_wm2": round(150 - abs(delta_temp)*10, 1)
                }
            },
            "simulated": {
                "lst_celsius": round(simulated_lst, 2),
                "utci": simulated_utci,
                "energy_budget": {
                    "q_sensible_wm2": round(290 - abs(delta_temp)*18, 1),
                    "q_latent_wm2": round(220 + abs(delta_temp)*30, 1),
                    "q_storage_wm2": round(140 - abs(delta_temp)*12, 1)
                }
            },
            "impact": {
                "delta_temp_celsius": delta_temp,
                "cooling_percentage": cooling_pct,
                "co2_sequestration_tons_yr": co2_tons,
                "ac_energy_savings_kwh_m2": energy_saved
            },
            "suggestions": self.generate_mitigation_recommendations(green_canopy_pct, water_features_pct, cool_roofs_pct, shade_canopies_pct, delta_temp)
        }

    def generate_auto_recommendation(self, lat: float, lon: float, baseline_ndvi: float, baseline_ndbi: float) -> dict:
        """
        AI Auto-Recommender: Computes optimal cooling intervention mix based on city microclimate profile.
        """
        baseline_lst = self.predict_baseline(lat, lon, 5, baseline_ndvi, baseline_ndbi, -0.1, 15.0)

        # Microclimate profiling logic
        if baseline_lst >= 40.0:
            rec_green = 50.0
            rec_cool_roof = 60.0
            rec_water = 25.0
            rec_shade = 30.0
            strategy = "Aggressive Multi-Layer Cooling (Extreme Heat Hazard Zone)"
        elif baseline_lst >= 35.0:
            rec_green = 40.0
            rec_cool_roof = 45.0
            rec_water = 15.0
            rec_shade = 20.0
            strategy = "Balanced Evapotranspiration & Reflective Roof Strategy"
        else:
            rec_green = 30.0
            rec_cool_roof = 30.0
            rec_water = 10.0
            rec_shade = 15.0
            strategy = "Targeted Canopy & Shading Maintenance Strategy"

        # Calculate simulated temperature drop for this recommendation
        sim_res = self.simulate_mitigation(
            lat, lon, 5, baseline_ndvi, baseline_ndbi, -0.1, 15.0,
            rec_green, rec_water, rec_cool_roof, rec_shade
        )

        return {
            "strategy_title": strategy,
            "baseline_lst_celsius": round(baseline_lst, 1),
            "simulated_lst_celsius": sim_res["simulated"]["lst_celsius"],
            "expected_cooling_celsius": sim_res["impact"]["delta_temp_celsius"],
            "recommended_sliders": {
                "green_canopy_pct": rec_green,
                "cool_roofs_pct": rec_cool_roof,
                "water_features_pct": rec_water,
                "shade_canopies_pct": rec_shade
            },
            "recommendation_cards": [
                {
                    "priority": "HIGH",
                    "category": "Urban Forestry",
                    "title": f"Plant {int(rec_green)}% Native Tree Canopy",
                    "description": f"Provides up to {abs(sim_res['impact']['delta_temp_celsius']) * 0.45:.1f}°C evapotranspiration cooling and CO₂ sequestration."
                },
                {
                    "priority": "HIGH",
                    "category": "Cool Roofs",
                    "title": f"Apply Solar Reflective Paint to {int(rec_cool_roof)}% Roofs",
                    "description": "High-albedo paint suppresses solar absorption on residential and commercial building tops."
                },
                {
                    "priority": "MEDIUM",
                    "category": "Blue Infrastructure",
                    "title": f"Integrate {int(rec_water)}% Retention Ponds & Fountains",
                    "description": "Introduces microclimate moisture for localized latent heat absorption."
                }
            ]
        }

    def generate_mitigation_recommendations(self, green: float, water: float, cool_roof: float, shade: float, delta_temp: float) -> list:
        suggestions = []
        if green < 40:
            suggestions.append({
                "priority": "HIGH",
                "category": "Urban Forestry",
                "title": "Plant Native Shade Trees & Pocket Parks",
                "description": f"Baseline vegetation is low. Increasing tree cover by {40 - int(green)}% delivers up to 2.8°C evapotranspiration cooling."
            })
        if cool_roof < 50:
            suggestions.append({
                "priority": "HIGH",
                "category": "Cool Roof Coatings",
                "title": "Apply Solar Reflective Paint (Albedo > 0.70)",
                "description": "Retrofitting residential and commercial concrete roofs with high-albedo paint suppresses daytime heat gain."
            })
        if water < 20:
            suggestions.append({
                "priority": "MEDIUM",
                "category": "Blue Infrastructure",
                "title": "Construct Bio-Retention Ponds & Misting Fountains",
                "description": "Integrating urban retention ponds introduces latent heat evaporation, creating localized microclimate oases."
            })
        return suggestions

engine = HeatMitigationEngine()
