"""
Google Earth Engine Live Data Fetcher
Queries real GEE satellite collections: NASA/ASTER_GED, MODIS, Sentinel-2.
Falls back to D:\\MINI PROJECT_NEW\\data\\processed parquet files (no mock data).
"""
import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

import ee

GEE_COLLECTIONS = {
    "aster_ged": "NASA/ASTER_GED/AG100_003",
    "modis_daily": "MODIS/061/MOD11A1",
    "sentinel2_sr": "COPERNICUS/S2_SR_HARMONIZED",
    "srtm_elevation": "USGS/SRTM90_V4"
}

DATA_DIR = r"D:\MINI PROJECT_NEW\data\processed"


class GEELiveDataFetcher:
    def __init__(self):
        self.is_authenticated = False
        self.init_gee()

    def init_gee(self):
        try:
            ee.Initialize()
            self.is_authenticated = True
            print("Google Earth Engine API initialized with active credentials.")
        except Exception as err:
            try:
                ee.Initialize(opt_url='https://earthengine.googleapis.com')
                self.is_authenticated = True
            except Exception:
                self.is_authenticated = False
                print(f"GEE not authenticated: {err}. Using real dataset from {DATA_DIR}.")

    def fetch_city_gee_observation(self, city_info: dict) -> dict:
        lat = city_info['lat']
        lon = city_info['lon']

        if self.is_authenticated:
            try:
                point = ee.Geometry.Point([lon, lat])

                # NASA ASTER GED LST
                aster_img = ee.Image(GEE_COLLECTIONS["aster_ged"])
                aster_stats = aster_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=point, scale=100
                ).getInfo()
                lst_kelvin = aster_stats.get('mean_temp', 305.0)
                lst_celsius = (lst_kelvin * 0.01) - 273.15

                # Sentinel-2 Surface Reflectance
                s2_col = ee.ImageCollection(GEE_COLLECTIONS["sentinel2_sr"]) \
                    .filterBounds(point) \
                    .filterDate('2024-01-01', datetime.date.today().strftime('%Y-%m-%d')) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                s2_img = s2_col.median()

                ndvi_val = s2_img.normalizedDifference(['B8', 'B4']).reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=point, scale=20
                ).getInfo().get('nd', None)
                ndbi_val = s2_img.normalizedDifference(['B11', 'B8']).reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=point, scale=20
                ).getInfo().get('nd', None)
                ndwi_val = s2_img.normalizedDifference(['B3', 'B8']).reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=point, scale=20
                ).getInfo().get('nd', None)

                # SRTM Elevation
                srtm = ee.Image(GEE_COLLECTIONS["srtm_elevation"])
                elev_val = srtm.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=point, scale=90
                ).getInfo().get('elevation', None)

                return {
                    "City": city_info['name'],
                    "State": city_info.get('state', ''),
                    "Latitude": lat,
                    "Longitude": lon,
                    "Month_Num": datetime.date.today().month,
                    "NDVI": round(float(ndvi_val), 4) if ndvi_val is not None else None,
                    "NDBI": round(float(ndbi_val), 4) if ndbi_val is not None else None,
                    "NDWI": round(float(ndwi_val), 4) if ndwi_val is not None else None,
                    "Elevation": round(float(elev_val), 1) if elev_val is not None else None,
                    "LST": round(float(lst_celsius), 2),
                    "Data_Source": "LIVE_GEE_SERVER_QUERY"
                }
            except Exception as e:
                print(f"GEE fetch error for {city_info['name']}: {e}")

        # Fallback: read from real parquet dataset
        return self._fetch_from_parquet(city_info)

    def _fetch_from_parquet(self, city_info: dict) -> dict:
        """
        Read real satellite data from D:\\MINI PROJECT_NEW\\data\\processed.
        No synthetic / mock values.
        """
        name_lower = city_info['name'].lower().replace(' ', '_')
        parquet_path = os.path.join(DATA_DIR, name_lower, "features.parquet")

        if not os.path.exists(parquet_path):
            matches = glob.glob(os.path.join(DATA_DIR, f"*{name_lower}*", "features.parquet"))
            if matches:
                parquet_path = matches[0]
            else:
                raise RuntimeError(f"No dataset found for {city_info['name']} in {DATA_DIR}")

        df = pd.read_parquet(parquet_path)

        return {
            "City": city_info['name'],
            "State": city_info.get('state', ''),
            "Latitude": float(df['centroid_lat'].mean()),
            "Longitude": float(df['centroid_lon'].mean()),
            "Month_Num": datetime.date.today().month,
            "NDVI": round(float(df['ndvi_mean'].mean()), 4),
            "NDBI": round(float(df['ndbi_mean'].mean()), 4),
            "NDWI": None,
            "Elevation": None,
            "LST": round(float(df['lst_mean'].mean()), 2),
            "Data_Source": f"REAL_DATASET: {parquet_path}"
        }


fetcher = GEELiveDataFetcher()
