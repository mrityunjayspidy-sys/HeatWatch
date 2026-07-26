"""
Feature Engineering Module for Urban Heat Island (UHI) & LST Prediction.
Derives spatial and spectral indices per grid-cell:
- NDVI (Normalized Difference Vegetation Index)
- NDBI (Normalized Difference Built-up Index)
- NDWI (Normalized Difference Water Index)
- Albedo (Surface Reflectivity)
- Impervious Surface Fraction
- Sky View Factor (SVF)
- Land Surface Emissivity (Planck's Law LST Conversion)
- Spatial Distance to Nearest Green / Blue Space
"""

import numpy as np
import pandas as pd


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)"""
    denom = np.where((nir + red) == 0, 1e-6, nir + red)
    return (nir - red) / denom


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDBI = (SWIR - NIR) / (SWIR + NIR)"""
    denom = np.where((swir + nir) == 0, 1e-6, swir + nir)
    return (swir - nir) / denom


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR)"""
    denom = np.where((green + nir) == 0, 1e-6, green + nir)
    return (green - nir) / denom


def compute_albedo(blue: np.ndarray, green: np.ndarray, red: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Broadband Surface Albedo approximation from Sentinel-2 / Landsat optical bands
    Formula derived from Liang (2001) for surface reflectance.
    """
    albedo = 0.356 * blue + 0.130 * green + 0.373 * red + 0.085 * nir + 0.072 * swir1 + 0.018 * swir2 - 0.0018
    return np.clip(albedo, 0.0, 1.0)


def compute_emissivity(ndvi: np.ndarray) -> np.ndarray:
    """
    Land Surface Emissivity (ε) calculation based on NDVI Threshold Method (Sobrino et al., 2004)
    Used to convert thermal brightness temperature -> Land Surface Temperature (LST) via Planck's Law.
    """
    emissivity = np.zeros_like(ndvi, dtype=float)
    
    # Soil pixels (NDVI < 0.2)
    soil_mask = ndvi < 0.2
    emissivity[soil_mask] = 0.97 - 0.005 * ndvi[soil_mask]

    # Fully vegetated pixels (NDVI > 0.5)
    veg_mask = ndvi > 0.5
    emissivity[veg_mask] = 0.99

    # Mixed vegetation/built-up pixels (0.2 <= NDVI <= 0.5)
    mixed_mask = (ndvi >= 0.2) & (ndvi <= 0.5)
    pv = ((ndvi[mixed_mask] - 0.2) / (0.5 - 0.2)) ** 2  # Fractional vegetation
    d_eps = (1 - 0.97) * (1 - pv) * 0.55 * 0.99          # Cavity effect
    emissivity[mixed_mask] = 0.97 * (1 - pv) + 0.99 * pv + d_eps

    return np.clip(emissivity, 0.90, 0.995)


def compute_brightness_temp_to_lst(bt_kelvin: np.ndarray, emissivity: np.ndarray, wavelength_um: float = 10.8) -> np.ndarray:
    """
    Converts Brightness Temperature (K) to Land Surface Temperature (°C) using Planck's Law split-window / single-channel formula:
    LST = BT / (1 + (λ * BT / ρ) * ln(ε)) - 273.15
    where ρ = h * c / σ = 1.4388e-2 m K
    """
    rho = 14388.0  # μm K
    lst_kelvin = bt_kelvin / (1.0 + (wavelength_um * bt_kelvin / rho) * np.log(np.clip(emissivity, 1e-4, 1.0)))
    return lst_kelvin - 273.15


def derive_grid_cell_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs comprehensive feature engineering on raw tabular/grid satellite data.
    Ensures all key engineered features exist and are cleaned.
    """
    df = df.copy()

    # Spectral indices if raw bands exist, otherwise fallback/clean
    if 'ndvi_mean' not in df.columns and 'nir' in df.columns and 'red' in df.columns:
        df['ndvi_mean'] = compute_ndvi(df['nir'].values, df['red'].values)
    
    if 'ndbi_mean' not in df.columns and 'swir' in df.columns and 'nir' in df.columns:
        df['ndbi_mean'] = compute_ndbi(df['swir'].values, df['nir'].values)

    if 'ndwi_mean' not in df.columns and 'green' in df.columns and 'nir' in df.columns:
        df['ndwi_mean'] = compute_ndwi(df['green'].values, df['nir'].values)

    # Derived fractions if not explicitly present
    if 'frac_vegetation' not in df.columns:
        df['frac_vegetation'] = np.clip(df.get('ndvi_mean', 0.1), 0.0, 1.0)

    if 'frac_impervious' not in df.columns:
        df['frac_impervious'] = np.clip(df.get('ndbi_mean', 0.1), 0.0, 1.0)

    if 'frac_water' not in df.columns:
        df['frac_water'] = np.clip(df.get('ndwi_mean', 0.0), 0.0, 1.0)

    if 'frac_bare_soil' not in df.columns:
        df['frac_bare_soil'] = np.clip(1.0 - df['frac_vegetation'] - df['frac_impervious'] - df['frac_water'], 0.0, 1.0)

    if 'sky_view_factor' not in df.columns:
        # Sky View Factor default based on building density
        df['sky_view_factor'] = np.clip(1.0 - 0.4 * df.get('building_density', df['frac_impervious']), 0.2, 1.0)

    if 'emissivity' not in df.columns:
        df['emissivity'] = compute_emissivity(df.get('ndvi_mean', pd.Series(0.2, index=df.index)).values)

    if 'cooling_potential' not in df.columns:
        df['cooling_potential'] = df.get('frac_vegetation', 0.1) * 2.5 + df.get('frac_water', 0.0) * 3.0

    return df
