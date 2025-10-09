import numpy as np
import pandas as pd
from ..config.runtime import GRID_SIZE, LAT_COL, LON_COL

def make_bins(series: pd.Series, pad=0.5, bins=GRID_SIZE):
    smin, smax = series.min() - pad, series.max() + pad
    return np.linspace(smin, smax, bins + 1)

def index_to_grid(lat, lon, lat_bins, lon_bins):
    lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, GRID_SIZE - 1)
    lon_idx = np.clip(np.digitize(lon, lon_bins) - 1, 0, GRID_SIZE - 1)
    return lat_idx, lon_idx

def density_grid(events_window: pd.DataFrame, lat_bins, lon_bins, time_decay=0.0):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    la = events_window[LAT_COL].to_numpy()
    lo = events_window[LON_COL].to_numpy()
    lat_idx, lon_idx = index_to_grid(la, lo, lat_bins, lon_bins)
    weights = np.ones_like(lat_idx, dtype=np.float32)
    if time_decay > 0 and "age" in events_window.columns:
        weights = np.exp(-time_decay * events_window["age"].to_numpy(dtype=np.float32))
    np.add.at(grid, (lat_idx, lon_idx), weights)
    return grid[..., np.newaxis]
