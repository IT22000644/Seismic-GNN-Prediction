import numpy as np
import pandas as pd
from .config import GRID_SIZE, LAT_COL, LON_COL

def make_bins(series: pd.Series, pad=0.5, bins=GRID_SIZE):
    smin, smax = series.min() - pad, series.max() + pad
    return np.linspace(smin, smax, bins + 1)

def index_to_grid(lat, lon, lat_bins, lon_bins):
    lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, GRID_SIZE - 1)
    lon_idx = np.clip(np.digitize(lon, lon_bins) - 1, 0, GRID_SIZE - 1)
    return lat_idx, lon_idx

def density_grid(events_window: pd.DataFrame, lat_bins, lon_bins):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    la = events_window[LAT_COL].to_numpy()
    lo = events_window[LON_COL].to_numpy()
    lat_idx, lon_idx = index_to_grid(la, lo, lat_bins, lon_bins)
    np.add.at(grid, (lat_idx, lon_idx), 1.0)
    return grid[..., np.newaxis]  # (H, W, 1)
