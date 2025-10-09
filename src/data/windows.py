import numpy as np
import pandas as pd
from ..config.runtime import GRID_SIZE, TIME_COL, MAG_COL
from .grids import index_to_grid

def magnitude_grid_future(df, i, lat_bins, lon_bins, horizon_hours=24):
    t0 = pd.to_datetime(df.iloc[i][TIME_COL])
    t1 = t0 + pd.to_timedelta(horizon_hours, unit="h")
    fut = df.loc[(df[TIME_COL] > t0) & (df[TIME_COL] <= t1), ["latitude","longitude",MAG_COL]]
    Y = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if not fut.empty:
        la = fut["latitude"].to_numpy()
        lo = fut["longitude"].to_numpy()
        mag = fut[MAG_COL].to_numpy(dtype=np.float32)
        lat_idx, lon_idx = index_to_grid(la, lo, lat_bins, lon_bins)
        np.maximum.at(Y, (lat_idx, lon_idx), mag)
    return Y[..., None]
