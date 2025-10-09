# src/data/loading.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ..config.runtime import (
    Xy_FILES, FALLBACK_DATA, TIME_COL, MAG_COL, LOOKBACK, GRID_SIZE,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC, LOG1P_INPUT, TIME_DECAY, HORIZON_HOURS
)
from .grids import make_bins, density_grid
from .windows import magnitude_grid_future

def _load_cached():
    files = {k: Path(v) for k, v in Xy_FILES.items()}
    if all(p.exists() for p in files.values()):
        return tuple(np.load(files[k]) for k in ("X_train","y_train","X_val","y_val","X_test","y_test"))
    return None

def _read_events_table():
    p = Path(FALLBACK_DATA)
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Export preprocessed events there (CSV or Parquet).")
    df = pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, MAG_COL]).reset_index(drop=True)
    return df

def _chronosplit(N):
    n_train = int(N * TRAIN_FRAC); n_val = int(N * VAL_FRAC); n_test = N - n_train - n_val
    return (0,n_train), (n_train, n_train+n_val), (n_train+n_val, N)

def load_data_map(force_rebuild=False):
    if not force_rebuild:
        cached = _load_cached()
        if cached is not None: return cached

    df = _read_events_table().sort_values(TIME_COL).reset_index(drop=True)

    # Build bins
    lat_bins = make_bins(df["latitude"], bins=GRID_SIZE)
    lon_bins = make_bins(df["longitude"], bins=GRID_SIZE)

    # Precompute age in the LOOKBACK window 
    df["idx"] = np.arange(len(df))
    X, Y = [], []
    for i in range(LOOKBACK, len(df)-1):
        window = df.iloc[i-LOOKBACK:i].copy()
        if TIME_DECAY > 0:
            window["age"] = np.arange(LOOKBACK-1, -1, -1, dtype=np.float32)

        X.append(density_grid(window, lat_bins, lon_bins, time_decay=TIME_DECAY))
        Y.append(magnitude_grid_future(df, i, lat_bins, lon_bins, horizon_hours=HORIZON_HOURS))

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if LOG1P_INPUT: X = np.log1p(X)

    (s0,e0), (s1,e1), (s2,e2) = _chronosplit(len(X))
    Xt, yt = X[s0:e0], Y[s0:e0]
    Xv, yv = X[s1:e1], Y[s1:e1]
    Xs, ys = X[s2:e2], Y[s2:e2]

    # Cache to disk
    Path(Xy_FILES["X_train"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(Xy_FILES["X_train"], Xt); np.save(Xy_FILES["y_train"], yt)
    np.save(Xy_FILES["X_val"],   Xv); np.save(Xy_FILES["y_val"],   yv)
    np.save(Xy_FILES["X_test"],  Xs); np.save(Xy_FILES["y_test"],  ys)
    return Xt, yt, Xv, yv, Xs, ys
