import json, os
import numpy as np
import pandas as pd
from pathlib import Path
from .config import (
    Xy_FILES, FALLBACK_DATA, TIME_COL, MAG_COL, GRID_SIZE, LOOKBACK,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC, LOG1P_INPUT, STANDARDIZE_Y, ARTIFACTS
)
from .grids import make_bins, density_grid

def _load_cached():
    files = {k: Path(v) for k, v in Xy_FILES.items()}
    if all(p.exists() for p in files.values()):
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        X_val   = np.load(files["X_val"])
        y_val   = np.load(files["y_val"])
        X_test  = np.load(files["X_test"])
        y_test  = np.load(files["y_test"])
        return X_train, y_train, X_val, y_val, X_test, y_test
    return None

def _read_fallback_table():
    path = Path(FALLBACK_DATA)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Either export CNN arrays via your notebook "
            f"or place a cleaned events file (.parquet or .csv) at this path."
        )
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df

def _chronological_split(N):
    n_train = int(N * TRAIN_FRAC)
    n_val   = int(N * VAL_FRAC)
    n_test  = N - n_train - n_val
    idxs = dict(
        train=(0, n_train),
        val  =(n_train, n_train + n_val),
        test =(n_train + n_val, n_train + n_val + n_test),
    )
    return idxs

def _build_from_events(df: pd.DataFrame):
    # sort by time and keep needed cols
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Bins for the whole dataset
    lat_bins = make_bins(df["latitude"])
    lon_bins = make_bins(df["longitude"])

    X, y = [], []
    for i in range(LOOKBACK, len(df)):
        window = df.iloc[i-LOOKBACK:i]
        X.append(density_grid(window, lat_bins, lon_bins))
        y.append(df.iloc[i][MAG_COL])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Optional input transform (log1p density)
    if LOG1P_INPUT:
        X = np.log1p(X)

    # Optional target scaling (usually keep magnitudes raw)
    if STANDARDIZE_Y:
        y_mean, y_std = y.mean(), (y.std() + 1e-8)
        y = (y - y_mean) / y_std
        (Path(ARTIFACTS)).mkdir(parents=True, exist_ok=True)
        with open(Path(ARTIFACTS) / "y_scaler.json", "w") as f:
            json.dump({"mean": float(y_mean), "std": float(y_std)}, f)

    # Chrono split
    idxs = _chronological_split(len(X))
    (s0, e0), (s1, e1), (s2, e2) = idxs["train"], idxs["val"], idxs["test"]

    return X[s0:e0], y[s0:e0], X[s1:e1], y[s1:e1], X[s2:e2], y[s2:e2]

def load_data(force_rebuild: bool = False):
    if not force_rebuild:
        cached = _load_cached()
        if cached is not None:
            return cached
    df = _read_fallback_table()
    return _build_from_events(df)
