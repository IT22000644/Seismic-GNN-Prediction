from pathlib import Path

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED    = DATA_DIR / "processed" / "model_ready" / "cnn_map"
INTERIM      = DATA_DIR / "interim"
ARTIFACTS    = PROJECT_ROOT / "artifacts" / "cnn" / "map"

# Data schema
TIME_COL, LAT_COL, LON_COL, MAG_COL = "time", "latitude", "longitude", "magnitude"

# Grid + windowing
GRID_SIZE = 50
LOOKBACK  = 50
HORIZON_HOURS = 72
LOG1P_INPUT = True
TIME_DECAY = 0.0

# Splits + seed
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SEED = 42

# Model/training
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10
USE_MIXED_PRECISION = True

# Cache file names
Xy_FILES = dict(
    X_train = PROCESSED / "X_train_cnn_map.npy",
    y_train = PROCESSED / "y_train_cnn_map.npy",
    X_val   = PROCESSED / "X_val_cnn_map.npy",
    y_val   = PROCESSED / "y_val_cnn_map.npy",
    X_test  = PROCESSED / "X_test_cnn_map.npy",
    y_test  = PROCESSED / "y_test_cnn_map.npy",
)

FALLBACK_DATA = INTERIM / "quakes_clean.csv"
