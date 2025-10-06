
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED    = DATA_DIR / "processed" / "model_ready"
INTERIM      = DATA_DIR / "interim"
ARTIFACTS    = PROJECT_ROOT / "artifacts" / "cnn"

GRID_SIZE = 50
LOOKBACK  = 50
DENSITY_ONLY = True

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SEED = 42

Xy_FILES = dict(
    X_train = PROCESSED / "X_train_cnn.npy",
    y_train = PROCESSED / "y_train_cnn.npy",
    X_val   = PROCESSED / "X_val_cnn.npy",
    y_val   = PROCESSED / "y_val_cnn.npy",
    X_test  = PROCESSED / "X_test_cnn.npy",
    y_test  = PROCESSED / "y_test_cnn.npy",
)


FALLBACK_DATA = INTERIM / "quakes_clean.csv"

TIME_COL = "time"
LAT_COL  = "latitude"
LON_COL  = "longitude"
MAG_COL  = "magnitude"

LOG1P_INPUT = True
STANDARDIZE_Y = False
