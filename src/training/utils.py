# models/cnn/utils.py
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    """Create directory p (Path or str) if it doesn't exist."""
    Path(p).mkdir(parents=True, exist_ok=True)

def save_artifacts(metrics: dict, y_test, y_pred, outdir):
    """Save metrics and arrays to outdir."""
    ensure_dir(outdir)
    out = Path(outdir)
    with open(out / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    np.save(out / "y_test.npy", y_test)
    np.save(out / "y_pred.npy", y_pred)

def plot_learning_curves(history):
    """Plot train/val MAE curves from a Keras History."""
    plt.figure(figsize=(6,4))
    if "mae" in history.history:
        plt.plot(history.history["mae"], label="train MAE")
    if "val_mae" in history.history:
        plt.plot(history.history["val_mae"], label="val MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE")
    plt.title("CNN Magnitude — Learning Curves")
    plt.grid(True); plt.legend(); plt.show()
