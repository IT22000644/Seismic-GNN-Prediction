import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from .config import ARTIFACTS
from .data import load_data

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    model_path = Path(ARTIFACTS) / "cnn_best.keras"
    if not model_path.exists():
        model_path = Path(ARTIFACTS) / "cnn_final.keras"
    model = tf.keras.models.load_model(model_path, compile=False)
    # recompile for metrics if needed
    model.compile(loss="mae", metrics=["mae","mse"])

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
    metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    print("Test metrics:", metrics)

    y_pred = model.predict(test_ds).ravel()

    # Baseline
    baseline_mae = np.mean(np.abs(y_test - np.mean(y_train)))
    print(f"Baseline (train-mean) MAE: {baseline_mae:.4f}")

    # Plots
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, s=8, alpha=0.6)
    lo = min(y_test.min(), y_pred.min()); hi = max(y_test.max(), y_pred.max())
    plt.plot([lo,hi],[lo,hi],'--')
    plt.xlabel("True magnitude"); plt.ylabel("Predicted")
    plt.title("Pred vs True (Test)"); plt.grid(True); plt.show()

    plt.figure(figsize=(6,4))
    resid = y_pred - y_test
    plt.hist(resid, bins=40, edgecolor="k")
    plt.title("Residuals (ŷ − y)"); plt.xlabel("Error"); plt.ylabel("Count"); plt.grid(True); plt.show()

if __name__ == "__main__":
    main()
