import numpy as np, matplotlib.pyplot as plt, json
import tensorflow as tf
from pathlib import Path
from ...training.utils import ensure_dir

def eval_and_save(model_path, X_test, y_test, outdir):
    ensure_dir(outdir)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss="mae", metrics=["mae","mse"])
    ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

    metrics = model.evaluate(ds, return_dict=True, verbose=0)
    with open(Path(outdir)/"test_metrics.json","w") as f: json.dump({k: float(v) for k,v in metrics.items()}, f, indent=2)

    y_pred = model.predict(ds)
    np.save(Path(outdir)/"y_test.npy", y_test); np.save(Path(outdir)/"y_pred.npy", y_pred)

    yt, yp = y_test.ravel(), y_pred.ravel()
    # Parity
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    plt.figure(figsize=(5,5)); plt.scatter(yt, yp, s=2, alpha=0.3); plt.plot([lo,hi],[lo,hi],'--')
    plt.xlabel("True"); plt.ylabel("Pred"); plt.title("Parity (all cells)"); plt.grid(True)
    plt.savefig(Path(outdir)/"parity.png", dpi=140); plt.close()

    # Residuals
    resid = yp - yt
    plt.figure(figsize=(6,4)); plt.hist(resid, bins=60, edgecolor="k")
    plt.title("Residuals (ŷ − y)"); plt.xlabel("Error"); plt.ylabel("Count"); plt.grid(True)
    plt.savefig(Path(outdir)/"residuals.png", dpi=140); plt.close()

    # Qualitative maps
    for i in range(min(3, len(y_test))):
        fig, axs = plt.subplots(1,3, figsize=(9,3))
        for a in axs: a.axis("off")
        axs[0].imshow(y_test[i,...,0]); axs[0].set_title("True")
        axs[1].imshow(y_pred[i,...,0]); axs[1].set_title("Pred")
        axs[2].imshow(y_pred[i,...,0] - y_test[i,...,0]); axs[2].set_title("Error")
        fig.tight_layout(); fig.savefig(Path(outdir)/f"sample_{i}.png", dpi=140); plt.close(fig)
