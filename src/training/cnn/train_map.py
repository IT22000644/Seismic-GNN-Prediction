import json, datetime as dt
import tensorflow as tf
from pathlib import Path
from ...config.runtime import ARTIFACTS, SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, DROPOUT, USE_MIXED_PRECISION
from ...data.loading import load_data_map
from ...models.cnn.map_regressor import build_cnn_map
from ..utils import ensure_dir, plot_learning_curves

def main():
    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.random.set_seed(SEED)

    Xtr, ytr, Xv, yv, Xte, yte = load_data_map()
    outdir = Path(ARTIFACTS) / f"map_run_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ensure_dir(outdir)

    model = build_cnn_map(input_shape=Xtr.shape[1:], lr=LR, wd=WEIGHT_DECAY, dropout=DROPOUT)
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", patience=6, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(outdir / "best.keras"), monitor="val_mae", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(outdir / "history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(outdir / "tb")),
    ]

    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
                .shuffle(4096, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val_ds   = tf.data.Dataset.from_tensor_slices((Xv, yv)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((Xte, yte)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=1)
    with open(outdir / "history.json", "w") as f: json.dump(hist.history, f, indent=2)
    plot_learning_curves(hist)  

    model.save(outdir / "final.keras")
    metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    print("Test metrics:", metrics)
    with open(outdir / "test_metrics.json", "w") as f: json.dump({k: float(v) for k,v in metrics.items()}, f, indent=2)

if __name__ == "__main__":
    main()
