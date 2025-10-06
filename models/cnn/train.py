import tensorflow as tf
from pathlib import Path
from .config import ARTIFACTS, SEED
from .data import load_data
from .model import build_cnn
from .utils import ensure_dir, plot_learning_curves

def main():
    tf.random.set_seed(SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    H, W, C = X_train.shape[1:]
    model = build_cnn((H, W, C))
    model.summary()

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4096, seed=SEED).batch(64).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64).prefetch(tf.data.AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", patience=6, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(Path(ARTIFACTS) / "cnn_best.keras"),
                                           monitor="val_mae", save_best_only=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=cbs, verbose=1)
    plot_learning_curves(history)

    # Final save
    ensure_dir(ARTIFACTS)
    model.save(Path(ARTIFACTS) / "cnn_final.keras")

    # Quick test eval here for convenience
    metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    print("Test metrics:", metrics)

if __name__ == "__main__":
    main()
