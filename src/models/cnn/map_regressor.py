import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

def build_cnn_map(input_shape=(50,50,1), lr=1e-3, wd=1e-4, dropout=0.10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(wd))(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.SpatialDropout2D(dropout)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # 1x1 conv to regress per-cell magnitude (same HxW)
    outputs = layers.Conv2D(1, 1, padding="same", activation=None)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model
