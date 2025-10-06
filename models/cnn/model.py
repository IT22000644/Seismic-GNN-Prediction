import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)  # regression to magnitude
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="mae",
        metrics=["mae","mse", r2_metric]
    )
    return model

@tf.function
def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + 1e-8)
