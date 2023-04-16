import tensorflow as tf
from tensorflow.keras import models, layers


def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (6, 6), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        steps_per_execution=1,
    )
    return model
