from tensorflow.keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from tensorflow.keras import optimizers
from tensorflow.keras.applications import EfficientNetB1
import tensorflow as tf
from keras import backend as K


def create_model(input_shape):

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            input_shape[0], (3, 3), activation="relu", input_shape=input_shape
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model
