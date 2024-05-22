import argparse
import configparser
import logging

import mlflow
import tensorflow as tf
from mlflow.keras import autolog

from src.model import (
    create_cnn_model,
    create_efficientnetvtwo_model,
    create_mobilenet_model,
)
from src.preprocess import create_generators


def main(args, config):
    height = int(config["DEFAULT"]["image_height"])
    width = int(config["DEFAULT"]["image_width"])
    dimensions = int(config["DEFAULT"]["image_dimensions"])

    input_shape = (height, width, dimensions)
    epochs = int(config["TRAINING"]["epochs"])

    train_generator, validation_generator = create_generators(config)
    autolog(log_models=True)
    kernel_size, n_cnn_layer, n_hidden_layer = 5, 2, 0
    model = create_cnn_model(
        input_shape,
        kernel_size,
        n_cnn_layer,
        n_hidden_layer,
    )
    logging.info("Model created ...")
    logging.info(model.summary())
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./tmp/checkpoint/",
        save_weights_only=True,
        monitor="val_precision",
        mode="max",
        save_best_only=True,
    )
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        use_multiprocessing=False,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        workers=1,
    )
    mlflow.end_run()
    logging.info("Program finished.")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./configs/config.ini")

    logging.basicConfig(
        format="%(levelname)s: %(message)s", level=str(config["LOGGING"]["log_level"])
    )
    logging.info("Program started...")

    parser = argparse.ArgumentParser(description="Additional Time Prediction")
    parser.add_argument(
        "--debug",
        type=bool,
        help="debug to specify, if wrongly classified images should be analyzed or not. Options (True | False)",
        choices=[True, False],
        default=False,
    )
    args = parser.parse_args()

    main(args, config)
