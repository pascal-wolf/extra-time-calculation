import configparser
import itertools
import logging

import mlflow
import tensorflow as tf
from keras import backend as K
from mlflow.keras import autolog

from src.model import create_cnn_model
from src.preprocess import create_generators
from tqdm import tqdm


def perform_hyperparameter_training():
    """
    Perform hyperparameter training on a CNN model.

    This function generates all permutations of the given hyperparameters, updates the configuration file with each permutation, and trains the model with these parameters. The model's performance is logged with MLFlow.

    The hyperparameters include kernel size, number of CNN layers, number of channels, number of hidden layers, and input size.

    The function uses early stopping to halt training when the model's loss stops improving.

    If an error occurs during training, the error is printed and the current MLFlow run is ended.
    """
    parameters = {
        "kernel_size": [3, 5],
        "cnn_layer": [1, 2, 3],
        "channels": [1, 3],
        "hidden_layer": [0, 1],
        "input_size": [128, 256],
    }

    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    config = configparser.ConfigParser()
    config.read("./configs/config.ini")
    for parameter in tqdm(permutations_dicts):
        config.set("DEFAULT", "image_width", str(parameter["input_size"]))
        config.set("DEFAULT", "image_height", str(parameter["input_size"]))
        config.set("DEFAULT", "image_dimensions", str(parameter["channels"]))

        config.set("MODEL", "hiden_layer", str(parameter["hidden_layer"]))
        config.set("MODEL", "kernel_size", str(parameter["kernel_size"]))
        config.set("MODEL", "cnn_layer", str(parameter["cnn_layer"]))
        train_generator, validation_generator = create_generators(config)
        mlflow.log_params(parameter)
        input_shape = (
            parameter["input_size"],
            parameter["input_size"],
            parameter["channels"],
        )
        kernel_size = (parameter["kernel_size"], parameter["kernel_size"])

        autolog(log_models=True)
        model = create_cnn_model(
            input_shape,
            kernel_size,
            parameter["cnn_layer"],
            parameter["hidden_layer"],
        )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True
        )
        try:
            history = model.fit(
                train_generator,
                epochs=20,
                validation_data=validation_generator,
                use_multiprocessing=False,
                callbacks=[early_stopping_callback],
                workers=1,
                verbose=0,
            )
        except Exception as e:
            print("Error on run", parameter)
        mlflow.end_run()


def main():
    perform_hyperparameter_training()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./configs/config.ini")

    logging.basicConfig(
        format="%(levelname)s: %(message)s", level=str(config["LOGGING"]["log_level"])
    )
    logging.info("Program started...")

    main()
