import argparse
import configparser
import logging
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
from mlflow import keras
from sklearn.metrics import confusion_matrix

from src.analysis import analyse_results
from src.model import create_model
from src.preprocess import create_generators
from src.utils import get_number_of_files


def main(args):
    keras.autolog()
    train_generator, validation_generator = create_generators()

    # Here Hyperparameter

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        model = create_model(input_shape)

        logging.info(model.summary())
        logging.info("Model created ...")
        history = model.fit(
            train_generator,
            epochs=epochs,
            # steps_per_epoch=number_of_training_images // batch_size,
            validation_data=validation_generator,
            use_multiprocessing=False,
            workers=1,
        )
        # mlflow.log_figure(cm.figure_, 'test_confusion_matrix.png')
        logging.info("Predicting Validation data")
        predictions_proba = model.predict(validation_generator).flatten()
        predictions = np.where(predictions_proba < threshold, 0, 1).flatten()
        y_true = validation_generator.classes.flatten()

        result_df = pd.DataFrame(
            {
                "Prediction": predictions,
                "Probability": predictions_proba,
                "Label": y_true,
                "Path": validation_generator.filenames,
            }
        )

        cm = confusion_matrix(
            result_df["Label"], result_df["Prediction"], labels=[0, 1]
        )

        logging.info(cm)
        # Log Hyperparameter here
        # mlflow.log_param("alpha", alpha)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            keras.log_model(model, "model", registered_model_name="CNNClassification")
        else:
            keras.log_model(model, "model")

        # If analyze mode is activated - feature importance for randomly wrong classified frame
        if args.debug:
            analyse_results(model, result_df)

        logging.info("Program finished.")


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config.ini")

    logging.basicConfig(format="%(levelname)s: %(message)s", level=str(config["LOGGING"]["log_level"]))
    logging.info("Program started...")

    height = int(config["DEFAULT"]["image_height"])
    width = int(config["DEFAULT"]["image_width"])
    dimensions = int(config["DEFAULT"]["image_dimensions"])                  # 1 -> grayscale; 3 -> rgb
    threshold = float(config["DEFAULT"]["threshold"])

    input_shape = (
        height,
        width,
        dimensions
    )
    batch_size = int(config["Training"]["batch_size"])
    epochs = int(config["Training"]["epochs"])

    train_images_path = str(config["Training"]["train_images_path"])
    val_images_path = str(config["Training"]["val_images_path"])

    number_of_training_images = get_number_of_files(train_images_path)
    number_of_validation_images = get_number_of_files(val_images_path)

    classes = config["Training"]["classes"].split(",")
    number_of_classes = len(classes)

    parser = argparse.ArgumentParser(description="Additional Time Prediction")
    parser.add_argument(
        "--debug",
        type=bool,
        help="debug to specify, if wrongly classified images should be analyzed or not. Options (True | False)",
        choices=[True, False],
        default=False
    )
    args = parser.parse_args()

    main(args)
