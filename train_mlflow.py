import argparse
import logging
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow import keras
from sklearn.metrics import confusion_matrix

from src.analysis import analyse_results
from src.model import create_model
from src.preprocess import create_generators
from src.settings import Settings


def main(args):

    keras.autolog()
    train_generator, validation_generator = create_generators()

    # Here Hyperparameter

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        model = create_model()
        logging.info("Model created ...")

        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
        history = model.fit(
            train_generator,
            epochs=Settings.epochs,
            validation_data=validation_generator,
            use_multiprocessing=False,
            callbacks=[callback],
            workers=1,
        )
        # mlflow.log_figure(cm.figure_, 'test_confusion_matrix.png')
        logging.info("Predicting Validation data")
        predictions_proba = model.predict(validation_generator).flatten()
        predictions = np.where(predictions_proba < Settings.threshold, 0, 1).flatten()
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

    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=Settings.log_level)

    logging.info("Program started...")

    parser = argparse.ArgumentParser(description="Additional Time Prediction")
    parser.add_argument(
        "--debug",
        type=bool,
        help="debug to specify, if wrongly classified images should be analyzed or not. Options (True | False)",
        choices=[True, False],
        default=True,
    )
    args = parser.parse_args()

    main(args)
