import numpy as np
from urllib.parse import urlparse
import mlflow
from mlflow import keras
from src.model import create_model
import logging
from src.preprocess import create_generators
from src.utils import get_number_of_files
import configparser

config = configparser.ConfigParser()
config.read("config.ini")


height = int(config["DEFAULT"]["image_height"])
width = int(config["DEFAULT"]["image_width"])
threshold = float(config["DEFAULT"]["threshold"])
input_shape = (
    height,
    width,
    3,
)
batch_size = int(config["Training"]["batch_size"])
epochs = int(config["Training"]["epochs"])

train_images_path = str(config["Training"]["train_images_path"])
val_images_path = str(config["Training"]["val_images_path"])

number_of_training_images = get_number_of_files(train_images_path)
number_of_validation_images = get_number_of_files(val_images_path)

classes = config["Training"]["classes"].split(",")
number_of_classes = len(classes)


if __name__ == "__main__":
    keras.autolog()

    train_generator, validation_generator = create_generators()

    # Here Hyperparameter

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        model = create_model(input_shape)
        history = model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=number_of_training_images // batch_size,
            validation_data=validation_generator,
        )

        predictions_proba = model.predict(validation_generator).flatten()
        predictions = np.where(predictions_proba < threshold, 0, 1).flatten()
        y_true = validation_generator.classes.flatten()

        # Evaluate on Validation data
        scores = model.evaluate(validation_generator, verbose=0)
        for i, metric_name in enumerate(model.metrics_names[1:]):
            print("%s%s: %.2f%%" % ("evaluate ", metric_name, scores[i]))
            mlflow.log_metric(metric_name, scores[i])

        # Log Hyperparameter here
        # mlflow.log_param("alpha", alpha)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            keras.log_model(model, "model", registered_model_name="CNNClassification")
        else:
            keras.log_model(model, "model")
