import configparser
import logging

import cv2
import numpy as np
from tensorflow import keras

from src.explainability import make_gradcam_heatmap, save_and_display_gradcam
from src.model import f1, precision, recall
from src.preprocess import preprocess_single_frame
from src.utils import calculate_extra_time, create_text_color


def load_model(is_cnn, is_mobilenet, scene):
    """
    Load the appropriate model based on the given parameters.

    Parameters:
    is_cnn (bool): Whether the model is a CNN.
    is_mobilenet (bool): Whether the model is MobileNet.
    scene (str): The scene for which the model is being loaded.

    Returns:
    tuple: The loaded model and the name of the model.
    """
    config = configparser.ConfigParser()
    logging.info("Loading Model ...")

    if is_cnn:
        config.read("./configs/config.ini")
        model = keras.models.load_model(
            "./trained_models/cnn_individual",
            custom_objects={"precision": precision, "recall": recall, "f1": f1},
        )
        model_name = "cnn"
    else:
        if is_mobilenet:
            config.read("./configs/config_mobilenet.ini")
            model = keras.models.load_model(
                "./trained_models/mobilenet.keras",
                custom_objects={"precision": precision, "recall": recall, "f1": f1},
            )
            model_name = "mobilenet"
        else:
            config.read("./configs/config_efficientnet.ini")
            model = keras.models.load_model(
                "./trained_models/efficientnet.keras",
                custom_objects={"precision": precision, "recall": recall, "f1": f1},
            )
            model_name = "efficientnet"

    np_name = scene + model_name + ".npy"
    logging.info("Model loaded")

    return model, np_name, config


def process_frame(frame, config, is_cnn, is_mobilenet):
    """
    Process a single frame.

    Parameters:
    frame (numpy array): The input frame.
    config (ConfigParser): The configuration parser.
    is_cnn (bool): Whether the model is a CNN.
    is_mobilenet (bool): Whether the model is MobileNet.

    Returns:
    numpy array: The processed frame.
    """
    if int(config["DEFAULT"]["image_dimensions"]) == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_resized = cv2.resize(
        frame,
        dsize=(
            int(config["DEFAULT"]["image_height"]),
            int(config["DEFAULT"]["image_width"]),
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    if is_cnn:
        frame_preprocessed = preprocess_single_frame(frame_resized)
    else:
        if is_mobilenet:
            frame_preprocessed = preprocess_single_frame(
                frame_resized, is_mobile_net=is_mobilenet
            )
        else:
            frame_preprocessed = frame_resized.copy()

    return frame_preprocessed


def predict_class(model, frame_preprocessed, config, is_cnn, is_mobilenet):
    """
    Predict the class of the given frame.

    Parameters:
    model (keras Model): The trained model.
    frame_preprocessed (numpy array): The preprocessed frame.
    config (ConfigParser): The configuration parser.
    is_cnn (bool): Whether the model is a CNN.
    is_mobilenet (bool): Whether the model is MobileNet.

    Returns:
    tuple: The predicted class and the prediction probability.
    """
    if is_cnn or is_mobilenet:
        prediction_proba = model.predict(frame_preprocessed, verbose=0)[0][0]
    else:
        prediction_proba = model.predict(
            np.expand_dims(frame_preprocessed, axis=0), verbose=0
        )[0][0]

    predicted_class = np.where(
        prediction_proba <= float(config["DEFAULT"]["threshold"]), 0, 1
    )

    return predicted_class, prediction_proba


def load_video(path):
    """
    Load a video from the given path.

    Parameters:
    path (str): The path to the video.

    Returns:
    cv2.VideoCapture: The loaded video.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps


def create_video_writer():
    """
    Create a video writer.

    Returns:
    cv2.VideoWriter: The video writer.
    """
    out = cv2.VideoWriter(
        "./eckball_demo.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (1920, 710),
    )
    return out


def process_video(cap, fps, model, config, is_cnn, is_mobilenet):
    """
    Process a video.

    Parameters:
    cap (cv2.VideoCapture): The video to process.
    fps (int): The frames per second of the video.
    model (keras Model): The trained model.
    config (ConfigParser): The configuration parser.
    is_cnn (bool): Whether the model is a CNN.
    is_mobilenet (bool): Whether the model is MobileNet.

    Returns:
    list: The predicted classes for each frame.
    """
    time_stop_counter = 0
    preds = []

    while cap.isOpened():
        ret, original_frame = cap.read()
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            break

        cropped_frame = original_frame[230:-130,]
        frame_preprocessed = process_frame(cropped_frame, config, is_cnn, is_mobilenet)
        predicted_class, prediction_proba = predict_class(
            model, frame_preprocessed, config, is_cnn, is_mobilenet
        )

        preds.append(predicted_class)
        if predicted_class == 1:
            time_stop_counter += 1

        original_frame = annotate_frame(
            original_frame, predicted_class, prediction_proba, time_stop_counter, fps
        )

        if bool(config["DEFAULT"]["verbose"]):
            logging.info(
                f"Model predicted class {predicted_class} ({round(prediction_proba,2)})"
            )

        if is_cnn:
            model.layers[-1].activation = None
            heatmap = make_gradcam_heatmap(frame_preprocessed, model, "conv2d_30")

            explained_image = save_and_display_gradcam(
                cropped_frame, heatmap, plot_image=False
            )
            cv2.imshow("explained_image", explained_image)

        cv2.imshow("frame", original_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return preds


def annotate_frame(
    original_frame, predicted_class, prediction_proba, time_stop_counter, fps
):
    """
    Annotate a frame with the predicted class and extra time.

    Parameters:
    original_frame (numpy array): The original frame.
    predicted_class (int): The predicted class.
    prediction_proba (float): The prediction probability.
    time_stop_counter (int): The time stop counter.
    fps (int): The frames per second.

    Returns:
    numpy array: The annotated frame.
    """
    font = cv2.QT_FONT_NORMAL
    fontScale = 1
    text, color = create_text_color(predicted_class, prediction_proba)
    original_frame = cv2.putText(
        original_frame, text, (1220, 120), font, fontScale, color, 2, cv2.LINE_AA
    )
    extra_time = calculate_extra_time(time_stop_counter, fps)
    original_frame = cv2.putText(
        original_frame,
        str(int(extra_time)) + "s Extra Time",
        (1560, 120),
        font,
        fontScale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return original_frame
