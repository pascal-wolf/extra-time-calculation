import configparser
import logging

import cv2
import numpy as np
from tensorflow import keras

from src.explainability import make_gradcam_heatmap, save_and_display_gradcam
from src.model import f1, precision, recall
from src.preprocess import preprocess_single_frame
from src.utils import calculate_extra_time, create_text_color


IS_CNN = True
IS_MOBILENET = False
SCENE = "Foulspiel"

config = configparser.ConfigParser()
logging.info("Loading Model ...")

if IS_CNN:
    config.read("./configs/config.ini")
    model = keras.models.load_model(
        "./trained_models/cnn_individual",
        custom_objects={"precision": precision, "recall": recall, "f1": f1},
    )
    np_name = SCENE + "cnn.npy"
    MODEL_NAME = "cnn"
else:
    if IS_MOBILENET:
        config.read("./configs/config_mobilenet.ini")
        model = keras.models.load_model(
            "./trained_models/mobilenet.keras",
            custom_objects={"precision": precision, "recall": recall, "f1": f1},
        )
        MODEL_NAME = "mobilenet"
    else:
        config.read("./configs/config_efficientnet.ini")
        model = keras.models.load_model(
            "./trained_models/efficientnet.keras",
            custom_objects={"precision": precision, "recall": recall, "f1": f1},
        )
        MODEL_NAME = "efficientnet"
np_name = SCENE + MODEL_NAME + ".npy"
logging.info("Model loaded")


path = f"./data/one_game/Test_{SCENE}.mp4"

cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
time_stop_counter = 0

out = cv2.VideoWriter(
    "./eckball_demo.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    25,
    (1920, 710),
)

preds = []

while cap.isOpened():
    ret, original_frame = cap.read()
    if not ret:
        logging.error("Can't receive frame (stream end?). Exiting ...")
        break

    cropped_frame = original_frame[230:-130,]

    if int(config["DEFAULT"]["image_dimensions"]) == 1:
        frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

    frame_resized = cv2.resize(
        frame,
        dsize=(
            int(config["DEFAULT"]["image_height"]),
            int(config["DEFAULT"]["image_width"]),
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    if IS_CNN:
        frame_preprocessed = preprocess_single_frame(frame_resized)
    else:
        if IS_MOBILENET:
            frame_preprocessed = preprocess_single_frame(
                frame_resized, is_mobile_net=IS_MOBILENET
            )
        else:
            frame_preprocessed = frame_resized.copy()

    if IS_CNN:
        prediction_proba = model.predict(frame_preprocessed, verbose=0)[0][0]
    elif IS_MOBILENET:
        prediction_proba = model.predict(frame_preprocessed, verbose=0)[0][0]
    else:
        prediction_proba = model.predict(
            np.expand_dims(frame_preprocessed, axis=0), verbose=0
        )[0][0]

    predicted_class = np.where(
        prediction_proba <= float(config["DEFAULT"]["threshold"]), 0, 1
    )
    preds.append(predicted_class)
    if predicted_class == 1:
        time_stop_counter += 1

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

    if bool(config["DEFAULT"]["verbose"]):
        logging.info(
            f"Model predicted class {predicted_class} ({round(prediction_proba,2)})"
        )

    if IS_CNN:
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
