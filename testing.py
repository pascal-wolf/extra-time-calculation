import numpy as np
import cv2
import os
from ultralyticsplus import YOLO, render_result
from tensorflow import keras
from src.preprocess import preprocess_single_frame
from src.utils import create_text_color, calculate_extra_time
import logging
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

logging.info("Loading Model ...")
model = keras.models.load_model("./trained_models/classification_v2")
logging.info("Model loaded")

path = "./data/test_data/Test.mp4"
time_stop_counter = 0
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
# How long we need extra time counter / fps
time_stop_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.error("Can't receive frame (stream end?). Exiting ...")
        break

    # MODEL
    frame_resized = cv2.resize(
        frame,
        dsize=(
            int(config["DEFAULT"]["image_height"]),
            int(config["DEFAULT"]["image_width"]),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    frame_preprocessed = preprocess_single_frame(frame_resized)

    prediction_proba = model.predict(frame_preprocessed, verbose=0)[0][0]
    predicted_class = np.where(
        prediction_proba < float(config["DEFAULT"]["threshold"]), 0, 1
    )
    if predicted_class == 1:
        time_stop_counter += 1
    # Text on Image
    font = cv2.QT_FONT_NORMAL
    fontScale = 1
    text, color = create_text_color(predicted_class, prediction_proba)
    frame = cv2.putText(
        frame, text, (1220, 120), font, fontScale, color, 1, cv2.LINE_AA
    )
    extra_time = calculate_extra_time(time_stop_counter, fps)
    frame = cv2.putText(
        frame,
        "{:3.2f}".format(extra_time) + " Extra Time",
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

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
