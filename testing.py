import numpy as np
import cv2
import os
from ultralyticsplus import YOLO, render_result
from tensorflow import keras
from src.preprocess import preprocess_single_frame
from src.utils import create_text_color
import logging
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

logging.info("Loading Model ...")
model = keras.models.load_model("./trained_models/classification_v2")
logging.info("Model loaded")

path = "./data/test_data/Test.mp4"

cap = cv2.VideoCapture(path)
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

    # Text on Image
    font = cv2.QT_FONT_NORMAL
    org = (1300, 120)
    fontScale = 1

    thickness = 2
    text, color = create_text_color(predicted_class, prediction_proba)
    frame = cv2.putText(
        frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA
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
