import configparser
import os
import random

import cv2

config = configparser.ConfigParser()
config.read("config.ini")


def analyse_results(result_df):
    """ Method to plot wrongly classified frames
    Args:
        result_df: Model result as Pandas DataFrame
    """
    wrong_list_idx = result_df[result_df["Prediction"] != result_df["Label"]].index
    frame_idx = random.choice(wrong_list_idx)
    frame_path = result_df.iloc[[frame_idx]]["Path"].item()
    path = os.path.join(str(config["Training"]["val_images_path"]), frame_path)

    img = cv2.imread(path)
    img = cv2.resize(img, (800, 500))
    cv2.imshow("Originalbild", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
