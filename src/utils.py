from os import listdir
from os.path import isfile, join
import configparser

config = configparser.ConfigParser()
config.read("config.ini")


def create_text_color(predicted_class, prediction_proba):
    if predicted_class == 1:
        text = "Ball out of game "
        color = (220, 20, 60)
    else:
        text = "Ball in game "
        color = (0, 100, 0)
        prediction_proba = 1 - prediction_proba
    text += "(" + str(round(prediction_proba, 2)) + ")"
    return text, color


def get_number_of_files(path):
    classes = config["Training"]["classes"].split(",")
    print(classes)
    for i in range(len(classes)):
        adapted_path = path + str(i) + "/"
        files = [f for f in listdir(adapted_path) if isfile(join(adapted_path, f))]

    return len(files)
