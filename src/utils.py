from os import listdir
from os.path import isfile, join


def create_text_color(predicted_class, prediction_proba):
    if predicted_class == 1:
        text = "Out game "
        color = (60, 20, 220)
    else:
        text = "In game "
        color = (0, 100, 0)
        prediction_proba = 1 - prediction_proba
    text += "(" + "{:2.2f}".format(round(prediction_proba, 2)) + ")"
    return text, color


def calculate_extra_time(time_stop_counter, fps):
    seconds = time_stop_counter / fps
    minutes = seconds / 60
    return round(minutes, 2)


def get_number_of_files(path):
    # classes = Settings.classes
    # logging.info(classes)
    for i in range(2):
        adapted_path = path + str(i) + "/"
        files = [f for f in listdir(adapted_path) if isfile(join(adapted_path, f))]
    return len(files)
