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
    return round(seconds)
