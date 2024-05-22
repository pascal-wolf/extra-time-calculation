def create_text_color(predicted_class, prediction_proba):
    """
    Create text and color based on the predicted class.

    Parameters:
    predicted_class (int): The predicted class.
    prediction_proba (float): The prediction probability.

    Returns:
    tuple: The text and color.
    """
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
    """
    Calculate extra time.

    Parameters:
    time_stop_counter (int): The time stop counter.
    fps (int): The frames per second.

    Returns:
    int: The extra time in seconds.
    """
    seconds = time_stop_counter / fps
    return round(seconds)
