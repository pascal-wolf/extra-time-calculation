import numpy as np
from tensorflow.keras.utils import img_to_array


def preprocess_single_frame(image):
    """
    Preprocessing of a single frame

    Args:
        image (PIL): Single Frame in PIL format

    Returns:
        np.array: Numpy Array of a 3 channel image
    """

    image = img_to_array(image)
    image /= 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image
