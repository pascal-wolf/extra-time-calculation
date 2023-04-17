import numpy as np
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import configparser

config = configparser.ConfigParser()
config.read("config.ini")


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


def create_generators():
    train_images_path = str(config["Training"]["train_images_path"])
    val_images_path = str(config["Training"]["val_images_path"])
    height = int(config["DEFAULT"]["image_height"])
    width = int(config["DEFAULT"]["image_width"])
    batch_size = int(config["Training"]["batch_size"])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_images_path,
        # All images will be resized to target height and width.
        target_size=(height, width),
        class_mode="categorical",
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        # color_mode="grayscale",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = test_datagen.flow_from_directory(
        val_images_path,
        class_mode="categorical",
        target_size=(height, width),
        batch_size=batch_size,
        # color_mode="grayscale",
    )

    return train_generator, validation_generator
