from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array

from src.settings import Settings


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
        directory=Settings.train_images_path,
        # All images will be resized to target height and width.
        target_size=(Settings.image_height, Settings.image_width),
        class_mode="binary",
        batch_size=Settings.batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        color_mode="grayscale" if Settings.image_dimensions == 1 else "rgb",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = test_datagen.flow_from_directory(
        directory=Settings.val_images_path,
        class_mode="binary",
        target_size=(Settings.image_height, Settings.image_width),
        batch_size=Settings.batch_size,
        color_mode="grayscale" if Settings.image_dimensions == 1 else "rgb",
    )

    return train_generator, validation_generator
