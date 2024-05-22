from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array


class FixedImageDataGenerator(ImageDataGenerator):
    """
    Image data generator with fixed standardization method.

    Inherits from keras.preprocessing.image.ImageDataGenerator.
    """

    def standardize(self, x):
        """
        Overrides the standardize method of the parent class.

        Parameters:
        x (numpy array): The input image.

        Returns:
        numpy array: The standardized image.
        """
        if self.featurewise_center:
            x = ((x / 255.0) - 0.5) * 2.0
        return x


def preprocess_single_frame(image, is_mobile_net=False):
    """
    Preprocess a single frame.

    Parameters:
    image (numpy array): The input image.
    is_mobile_net (bool): Whether the model is MobileNet.

    Returns:
    numpy array: The preprocessed image.
    """
    image = img_to_array(image)
    if is_mobile_net:
        image = (image - image.max()) / (image.max() - image.min())
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    else:
        image = ((image / 255.0) - 0.5) * 2.0
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    return image


def create_generators(config):
    """
    Create image data generators for training and validation.

    Parameters:
    config (dict): Configuration dictionary.

    Returns:
    tuple: The training and validation generators.
    """
    train_images_path = str(config["TRAINING"]["train_images_path"])
    val_images_path = str(config["TRAINING"]["val_images_path"])
    height = int(config["DEFAULT"]["image_height"])
    width = int(config["DEFAULT"]["image_width"])
    batch_size = int(config["TRAINING"]["batch_size"])
    dimension = int(config["DEFAULT"]["image_dimensions"])
    model_type = str(config["MODEL"]["model_type"])

    train_datagen = FixedImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        train_images_path,
        target_size=(height, width),
        class_mode="binary" if model_type.lower() == "standard" else "categorical",
        batch_size=batch_size,
        color_mode="grayscale" if dimension == 1 else "rgb",
    )

    val_datagen = FixedImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
    )

    validation_generator = val_datagen.flow_from_directory(
        val_images_path,
        class_mode="binary" if model_type.lower() == "standard" else "categorical",
        target_size=(height, width),
        batch_size=batch_size,
        color_mode="grayscale" if dimension == 1 else "rgb",
    )

    return train_generator, validation_generator
