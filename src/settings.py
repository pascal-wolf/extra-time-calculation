import configparser

config = configparser.ConfigParser()
config.read("config.ini")


class Settings:
    image_height = int(config["DEFAULT"]["image_height"])
    image_width = int(config["DEFAULT"]["image_width"])
    image_dimensions = int(config["DEFAULT"]["image_dimensions"])  # 1 -> grayscale; 3 -> rgb
    input_shape = (image_height, image_width, image_dimensions)

    batch_size = int(config["TRAINING"]["batch_size"])
    epochs = int(config["TRAINING"]["epochs"])

    train_images_path = str(config["TRAINING"]["train_images_path"])
    val_images_path = str(config["TRAINING"]["val_images_path"])

    threshold = float(config["DEFAULT"]["threshold"])
    verbose = bool(config["DEFAULT"]["verbose"])
    classes = config["TRAINING"]["classes"].split(",")
    number_of_classes = len(classes)
    log_level = str(config["LOGGING"]["log_level"])
