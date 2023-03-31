from tensorflow.keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from tensorflow.keras import optimizers
from tensorflow.keras.applications import EfficientNetB1
import tensorflow as tf
from keras import backend as K
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

height = int(config["DEFAULT"]["image_height"])
width = int(config["DEFAULT"]["image_width"])
input_shape = (
    height,
    width,
    3,
)
batch_size = int(config["Training"]["batch_size"])
epochs = int(config["Training"]["epochs"])

NUMBER_OF_TRAINING_IMAGES = 1960 + 1041
NUMBER_OF_VALIDATION_IMAGES = 207 + 225
train_images_path = str(config["Training"]["train_images_path"])
val_images_path = str(config["Training"]["val_images_path"])


classes = list(config["Training"]["classes"])
number_of_classes = len(classes)


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
# Note that the validation data should not be augmented!
# and a very important step is to normalise the images through  rescaling
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_images_path,
    # All images will be resized to target height and width.
    target_size=(height, width),
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="binary",
)
validation_generator = test_datagen.flow_from_directory(
    val_images_path,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="binary",
)


model = models.Sequential()
model.add(
    layers.Conv2D(input_shape[0], (3, 3), activation="relu", input_shape=input_shape)
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=NUMBER_OF_TRAINING_IMAGES // batch_size,
    validation_data=validation_generator,
)
model.save("./models/classification_v3")
