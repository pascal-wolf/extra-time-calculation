import tensorflow as tf
from keras import backend as K
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0


def f1(y_true, y_pred):
    """
    Calculate the F1 score.

    Parameters:
    y_true (Tensor): Ground truth labels.
    y_pred (Tensor): Predicted labels.

    Returns:
    float: F1 score.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def precision(y_true, y_pred):
    """
    Calculate the precision.

    Parameters:
    y_true (Tensor): Ground truth labels.
    y_pred (Tensor): Predicted labels.

    Returns:
    float: Precision score.
    """
    return precision_score(
        y_true.numpy().argmax(axis=1),
        y_pred.numpy().argmax(axis=1),
    )


def recall(y_true, y_pred):
    """
    Calculate the recall.

    Parameters:
    y_true (Tensor): Ground truth labels.
    y_pred (Tensor): Predicted labels.

    Returns:
    float: Recall score.
    """
    return recall_score(
        y_true.numpy().argmax(axis=1),
        y_pred.numpy().argmax(axis=1),
    )


def create_cnn_model(input_shape, kernel_size, n_cnn_layer, n_hidden_layer):
    """
    Create a CNN model.

    Parameters:
    input_shape (tuple): Shape of the input.
    kernel_size (int): Size of the convolution kernel.
    n_cnn_layer (int): Number of convolutional layers.
    n_hidden_layer (int): Number of hidden layers.

    Returns:
    Model: A compiled keras model.
    """
    model = models.Sequential()
    model.add(
        layers.Conv2D(16, kernel_size, activation="relu", input_shape=input_shape)
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    for cnn_layer in range(n_cnn_layer):
        output_channels = (cnn_layer + 2) * 16
        model.add(layers.Conv2D(output_channels, kernel_size, activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    for hidden_layer in range(n_hidden_layer, 1, -1):
        model.add(layers.Dense(hidden_layer * 8, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            f1,
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
        steps_per_execution=1,
    )
    return model


def create_model_v2(input_shape):
    """
    Create a CNN model (version 2).

    Parameters:
    input_shape (tuple): Shape of the input.

    Returns:
    Model: A compiled keras model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            f1,
        ],
        steps_per_execution=1,
    )
    return model


def create_efficientnetvtwo_model(input_shape):
    """
    Create an EfficientNetV2 model.

    Parameters:
    input_shape (tuple): Shape of the input.

    Returns:
    Model: A compiled keras model.
    """
    model_ENB0 = EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    model_ENB0.trainable = True

    model = models.Sequential()
    model.add(model_ENB0)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        run_eagerly=True,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            f1,
        ],
    )
    return model


def create_mobilenet_model(input_shape):
    """
    Create a MobileNetV2 model.

    Parameters:
    input_shape (tuple): Shape of the input.

    Returns:
    Model: A compiled keras model.
    """
    model_mobilenet = MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    model_mobilenet.trainable = True

    model = models.Sequential()
    model.add(model_mobilenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        run_eagerly=True,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            f1,
        ],
    )
    return model
