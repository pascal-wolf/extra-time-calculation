import keras
import numpy as np
import tensorflow as tf
from IPython.display import Image, display
from matplotlib import cm


def get_img_array(img_path, size):
    """
    Load an image and convert it to a numpy array.

    Parameters:
    img_path (str): Path to the image file.
    size (tuple): Target size for the image.

    Returns:
    array: A numpy array representing the image.
    """
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for an image.

    Parameters:
    img_array (array): The image as a numpy array.
    model (Model): The keras model.
    last_conv_layer_name (str): Name of the last convolutional layer in the model.
    pred_index (int, optional): Index of the predicted class. If None, the most probable class is used.

    Returns:
    heatmap: A numpy array representing the Grad-CAM heatmap.
    """
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(
    img, heatmap, cam_path="cam.jpg", alpha=0.4, plot_image=False, save=False
):
    """
    Save and display a Grad-CAM heatmap superimposed on the original image.

    Parameters:
    img (Image): The original image.
    heatmap (array): The Grad-CAM heatmap.
    cam_path (str, optional): Path to save the superimposed image. Default is "cam.jpg".
    alpha (float, optional): Alpha value for the heatmap when superimposing. Default is 0.4.
    plot_image (bool, optional): Whether to display the image. Default is False.
    save (bool, optional): Whether to save the image. Default is False.

    Returns:
    superimposed_img: A numpy array representing the superimposed image.
    """
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    if save:
        superimposed_img.save(cam_path)
    if plot_image:
        display(Image(cam_path))
    return np.array(superimposed_img)
