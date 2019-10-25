"""Prepare dataset to run neural style"""
import os

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from PIL import Image

IMG_MAX_SIZE = 512


def load_and_process_img(path_to_img):
    """Load image from path and apply VGG19 preprocessing
      Args:
        path_to_image: path of image.
      
      returns:
        Image of VGG19 preprocessed 
    """
    img = _load_img(path_to_img)
    img = tf.keras.applications.vgg19.process_input(img)
    return img


def deprocess_img(processed_img):
    """Inverse preprocess step in order to view the outputs of our optimization
  
    VGG networks are trained on image with each channel normalized by 
    mean = [103.939, 166.779, 123.68] and with channels BGR.
    """
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, axis=0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or "
                               "[height, with, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Make RGB. VGG use BGR color channel.

    # Optimized image may take its values anywhere between  −Inf and Inf,
    # we must clip to maintain our values from within the 0-255 range.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def _load_img(path_to_img):
    """Load image using path.

    Resize image if the image size is bigger than 512x512.
    Add batch dimension to the image.

    Args:
      path_to_img: path to image.
    
    Returns:
      A 3D Numpy array of image.
    """
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = IMG_MAX_SIZE / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),
                     Image.ANTIALIAS)  # A high-quality downsampling filter.

    img = image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension.
    img = np.expand_dims(img, axis=0)
    return img


def imshow(img, title=None):
    """Show image.

    Args:
      img: numpy array of image of (batch_size, width or height, width or height)
    """
    # Remove the batch dimension.
    out = np.squeeze(img, axis=0)
    # Normalize for display.
    out = out.astype('uint8')
    if title is not None:
        plt.title(title)
    plt.imshow(out)


if __name__ == '__main__':
    """Show Example content image and style image"""
    RESOURCES_PATH = os.path.join(os.path.dirname(__file__), 'resources')
    CONTENT_PATH = os.path.join(RESOURCES_PATH,
                                'Green_Sea_Turtle_grazing_seagrass.jpg')
    STYLE_PATH = os.path.join(RESOURCES_PATH,
                              'The_Great_Wave_off_Kanagawa.jpg')

    plt.figure(figsize=(10, 10))

    content = _load_img(CONTENT_PATH).astype('uint8')
    style = _load_img(STYLE_PATH).astype('uint8')

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')
    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.show()
    