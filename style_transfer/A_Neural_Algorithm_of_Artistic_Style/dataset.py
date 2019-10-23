"""Prepare dataset to run neural style"""
import os

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from PIL import Image

IMG_MAX_SIZE = 512


def load_img(path_to_img):
    """Load image using path.

    Resize image if the image size is bigger than 512x512.
    Add batch dimension to the image.

    Args:
      path_to_img: path to image.
    
    Returns:
      numpy array of image.
    """
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = IMG_MAX_SIZE / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),
                     Image.ANTIALIAS)  # a high-quality downsampling filter

    img = image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def imshow(img, title=None):
    """Show image.

    Args:
      img: numpy array of image of (batch_size, width or height, width or height)
    """
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    if title is not None:
        plt.title(title)
    plt.imshow(out)
    plt.show()


if __name__ == '__main__':
    """Show turtle image"""
    RESOURCES_PATH = os.path.join(os.path.dirname(__file__), 'resources')
    CONTENT_PATH = os.path.join(
        RESOURCES_PATH, 'Green_Sea_Turtle_grazing_seagrass.jpg')
    STYLE_PATH = os.path.join(
        RESOURCES_PATH, 'The_Great_Wave_off_Kanagawa.jpg')
    imshow(load_img(CONTENT_PATH))