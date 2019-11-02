"""Prepare dataset to run neural style"""
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from absl import flags
from tensorflow.keras.preprocessing import image
from utils import imshow
from PIL import Image

IMG_MAX_SIZE = 512


def load_and_process_img(path_to_img):
  """Load image from path and apply VGG19 preprocessing

  Args:
    path_to_image: path of image.
  
  Returns:
    Numpy array image of VGG19 preprocessed 
  """
  img = _load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img * 255)
  img = tf.image.resize(img, (224, 224))
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
    A 3D Tensor of image. Range [0, 1]
  """
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = IMG_MAX_SIZE / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


if __name__ == '__main__':
  """Show Example content image and style image"""

  plt.figure(figsize=(10, 10))

  content = _load_img(flags.FLAGS.content)
  style = _load_img(flags.FLAGS.style)

  plt.subplot(1, 2, 1)
  imshow(content, 'Content Image')
  plt.subplot(1, 2, 2)
  imshow(style, 'Style Image')
  plt.show()
