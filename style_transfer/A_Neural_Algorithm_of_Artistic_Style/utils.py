"""Util functions and classes used by neural style"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
# from dataset import deprocess_img

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import imageio
import glob

from PIL import Image


def tensor_to_image(tensor):
  """Convert tensor into PIL.Image

  Args:
    tensor: Tensor.
  Returns:
    PIL.Image object of tensor.
  """
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor) > 3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)


def log_training_info(plot_img, losses, step, elapsed_time):
  imshow(plot_img, title='Generated image')
  print('Steps: {}'.format(step))
  print('total loss: {:.4e}, '.format(losses['total_loss']),
        'style loss: {:.4e}, '.format(losses['style_loss']),
        'content loss: {:.4e}, '.format(losses['content_loss']),
        'time: {:.4f}s'.format(elapsed_time))


def imshow(img, title=None):
  """Show image.

  Args:
    img: Tensor of image of (batch_size, width or height, width or height)
    title: String of title of image.
  """
  if len(img.shape) > 3:
    img = tf.squeeze(img, axis=0)
  # Normalize for display.
  if title is not None:
    plt.title(title)
  plt.imshow(img)


def plot_history(history):
  """Plot history of style, content, and total loss"""
  plt.figure()
  plt.plot(history['total_losses'], label='total loss')
  plt.plot(history['style_losses'], label='style loss')
  plt.plot(history['content_losses'], label='content loss')
  plt.xlabel('Training step')
  plt.ylabel('loss')
  plt.legend()
  # plt.ylim([0.5, 1])
  plt.show()


def create_gif():
  """Create gif using saved images"""
  anim_file = 'style.gif'

  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.jpg')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2 * (i**0.5)
      if round(frame) > round(last):
        last = frame
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


def clip_0_1(img):
  return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


def clip_0_255(img):
  return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)


def load_model():
  pass


def save_model():
  pass
