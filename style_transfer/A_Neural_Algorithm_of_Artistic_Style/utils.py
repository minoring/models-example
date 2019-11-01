"""Util functions and classes used by neural style"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
# from dataset import deprocess_img

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import imageio
import glob

from PIL import Image


def log_training_info(fig, plot_img, losses, step, elapsed_time):
  imshow(fig, plot_img, title='Generated image')
  print('Steps: {}'.format(step))
  print('total loss: {:.4e}, '.format(losses['total_loss']),
        'style loss: {:.4e}, '.format(losses['style_loss']),
        'content loss: {:.4e}, '.format(losses['content_loss']),
        'time: {:.4f}s'.format(elapsed_time))


def imshow(fig, img, title=None):
  """Show image.

  Args:
    img: numpy array of image of (batch_size, width or height, width or height)
    title: String of title of image.
  """
  plt.clf()
  print('img shape:', img.shape)
  # Normalize for display.
  img = img.astype('uint8')
  if title is not None:
    plt.title(title)
  plt.imshow(img)
  plt.pause(.001)
  fig.canvas.draw_idle()


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


def load_model():
  pass


def save_model():
  pass
