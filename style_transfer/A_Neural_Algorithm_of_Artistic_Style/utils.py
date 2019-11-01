"""Util functions and classes used by neural style"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
# from dataset import deprocess_img

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
from PIL import Image


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
    img: numpy array of image of (batch_size, width or height, width or height)
    title: String of title of image.
  """
  # import IPython.display as display
  # mpl.rcParams['figure.figsize'] = (12,12)
  # mpl.rcParams['axes.grid'] = False
  # display.clear_output(wait=True)
  # display.display(Image.fromarray(img))
  plt.clf()
  fig = plt.figure()
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


def load_model():
  pass


def save_model():
  pass
