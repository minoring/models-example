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
  import IPython.display as display
  mpl.rcParams['figure.figsize'] = (12,12)
  mpl.rcParams['axes.grid'] = False
  display.clear_output(wait=True)
  display.display(Image.fromarray(img))

  # fig = plt.figure()
  # print('img shape:', img.shape)
  # # Normalize for display.
  # img = img.astype('uint8')
  # if title is not None:
  #   plt.title(title)
  # plt.imshow(img)
  # fig.canvas.draw_idle()
  # plt.pause(.001)


def load_model():
  pass


def save_model():
  pass
