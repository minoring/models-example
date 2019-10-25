"""Util functions and classes used by neural style"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import matplotlib.pyplot as plt
import numpy as np


def define_flags():
    flags.DEFINE_integer('num_training_steps', 1000,
                         'The number of training steps for neural style')


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
    """
    # Remove the batch dimension.
    out = np.squeeze(img, axis=0)
    # Normalize for display.
    out = out.astype('uint8')
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def load_model():
    pass


def save_model():
    pass
