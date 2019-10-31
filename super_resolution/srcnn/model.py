"""Define srcnn model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from srcnn.utils import read_data_h5

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class SRCNN(tf.keras.Model):
  """Define model using tf.keras API"""

  def __init__(self,
               image_size=33,
               label_size=21,
               batch_size=128,
               filter_sizes=(9, 1, 5),
               c_dim=1,
               checkpoint_dir=None,
               sample_dir=None):
    super(SRCNN, self).__init__()
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size
    self.filter_sizes = filter_sizes
    self.c_dim = c_dim
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir

  def build(self, input_shape):
    """Build a model"""
    self.conv2a = tf.keras.layers.Conv2D(64,
                                         self.filter_sizes[0],
                                         activation='relu',
                                         padding='valid')
    self.conv2b = tf.keras.layers.Conv2D(32,
                                         self.filter_sizes[1],
                                         activation='relu',
                                         padding='valid')
    self.conv2c = tf.keras.layers.Conv2D(1,
                                         self.filter_sizes[2],
                                         padding='valid')

  def call(self, input_tensor):
    x = self.conv2a(input_tensor)
    x = self.conv2b(x)
    x = self.conv2c(x)

    return x