"""Implementation of paper image super-resolution using deep convolutional networks.

Related papers/blogs:
  https://arxiv.org/pdf/1501.00092.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import tensorflow as tf
import numpy as np

from srcnn.flags import define_flags
from srcnn.model import SRCNN


def run(flags_obj):
  srcnn = SRCNN(image_size=flags_obj.image_size,
               label_size=flags_obj.label_size,
               batch_size=flags_obj.batch_size,
               filter_sizes=(9, 1, 5),
               c_dim=flags_obj.c_dim,
               checkpoint_dir=flags_obj.checkpoint_dir,
               sample_dir=flags_obj.sample_dir)
  if flags_obj.is_train:


def main(_):
  define_flags()
  flags_obj = flags.FLAGS
  if not os.path.exists(flags_obj.checkpoint_dir):
    os.mkdir(flags_obj.checkpoint_dir)
  if not os.path.exists(flags_obj.sample_dir):
    os.mkdir(flags_obj.sample_dir)

  run(flags_obj)


if __name__ == '__main__':
  app.run(main)