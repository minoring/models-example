"""Implementation of paper image super-resolution using deep convolutional networks.

Using TF2.0 Keras API.

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
from srcnn.utils import create_sub_images
from srcnn.utils import read_data_h5
from srcnn.utils import sub_images_exists


def run(flags_obj):
  srcnn = SRCNN(image_size=flags_obj.image_size,
                label_size=flags_obj.label_size,
                batch_size=flags_obj.batch_size,
                filter_sizes=(9, 1, 5),
                c_dim=flags_obj.c_dim,
                checkpoint_dir=flags_obj.checkpoint_dir,
                sample_dir=flags_obj.sample_dir)

  if flags_obj.is_train:
    if not sub_images_exists():
      create_sub_images()
    from srcnn.utils import SUB_IMAGES_PATH
    train_inputs, train_labels = read_data_h5(SUB_IMAGES_PATH)
    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_inputs, train_labels))
    train_dataset = train_dataset.batch(flags_obj.batch_size)
    optimizer = tf.keras.optimizers.SGD(learning_rate=flags_obj.learning_rate)
    srcnn.compile(optimizer=optimizer,
                  loss='mse',
                  metric=['mse'])
    srcnn.fit(train_dataset,
              epochs=flags_obj.train_epochs)
  else:
    # For test set.
    pass


def main(_):
  define_flags()
  flags_obj = flags.FLAGS
  if not os.path.exists(flags_obj.checkpoint_dir):
    os.mkdir(flags_obj.checkpoint_dir)
  if not os.path.exists(flags_obj.sample_dir): # What is sample?
    os.mkdir(flags_obj.sample_dir)

  run(flags_obj)


if __name__ == '__main__':
  app.run(main)