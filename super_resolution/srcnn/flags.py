"""Flags for srcnn"""

from absl import flags


def define_flags():
  flags.DEFINE_integer("epoch", 15000, "Number of epoch [15000]")
  flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
  flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
  flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
  flags.DEFINE_float("learning_rate", 1e-4,
                     "The learning rate of gradient descent algorithm [1e-4]")
  flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
  flags.DEFINE_integer(
      "scale", 3, "The size of scale factor for preprocessing input image [3]")
  flags.DEFINE_integer("stride", 14,
                       "The size of stride to apply input image [14]")
  flags.DEFINE_string("checkpoint_dir", "checkpoint",
                      "Name of checkpoint directory [checkpoint]")
  flags.DEFINE_string("sample_dir", "sample",
                      "Name of sample directory [sample]")
  flags.DEFINE_boolean("is_train", True,
                       "True for training, False for testing [True]")