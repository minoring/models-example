"""Flags for neural style"""
from absl import flags


def define_flags():
  flags.DEFINE_integer('num_training_steps', 2000,
                       'The number of training steps for neural style')
  flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate of training steps')
  flags.DEFINE_integer('display_interval', 100,
                       'Display per training step to show image')