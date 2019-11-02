"""Flags for neural style"""
from absl import flags


def define_flags():
  flags.DEFINE_integer('num_training_steps', 1000,
                       'The number of training steps for neural style')
  flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate of training steps')
  flags.DEFINE_integer('display_interval', 50,
                       'Display per training step to show image')
  flags.DEFINE_string(
      'style',
      'resources/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
      'Style image file path')
  flags.DEFINE_string('content', 'resources/IMG_0886.JPG',
                      'Content image file path')
