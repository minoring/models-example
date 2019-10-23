"""Implementation of paper A Neural Algorithm of Artistic Style.

  https://arxiv.org/abs/1508.06576
  Using tf2.0 and keras API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


# Define global constants
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), 'resources')
CONTENT_PATH = os.path.join(
    RESOURCES_PATH, 'Green_Sea_Turtle_grazing_seagrass.jpg')
STYLE_PATH = os.path.join(
    RESOURCES_PATH, 'The_Great_Wave_off_Kanagawa.jpg')


def _setup_default_config():
    """Setup default configuration to run neural style"""


if __name__ == '__main__':
    pass