"""Implementation of paper A Neural Algorithm of Artistic Style.

Used tf2.0 and keras API.

Related papers/blogs:
  https://arxiv.org/abs/1508.06576
  https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from dataset import load_and_process_img
from model import NUM_CONTENT_LAYERS
from model import NUM_STYLE_LAYERS
from model import neural_style
from model import compute_style_loss
from model import compute_content_loss

import tensorflow as tf
import os

# Define global constants
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), 'resources')
CONTENT_PATH = os.path.join(RESOURCES_PATH,
                            'Green_Sea_Turtle_grazing_seagrass.jpg')
STYLE_PATH = os.path.join(RESOURCES_PATH, 'The_Great_Wave_off_Kanagawa.jpg')


def run(flags_obj):
    """Run Neural Style transfer training and eval loop using native Keras APIs.

    Args:
      flags_obj: An object containing parsed flag values.

    Raises:
     ValueError: If fp16 is passed as it is not currently supported.

    Returns:
      Dictionary of training and eval stats.
    """
    pass
    #  model = neural_style()
    # org_style_reprs, org_content_reprs = (
    #     _compute_original_image_feature_representation(model, CONTENT_PATH,
    #                                                    STYLE_PATH))


def _compute_original_image_feature_representation(model, content_path,
                                                   style_path):
    """Helper function to compute our content and style feature representation
    original image

    This function will simply load and preprocess both content and style
    images from their path. Then it will feed them through the network to obtain
    the outputs of the intermediate layers.

    Args:
      model: The model that we are using.
      content_path: The path to the content image.
      style_path: The path to the style image.
    
    Returns:
      Tuple of style and content representation of original image.
    """
    # Load our images in
    content_img = load_and_process_img(content_path)
    style_img = load_and_process_img(style_path)

    style_outputs = model(style_img)
    content_outputs = model(content_img)

    # Get the style and content feature representations from our model
    style_reprs = [
        style_layer[0] for style_layer in style_outputs[:NUM_STYLE_LAYERS]
    ]
    content_reprs = [
        content_layer[0]
        for content_layer in content_outputs[NUM_STYLE_LAYERS:]
    ]

    return style_reprs, content_reprs


def main():
    # model_helpers.apply_clean(flags.FLAGS)
    # with logger.benchmark_context(flags.FLAGS):
    #     stats = run(flags.FLAGS)
    # logging.info('Run stats:\n%s', stats)
    pass


if __name__ == '__main__':
    app.run(main)
