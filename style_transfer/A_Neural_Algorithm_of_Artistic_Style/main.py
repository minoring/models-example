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
from dataset import deprocess_img
from model import NUM_CONTENT_LAYERS
from model import NUM_STYLE_LAYERS
from model import NORM_MEANS
from model import neural_style
from model import compute_losses
from flags import define_flags
from utils import log_training_info

import tensorflow as tf
import os
import time

# Define global constants
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), 'resources')
CONTENT_PATH = os.path.join(RESOURCES_PATH,
                            'Green_Sea_Turtle_grazing_seagrass.jpg')
STYLE_PATH = os.path.join(RESOURCES_PATH, 'The_Great_Wave_off_Kanagawa.jpg')


def run(flags_obj):
  """Run Neural Style transfer training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Returns:
    Dictionary of training and eval stats.
  """
  model = neural_style()
  org_style_reprs, org_content_reprs = (
      _compute_original_image_feature_representation(model))

  for layer in model.layers:
    layer.trainable = False

  # Get image we are generating. Content image for now.
  gen_img = load_and_process_img(CONTENT_PATH)
  gen_img = tf.Variable(gen_img)
  opt = tf.optimizers.Adam(learning_rate=flags_obj.learning_rate,
                           beta_1=0.99,
                           epsilon=1e-1)

  best_loss, best_img = float('inf'), None
  start_time = time.time()
  history = dict(losses=[], images=[])

  for step in range(flags_obj.num_training_steps):
    with tf.GradientTape() as tape:
      tape.watch(gen_img)
      losses = compute_losses(model, gen_img, org_style_reprs,
                              org_content_reprs)
    total_loss = losses['total_loss']
    history['losses'].append(losses)

    grads = tape.gradient(total_loss, gen_img)

    opt.apply_gradients([(grads, gen_img)])

    # Clip by [-NORM_MEANS, 244 - NORM_MEANS] range.
    cliped_img = tf.clip_by_value(gen_img, -NORM_MEANS, 255 - NORM_MEANS)
    gen_img.assign(cliped_img)

    plot_img = gen_img.numpy()
    plot_img = deprocess_img(plot_img)
    history['images'].append(plot_img)

    if total_loss < best_loss:
      best_loss = total_loss
      best_img = plot_img

    if step % flags_obj.display_interval == 0:
      log_training_info(plot_img, losses, step,
                        time.time() - start_time)


def _compute_original_image_feature_representation(model):
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
  content_img = load_and_process_img(CONTENT_PATH)
  style_img = load_and_process_img(STYLE_PATH)

  style_outputs = model(style_img)
  content_outputs = model(content_img)

  # Get the style and content feature representations from our model
  style_reprs = [
      style_layer[0] for style_layer in style_outputs[:NUM_STYLE_LAYERS]
  ]
  content_reprs = [
      content_layer[0] for content_layer in content_outputs[NUM_STYLE_LAYERS:]
  ]

  return style_reprs, content_reprs


def main(_):
  """Main function to run neural style"""
  define_flags()
  run(flags.FLAGS)


if __name__ == '__main__':
  app.run(main)