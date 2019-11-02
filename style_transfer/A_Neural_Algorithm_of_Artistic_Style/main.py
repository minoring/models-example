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
from utils import plot_history
from utils import create_gif
from utils import imshow
from utils import clip_0_1
from utils import clip_0_255
from dataset import _load_img

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt


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
  gen_img = _load_img(flags_obj.content)
  gen_img = tf.Variable(gen_img)

  opt = tf.optimizers.Adam(learning_rate=flags_obj.learning_rate,
                           beta_1=0.99,
                           epsilon=1e-1)

  best_loss, best_img = float('inf'), None
  start_time = time.time()
  history = dict(total_losses=[], style_losses=[], content_losses=[], images=[])

  for step in range(flags_obj.num_training_steps):
    with tf.GradientTape() as tape:
      tape.watch(gen_img)
      losses = compute_losses(model, gen_img, org_style_reprs,
                              org_content_reprs)
    total_loss = losses['total_loss']
    history['total_losses'].append(total_loss)
    history['style_losses'].append(losses['style_loss'])
    history['content_losses'].append(losses['content_loss'])

    grads = tape.gradient(total_loss, gen_img)

    opt.apply_gradients([(grads, gen_img)])
    gen_img.assign(clip_0_1(gen_img))

    # plot_img = gen_img.numpy()
    # plot_img = deprocess_img(plot_img)
    # history['images'].append(plot_img)

    if total_loss < best_loss:
      best_loss = total_loss
      best_img = gen_img

    if step % flags_obj.display_interval == 0:
      plt.imshow(gen_img.numpy()[0])
      plt.axis('off')
      plt.savefig('image_at_step_{:04d}.jpg'.format(step))
      log_training_info(gen_img, losses, step, time.time() - start_time)
  plot_history(history)
  create_gif()


def _compute_original_image_feature_representation(model):
  """Helper function to compute our content and style feature representation
  original image

  This function will simply load and preprocess both content and style
  images from their path. Then it will feed them through the network to obtain
  the outputs of the intermediate layers.

  Args:
    model: The model that we are using.
  
  Returns:
    Tuple of style and content representation of original image.
  """
  # Load our images in
  style_img = _load_img(flags.FLAGS.style)
  content_img = _load_img(flags.FLAGS.content)
  style_img = tf.keras.applications.vgg19.preprocess_input(style_img * 255)
  content_img = tf.keras.applications.vgg19.preprocess_input(content_img * 255)

  style_outputs = model(style_img)
  content_outputs = model(content_img)

  # Get the style and content feature representations from our model
  style_reprs = [
      style_layer for style_layer in style_outputs[:NUM_STYLE_LAYERS]
  ]
  content_reprs = [
      content_layer for content_layer in content_outputs[NUM_STYLE_LAYERS:]
  ]

  return style_reprs, content_reprs


def main(_):
  """Main function to run neural style"""
  run(flags.FLAGS)


if __name__ == '__main__':
  define_flags()
  app.run(main)
