"""Functions to define and load model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Content layer where will pull our feature maps
CONTENT_LAYERS = ['block5_conv2']
# Style layer we are interested in
STYLE_LAYERS = [
    'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1',
    'block5_conv1'
]

NUM_STYLE_LAYERS = len(STYLE_LAYERS)
NUM_CONTENT_LAYERS = len(CONTENT_LAYERS)
STYLE_WEIGHT = 1e-2
CONTENT_WEIGHT = 1e4  # TODO(): Flag로.
WEIGHT_PER_STYLE_LAYER = 1.0 / float(NUM_STYLE_LAYERS)
WEIGHT_PER_CONTENT_LAYER = 1.0 / float(NUM_CONTENT_LAYERS)
NORM_MEANS = np.array([103.939, 116.779, 123.68])  # VGG19 normalization mean.


def neural_style():
  """Create out model with access to intermediate layers.

  This function will load the VGG19 model and access the intermediate layers.
  These layers will then be used to create a new model that will take input
  image and return the outputs from these intermediate layers from the VGG model.

  Returns:
    A keras model that take image inputs and outputs the style and
    content intermediate layers.
  """
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers
  style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
  content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]
  model_outputs = style_outputs + content_outputs
  # Build model
  return tf.keras.Model([vgg.input], model_outputs)


def compute_losses(model, gen_image, org_style_reprs, org_content_reprs):
  """This function will compute the total loss.

  Args:
    model: The model that will give us access to the intermediate layers
    gen_image: Image we are generating. This is what we are updating with our 
      optimization process. We apply the gradients the loss wrt this image.
    org_style_reprs: Precomputed gram metrices corresponding to the defined
      style layers of interest.
    org_content_reprs: Precomputed output from defined content layers 
        of interest.
  
  Returns:
    Dictionary of {'total_loss', 'style_loss', 'content_loss'}
  """
  # Feed our generating image through our model. This will give us the content
  # and style representations at our desired layers. Since we're using
  # eager execution, our model is callable just like any other function
  model_outputs = model(
      tf.keras.applications.vgg19.preprocess_input(gen_image * 255))

  gen_style_reprs = model_outputs[:NUM_STYLE_LAYERS]
  gen_content_reprs = model_outputs[NUM_STYLE_LAYERS:]

  style_loss = 0
  content_loss = 0

  # Accumulate style losses from all layers
  for org_style_repr, gen_style_repr in zip(org_style_reprs, gen_style_reprs):
    style_loss += WEIGHT_PER_STYLE_LAYER * compute_style_loss(
        org_style_repr, gen_style_repr)

  # Accumulate content losses from all layers
  for org_content_repr, gen_content_repr in zip(org_content_reprs,
                                                gen_content_reprs):
    content_loss += WEIGHT_PER_CONTENT_LAYER * compute_content_loss(
        org_content_repr, gen_content_repr)

  # Get total loss
  total_loss = STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss
  return dict(total_loss=total_loss,
              style_loss=style_loss,
              content_loss=content_loss)


def compute_content_loss(org_content_repr, gen_content_repr):
  """Compute content loss by calculating Euclidian distance between 
     intermediate content representation of original image and 
     image that is generated.

  Args:
    org_content_repr: Output of intermediate feature activation when 
      the input was original content image.
    gen_content_repr: Output of intermediate feature activation when 
      the input was image that is generated.

  Returns:
    Loss between feature representation of original image and generated image.
  """
  return tf.reduce_mean(tf.square(org_content_repr - gen_content_repr))


def compute_style_loss(org_style_repr, gen_style_repr):
  """Compute style loss between style representation of original image and
     generated image

  Args:
    org_style_repr: Tensor shape of (h, w, c). 
      Style representation of original image.
    gen_style_repr: Tensor shape of (h, w, c). 
      Style representation of generated image.
  
  Returns:
    Style loss.
  """
  gram_original = _gram_matrix(org_style_repr)
  gram_generated = _gram_matrix(gen_style_repr)

  # Why mean?
  return tf.reduce_mean(tf.square(gram_original - gram_generated))


def _gram_matrix(input_tensor):
  """Calculate gram matrix which is (channel, channel) shape.
  
  Args:
    input_tensor: tensor that is calculated.

  Returns:
    Gram matrix of input tensor.
  """
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
  return result / (num_locations)
