"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
from six.moves import range
import tensorflow as tf # pylint: disable=g-bad-import-order

from official.mnist import dataset
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


LEARNING_RATE = 1e-4


def create_model(data_format):
  """Model to recognize digits in the MNIST dataset.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But uses the tf.keras API.

  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See

       https://www.tensorflow.org/performance/performance_guide#data_formats

  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]
  
  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation='relu'),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation='relu'),
          l.Dropout(0.4),
          l.Dense(10)
      ])

  
def define_mnist_flags():
  """Defines flags for mnist."""
  flags_core.define_base(clean=True, train_epochs=True,
                         epoch_between_evals=True, stop_threshold=True,
                         num_gpu=True, hooks=True, export_dir=True,
                         distribution_strategy=True)
  flags_core.define_performance(inter_op=True, intra_op=True,
                                num_parallel_calls=False,
                                all_reduce_alg=True)
  flags_core.define_image()
  flags.adopt_module_key_flags(flags_core)
  flags_core.set_defaults(data_dir='mnist_data/tmp',
                          model_dir='mnist_data/tmp',
                          batch_size=100,
                          train_epochs=40)


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = create_model(params['data_format'])
  image = features
  if isinstance(image, dict):
    image = features['image']
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.Modekeys.PREDICT,
        predictions=predictions,
        export_outputs={
          'classify': tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.Modekeys.TRAIN:
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    logits = model(image, training=True)
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels,
                                                            logits=logits)
    accuracy = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    
    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(LEARNING_RATE, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(
            loss,
            tf.compat.v1.train.get_or_create_global_step()))
  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, 
                                                            logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.compat.v1.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        })


def run_mnist(flags_obj):
  """Run MNIST training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  model_helpers.apply_clean(flags_obj)
  model_function = model_fn

  session_config = tf.compat.v1.ConfigProto(
      inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
      allow_soft_placement=True)