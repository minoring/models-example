"""Flags which will be nearly universal across models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from official.utils.flags._conventions import help_wrap
from official.utils.logs import hooks_helper


def define_base(data_dir=True,
                model_dir=True,
                clean=False,
                train_epochs=False,
                epochs_between_evals=False,
                stop_threshold=False,
                batch_size=True,
                num_gpu=False,
                hooks=False,
                export_dir=False,
                distribution_strategy=False,
                run_eagerly=False):
    """Register base flags.

    Args:
        data_dir: Create a flag for specifying the input data directory.
        model_dir: Create a flag for specifying the model file directory.
        clean: Create a flag for removing the model_dir.
        train_epochs: Create a flag to specify the number of training epochs.
        epochs_between_evals: Create a flag to specify the frequency of testing.
        stop_threshold: Create a flag to specify a threshold accuracy of other
                        eval metric which should trigger the end of training.
        batch_size: Create a flag to specify the batch size.
        num_gpu: Create a flag to specify the number of GPUs used.
        hooks: Create a flag to specify hooks for logging.
        export_dir: Create a flag to specify where a SaveModel should be 
                    exported.
        distribution_strategy: Create a flag to specify which Distribution 
                               Strategy to use
        run_eagerly: Create a flag to specify to run eagerly op by op.
    Returns:
        A list of flags for core.py to marks as key flags.
    """
    key_flags = []

    if data_dir:
        flags.DEFINE_string(
            name="data_dir", short_name="dd", default="/tmp",
            help=help_wrap("The location of the input data."))
        key_flags.append("data_dir")

    if model_dir:
        flags.DEFINE_string(
            name="model_dir", short_name="md", default='/tmp',
            help=help_wrap('The location of the model checkpoint files.'))
        key_flags.append('model_dir')

    if clean:
        flags.DEFINE_boolean(
            name='clean', default=False,
            help=help_wrap("If set, model_dir will be removed if it exists."))
        key_flags.append('clean')
    
    if train_epochs:
        flags.DEFINE_integer(
            name='train_epochs', short_name='te', default=1,
            help=help_wrap('The number of epochs used to train.'))
        key_flags.append('train_epochs')
    
    if epochs_between_evals:
        flags.DEFINE_integer(
            name='epochs_between_evals', short_name='ebe', default=1,
            help=help_wrap('The number of training epochs to run between '
                           'evaluations.'))
        key_flags.append('epochs_between_evals')
    
    if stop_threshold:
        flags.DEFINE_float(
            name='stop_threshold', short_name='st', default=None,
            help=help_wrap('If passed, training will stop at the earlier of '
                           'train_epochs and when the evaluation metric is '
                           'greater than or equal to stop_threshold.'))
    
    