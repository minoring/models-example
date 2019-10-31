"""Utils for srcnn implementation

We only consider gray scale for now.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image
from absl import app
from absl import flags
import scipy.misc
import imageio
import scipy.ndimage
import numpy as np
import tensorflow as tf

from srcnn.flags import define_flags

# Global constants
RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
SUB_IMAGES_PATH = os.path.join(RESOURCES_PATH, 'sub_image.h5')


def create_sub_images():
  """Read image files, make their sub-images and save them as a h5 file format.
  """
  dataset_paths = find_dataset_paths(flags.FLAGS.is_train)

  sub_inputs = []
  sub_labels = []

  # Calculate paddings.  |33 - 21|/2 = 6 when setting was default.
  padding = abs(flags.FLAGS.image_size - flags.FLAGS.label_size) // 2
  scale = flags.FLAGS.scale
  image_size = flags.FLAGS.image_size
  label_size = flags.FLAGS.label_size
  stride = flags.FLAGS.stride

  for dataset_path in dataset_paths:
    # Prepare image one by one.
    img = imread(dataset_path)
    input_img, label = create_input_label(img, scale=scale)
    h, w = input_img.shape

    for y in range(0, h - image_size + 1, stride):
      for x in range(0, w - image_size + 1, stride):
        sub_input = input_img[y:y + image_size, x:x + image_size]  # [33 x 33]
        sub_label = label[y + padding:y + padding + label_size, x + padding:x +
                          padding + label_size]

        # Make channel.
        sub_input = sub_input.reshape([image_size, image_size, 1])
        sub_label = sub_label.reshape([label_size, label_size, 1])

        sub_inputs.append(sub_input)
        sub_labels.append(sub_label)

  save_as_h5_file(sub_inputs, sub_labels)


def sub_images_exists():
  """Return False for now."""
  return False 


def save_as_h5_file(sub_inputs, sub_labels):
  """Save sub images as h5 file format."""
  if not os.path.exists(RESOURCES_PATH):
    os.mkdir(RESOURCES_PATH)

  with h5py.File(SUB_IMAGES_PATH, 'w') as hf:
    hf.create_dataset('input_img', data=sub_inputs)
    hf.create_dataset('label', data=sub_labels)


def find_dataset_paths(is_train):
  """Find list of dataset paths.

  Args:
    is_train: boolean. 
      True if we want to load Training set, False if we we want Test set.

  Returns:
    List of dataset_paths
  """
  if is_train:
    # Find list in 'Train' directory
    data_dir = os.path.join(os.getcwd(), 'resources/train')
    dataset_paths = glob.glob(os.path.join(data_dir, 't1.bmp')) #TODO *.bmp for production
  else:
    data_dir = os.path.join(os.getcwd(),
                            (os.path.join(os.getcwd(), 'resources/test')),
                            'Set5')
    dataset_paths = glob.glob(os.path.join(data_dir, '*.bmp'))

  return dataset_paths


def create_input_label(img, scale=3):
  """Create preprocessed input and label of given image.
  
  Preprocessing follows steps below,
  (1) Normalize to have 0-1 range
  (2) Upscale it to the desired size using bicubic interpolation

  Args:
    img: Numpy array of image.
    scale: Desired upscale size.
  
  Returns:
    A tuple of (input_img, label)
  """
  label = _modcrop(img, scale)

  # Normalize.
  label = label / 255.

  input_img = scipy.ndimage.interpolation.zoom(label, (1. / scale),
                                               prefilter=False)
  input_img = scipy.ndimage.interpolation.zoom(input_img, (scale / 1.),
                                               prefilter=False)
  return input_img, label


def _modcrop(img, scale=3):
  """To scale down and up the original image, crop remainder part of image 

  Args:
    img: Image we want to crop
    scale: Upscale or downscale size. 

  Returns:
    Image that croped remainders.
  """
  h, w = img.shape
  new_h = h - np.mod(h, scale)
  new_w = w - np.mod(w, scale)

  return img[0:new_h, 0:new_w]


def imread(path, as_gray=True):
  """Read image frm its path.

  Default value is gray-scale, and image is read by YCbCr format as the paper 
    suggested.
  
  Args:
    path: Path to image.
    as_gray: Boolean to indicate read image as gray scale.
  
  Returns:
    Numpy array of image
  """
  img = imageio.imread(path, as_gray=as_gray, pilmode='YCbCr').astype(np.float)
  if len(img.shape) == 3:
    # Make sure image shape (h, w)
    img = img[:, :, 0]
  return img


def read_data_h5(file_path):
  """Read data from h5-format file in the path
  
  Args:
    file_path: file path containing data.
  
  Returns:
    A tuple of Numpy array containing (input_img, label).
  """
  with h5py.File(file_path, 'r') as hf:
    input_imgs = np.array(hf.get('input_img'))
    labels = np.array(hf.get('label'))
  return input_imgs, labels


def main(_):
  define_flags()
  create_sub_images()
  input_img, label = read_data_h5(SUB_IMAGES_PATH)
  print("Shape of input image shape", input_img.shape)
  print("Shape of label image shape", label.shape)


if __name__ == '__main__':
  app.run(main)