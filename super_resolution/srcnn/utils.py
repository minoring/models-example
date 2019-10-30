"""Utils for srcnn implementation"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image
from absl import flags
import scipy.misc
import scipy.ndimage
import numpy as np
import tensorflow as tf
import imageio


def create_sub_images():
  """Read image files, make their sub-images and save them as a h5 file format.
  """
  dataset_paths = find_dataset_paths(flags.FLAGS.is_train)

  sub_inputs = []
  sub_labels = []

  scale = flags.FLAGS.scale

  if flags.FLAGS.is_train:
    for i in range(len(dataset_paths)):
      # Prepare image one by one.
      img = imread(dataset_paths[i])
      input_img, label = create_input_label(img, scale=scale)

      if len(input_img.shape) == 3:
        h, w, _ = input_img.shape
      else:
        h, w = input_img.shape

      # Calculate paddings.  |33 - 21|/2 = 6 when setting was default.
      padding = abs(flags.FLAGS.image_size - flags.FLAGS.label_size) // 2
      image_size = flags.FLAGS.image_size
      label_size = flags.FLAGS.label_size
      stride = flags.FLAGS.stride
      for x in range(0, h - image_size + 1, stride):
        for y in range(0, w - image_size + 1, stride): #TODO: 이거 뭐하는건지 이해하기
          sub_input = input_img[x:x + image_size, y:y + image_size]  # [33 x 33]
          sub_label = label[x + padding:x + padding + label_size, y +
                            padding:y + padding + label_size]

          # Make sure image has one color channel.
          # Create if it does not exists.    
          sub_input = sub_input.reshape([image_size, image_size, 1])
          sub_label = sub_label.reshape([label_size, label_size, 1])

          sub_inputs.append(sub_input)
          sub_labels.append(sub_label)
        

  else:

    pass


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
    data_dir = os.path.join(os.getcwd(), 'Train')
    dataset_paths = glob.glob(os.path.join(data_dir, '*.bmp'))
  else:
    data_dir = os.path.join(os.getcwd(), (os.path.join(os.getcwd(), 'Test')),
                            'Set5')
    dataset_paths = glob.glob(os.path.join(data_dir, '*.bmp'))

  return dataset_paths


def create_input_label(img, scale=3):
  """Create preprocessed input and label using give image.
  
  Preprocessing follows steps below,
  (1) Normalize to have 0-1 range
  (2) Upscale it to the desired size using bicubic interpolation

  Args:
    img: Numpy array of image.
    scale: Desired upscale size.
  
  Returns:
    A tuple of (input_img, label)
  """
  label = modcrop(img, scale)

  # Normalize.
  img = img / 255.
  label = label / 255.

  input_img = scipy.ndimage.interpolation.zoom(label, (1. / scale),
                                               prefilter=False)
  input_img = scipy.ndimage.interpolation.zoom(input_img, (scale / 1.),
                                               prefilter=False)
  return input_img, label


def modcrop(img, scale=3):
  """To scale down and up the original image, crop remainder part of image 

  Args:
    img: Image we want to crop
    scale: Desired upscale of downscale size. 

  Returns:
    Image that croped remainders.
  """
  if len(img.shape) == 3:
    h, w, _ = img.shape
  else:
    h, w = img.shape

  new_h = h - np.mod(h, scale)
  new_w = w - np.mod(w, scale)

  return img[0:new_h, 0:new_w, :] if len(
      img.shape == 3) else img[0:new_h, 0:new_w]


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
  return imageio.imread(path, as_gray=True, pilmode='YCbCr').astype(np.float)


def read_data_h5(file_path):
  """Read data from h5-format file in the path
  
  Args:
    file_path: file path containing data.
  
  Returns:
    A tuple of Numpy array containing (data, label).
  """
  with h5py.File(file_path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
  return data, label
