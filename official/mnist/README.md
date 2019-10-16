# MNIST in TensorFlow

This directory builds a convolutional neural net to classify the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/) using the
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
and
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)
APIs.


## Setup

To begin, you'll simply need the latest version of TensorFlow installed.
First make sure you've [added the models folder to your Python path]:

```shell
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

Otherwise you may encounter an error like `ImportError: No module named official.mnist`.

Then to train the model, run the following:

```
python mnist.py
```

The model will begin training and will automatically evaluate itself on the
validation data.