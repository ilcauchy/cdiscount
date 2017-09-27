from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features['x'],[-1, 180, 180, 3])

    # Conv 1
    conv1 = tf.layers.conv2d()


# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()