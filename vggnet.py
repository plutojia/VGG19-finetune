"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np


class VGG19Net(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'vgg19.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""

        conv1_1 = conv(self.X, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, padding='SAME', name='pool1')

        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, padding='SAME', name='pool2')

        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        conv3_4 = conv(conv3_3, 3, 3, 256, 1, 1, padding='SAME', name='conv3_4')
        pool3 = max_pool(conv3_4, 2, 2, 2, 2, padding='SAME', name='pool3')

        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        conv4_4 = conv(conv4_3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_4')
        pool4 = max_pool(conv4_4, 2, 2, 2, 2, padding='SAME', name='pool4')

        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        conv5_4 = conv(conv5_3, 3, 3, 512, 1, 1, padding='SAME', name='conv5_4')
        pool5 = max_pool(conv5_4, 2, 2, 2, 2, padding='SAME', name='pool5')

        flattened = tf.reshape(pool5, [-1, 7 * 7 * 512])
        fc6 = fc(flattened, 7 * 7 * 512, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')


    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels / groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
