import tensorflow as tf
from model.projection_estimation_network import conv2d_periodic_padding, batch_norm_relu
from model.subnet.prj_est_impl import slice_concat


def model(inputs, training, i, **conv_args):
  with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
    shortcut_0 = inputs
    inputs = conv2d_periodic_padding(inputs, 64, (5, 3), strides=(1, 1), **conv_args)
    inputs = batch_norm_relu(inputs, training)
    inputs = conv2d_periodic_padding(inputs, 64, (5, 3), strides=(1, 1), **conv_args)
    inputs = batch_norm_relu(inputs, training)
    inputs = conv2d_periodic_padding(inputs, 64, (5, 3), strides=(1, 1), **conv_args)
    inputs = batch_norm_relu(inputs, training)
    shortcut_1 = inputs
  with tf.variable_scope('B{}'.format(i)):
    inputs = conv2d_periodic_padding(inputs, 64, (5, 3), strides=(1, 1), **conv_args)
    inputs = batch_norm_relu(inputs, training)
    inputs = conv2d_periodic_padding(inputs, 64, (5, 3), strides=(1, 1), **conv_args)
    inputs = batch_norm_relu(inputs, training)
    inputs += shortcut_1
    inputs = conv2d_periodic_padding(inputs, 16, (5, 3), strides=(1, 1), **conv_args)
    inputs = batch_norm_relu(inputs, training)
    inputs = conv2d_periodic_padding(inputs, 1, (5, 3), strides=(1, 1), **conv_args)
    inputs = batch_norm_relu(inputs, training)
    inputs += shortcut_0
    inputs = tf.nn.leaky_relu(inputs)
  return inputs


def projection_estimation_network_v2(inputs, training):
  projection_outputs = []
  for i in range(4):
    branch_outputs = model(inputs, training=training, i=i,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    projection_outputs.append(branch_outputs)
  projection_outputs.insert(2, inputs)
  projection_outputs = slice_concat(projection_outputs, axis=3)
  return projection_outputs
