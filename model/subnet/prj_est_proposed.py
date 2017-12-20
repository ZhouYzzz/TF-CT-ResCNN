import tensorflow as tf
from model.subnet.prj_est_impl import batch_norm_relu, conv2d_periodic_padding, slice_concat


def _prj_est_subnet_branch(inputs, index, is_training):
  with tf.variable_scope('Shared', reuse=tf.AUTO_REUSE):
    inputs = conv2d_periodic_padding(inputs, 64, (9,3), (1,1)) # 1/1
    inputs = batch_norm_relu(inputs, is_training)
    shortcuts_0 = inputs
    inputs = conv2d_periodic_padding(inputs, 128, (9,3), (2,2)) # 1/2
    inputs = batch_norm_relu(inputs, is_training)
    shortcuts_1 = inputs
    inputs = conv2d_periodic_padding(inputs, 256, (9,3), (2,2)) # 1/4
    inputs = batch_norm_relu(inputs, is_training)
    shortcuts_2 = inputs
    inputs = conv2d_periodic_padding(inputs, 512, (9,3), (2,2)) # 1/8
    inputs = batch_norm_relu(inputs, is_training)
  with tf.variable_scope('Branch{}'.format(index)):
    inputs = conv2d_periodic_padding(inputs, 512, (9,3), (1,1)) # 1/8
    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_periodic_padding(inputs, 256, (9,3), (2,2), transpose=True, padding=False) # 1/4
    inputs = batch_norm_relu(inputs, is_training)
    inputs += shortcuts_2
    inputs = conv2d_periodic_padding(inputs, 128, (9,3), (2,2), transpose=True, padding=False) # 1/2
    inputs = batch_norm_relu(inputs, is_training)
    inputs += shortcuts_1
    inputs = conv2d_periodic_padding(inputs, 64, (9,3), (2,2), transpose=True, padding=False) # 1/1
    inputs = batch_norm_relu(inputs, is_training)
    inputs += shortcuts_0
    inputs = conv2d_periodic_padding(inputs, 1, (9,3), (1,1)) # 1/1
    inputs = tf.identity(inputs, 'outputs')
  return inputs


def prj_est_subnet(inputs, is_training):
  branch_outputs = [None for _ in range(5)]
  for i in range(5):
    branch_outputs[i] = _prj_est_subnet_branch(inputs, i, is_training)
  inputs = slice_concat(branch_outputs, axis=3)
  inputs = tf.identity(inputs, 'outputs')
  return inputs
