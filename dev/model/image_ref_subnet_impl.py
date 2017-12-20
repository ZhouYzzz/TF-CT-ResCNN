import tensorflow as tf
from model_common import _BATCH_NORM_DECAY, _BATCH_NORM_EPSILON


def batch_norm_relu(inputs, is_training):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  wpad_total = kernel_size[1] - 1
  wpad_beg = wpad_total // 2
  wpad_end = wpad_total - wpad_beg
  hpad_total = kernel_size[0] - 1
  hpad_beg = hpad_total // 2
  hpad_end = hpad_total - hpad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                  [hpad_beg, hpad_end], [wpad_beg, wpad_end]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, transpose=False, padding=True):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if ((strides > 1) and padding):
    inputs = fixed_padding(inputs, kernel_size)
  conv2d_fn = tf.layers.conv2d_transpose if transpose else tf.layers.conv2d
  inputs = conv2d_fn(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('VALID' if padding else 'SAME'), use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      data_format='channels_first')
  return inputs


def image_ref_subnet_core_v0(inputs, is_training):
  #print('Using image_ref_subnet_core_v0')
  inputs = conv2d_fixed_padding(inputs, 64, (7,7), (1,1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(inputs, 64, (7,7), (1,1))
  inputs = batch_norm_relu(inputs, is_training)
  shortcuts = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (7,7), (1,1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs += shortcuts
  inputs = conv2d_fixed_padding(inputs, 64, (7,7), (1,1))
  inputs = batch_norm_relu(inputs, is_training)
  shortcuts = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (7,7), (1,1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs += shortcuts
  inputs = conv2d_fixed_padding(inputs, 16, (7,7), (1,1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(inputs, 1, (7,7), (1,1))
  return inputs

def image_ref_subnet_core_v1(inputs, is_training):
  inputs = conv2d_fixed_padding(inputs, 64, (5,5), (1,1)) # 1/1
  inputs = batch_norm_relu(inputs, is_training)
  shortcuts_0 = inputs
  inputs = conv2d_fixed_padding(inputs, 128, (3,3), (2,2)) # 1/2
  inputs = batch_norm_relu(inputs, is_training)
  shortcuts_1 = inputs
  inputs = conv2d_fixed_padding(inputs, 256, (3,3), (2,2)) # 1/4
  inputs = batch_norm_relu(inputs, is_training)
  shortcuts_2 = inputs
  inputs = conv2d_fixed_padding(inputs, 512, (3,3), (2,2)) # 1/8
  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(inputs, 512, (3,3), (1,1)) # 1/8
  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(inputs, 256, (3,3), (2,2), transpose=True, padding=False) # 1/4
  inputs = batch_norm_relu(inputs, is_training)
  inputs += shortcuts_2
  inputs = conv2d_fixed_padding(inputs, 128, (3,3), (2,2), transpose=True, padding=False) # 1/2
  inputs = batch_norm_relu(inputs, is_training)
  inputs += shortcuts_1
  inputs = conv2d_fixed_padding(inputs, 64, (3,3), (2,2), transpose=True, padding=False) # 1/1
  inputs = batch_norm_relu(inputs, is_training)
  inputs += shortcuts_0
  inputs = conv2d_fixed_padding(inputs, 1, (5,5), (1,1)) # 1/1
  inputs = tf.identity(inputs, 'outputs')
  return inputs
