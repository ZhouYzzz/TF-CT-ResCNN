import tensorflow as tf
from model_common import _BATCH_NORM_DECAY, _BATCH_NORM_EPSILON


def _periodic_padding_along_angles(inputs, kernel_size):
  """Performs a periodic padding along angle(width) dim."""
  hpad_beg = (kernel_size[0] - 1) // 2
  hpad_end = kernel_size[0] - 1 - hpad_beg
  wpad_beg = (kernel_size[1] - 1) // 2
  wpad_end = kernel_size[1] - 1 - wpad_beg
  inputs = tf.concat([inputs[:,:,:,slice(-1,-1-wpad_beg,-1)], inputs, inputs[:,:,:,slice(0,wpad_end,1)]], axis=3)
  inputs = tf.pad(inputs, [[0, 0], [0, 0], [hpad_beg, hpad_end], [0, 0]])
  return inputs


def batch_norm_relu(inputs, is_training):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def conv2d_periodic_padding(inputs, filters, kernel_size, strides, transpose=False):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  conv2d_fn = tf.layers.conv2d_transpose if transpose else tf.layers.conv2d
  inputs = _periodic_padding_along_angles(inputs, kernel_size)
  inputs = conv2d_fn(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID', use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format='channels_first')
  return inputs


def prj_est_subnet_core_v0(inputs, is_training):
  print('Using prj_est_subnet_core_v0')
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), (1, 1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), (1, 1))
  inputs = batch_norm_relu(inputs, is_training)
  shortcuts = inputs
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), (1, 1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs += shortcuts
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), (1, 1))
  inputs = batch_norm_relu(inputs, is_training)
  shortcuts = inputs
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), (1, 1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs += shortcuts
  inputs = conv2d_periodic_padding(inputs, 16, (9, 3), (1, 1))
  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_periodic_padding(inputs, 1, (9, 3), (1, 1))
  return inputs


def prj_est_subnet_core_v1(inputs, is_training):
  print('Using prj_est_subnet_core_v1')
  return inputs


def main(_):
  inputs = tf.placeholder(dtype=tf.float32, shape=(None, 1, 216, 72))
  inputs = prj_est_subnet_core_v0(inputs)


if __name__ == '__main__':
  tf.app.run()
