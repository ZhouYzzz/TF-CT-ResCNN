import tensorflow as tf
import dataset.info as info


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def _periodic_padding_along_angles(inputs, kernel_size):
  """Performs a periodic padding along angle(width) dim."""
  hpad_beg = (kernel_size[0] - 1) // 2
  hpad_end = kernel_size[0] - 1 - hpad_beg
  wpad_beg = (kernel_size[1] - 1) // 2
  wpad_end = kernel_size[1] - 1 - wpad_beg
  inputs = tf.concat([inputs[:,:,:,-1:(-1-wpad_beg):-1], inputs, inputs[:,:,:,0:wpad_end]], axis=3)
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


def conv2d_periodic_padding(inputs, filters, kernel_size, strides, transpose=False, padding=True):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  conv2d_fn = tf.layers.conv2d_transpose if transpose else tf.layers.conv2d
  if padding:
    inputs = _periodic_padding_along_angles(inputs, kernel_size)
  inputs = conv2d_fn(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID' if padding else 'SAME', use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format='channels_first')
  return inputs


def slice_concat(inputs_list, axis):
  sliced_inputs = [None for _ in range(info.PRJ_WIDTH)]
  for i in range(info.PRJ_SPARSE_NUM):
    sliced_inputs[i::info.PRJ_SPARSE_NUM] = tf.split(inputs_list[i], info.PRJ_SPARSE_WIDTH, axis=axis)
  inputs = tf.concat(sliced_inputs, axis=axis)
  return inputs
