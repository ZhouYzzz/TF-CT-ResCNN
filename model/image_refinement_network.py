import tensorflow as tf


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


################################################################################
# Functions building the ResNet model.
################################################################################
def batch_norm_relu(inputs, training, data_format='channels_first'):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)
  # inputs = tf.nn.relu(inputs)
  inputs = tf.nn.leaky_relu(inputs, alpha=0.1)
  return inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, **kwargs):
  """Strided 2-D convolution with explicit padding."""
  def _fixed_padding(inputs, kernel_size, data_format):
    hpad_beg = (kernel_size[0] - 1) // 2
    hpad_end = kernel_size[0] - 1 - hpad_beg
    wpad_beg = (kernel_size[1] - 1) // 2
    wpad_end = kernel_size[1] - 1 - wpad_beg

    if data_format == 'channels_first':
      padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                      [hpad_beg, hpad_end], [wpad_beg, wpad_end]])
    else:
      padded_inputs = tf.pad(inputs, [[0, 0], [hpad_beg, hpad_end],
                                      [wpad_beg, wpad_end], [0, 0]])
    return padded_inputs

  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  inputs = _fixed_padding(inputs, kernel_size, 'channels_first')
  inputs = tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size,
      padding='valid', data_format='channels_first', **kwargs)
  return inputs


def model(inputs, training, **conv_args):
  shortcut_0 = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (7, 7), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu1')
  shortcut_1 = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (7, 7), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu2')
  inputs = conv2d_fixed_padding(inputs, 64, (7, 7), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu3')
  inputs += shortcut_1
  shortcut_2 = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (7, 7), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu4')
  inputs = conv2d_fixed_padding(inputs, 64, (7, 7), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu5')
  inputs += shortcut_2
  inputs = conv2d_fixed_padding(inputs, 16, (7, 7), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu6')
  inputs = conv2d_fixed_padding(inputs, 1, (7, 7), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu7')
  inputs += shortcut_0
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def image_refinement_network(inputs, training):
  return model(inputs, training, kernel_initializer=tf.contrib.layers.xavier_initializer())
