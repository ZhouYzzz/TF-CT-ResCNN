import tensorflow as tf


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def conv2d_periodic_padding(inputs, filters, kernel_size, **kwargs):
  def _periodic_padding_along_angles(inputs, kernel_size):
    """Performs a periodic padding along angle(width) dim."""
    hpad_beg = (kernel_size[0] - 1) // 2
    hpad_end = kernel_size[0] - 1 - hpad_beg
    wpad_beg = (kernel_size[1] - 1) // 2
    wpad_end = kernel_size[1] - 1 - wpad_beg
    inputs = tf.concat([inputs[:, :, :, -1:(-1-wpad_beg):-1], inputs, inputs[:, :, :, 0:wpad_end]], axis=3)
    inputs = tf.pad(inputs, [[0, 0], [0, 0], [hpad_beg, hpad_end], [0, 0]])
    return inputs
  inputs = _periodic_padding_along_angles(inputs, kernel_size=kernel_size)
  inputs = tf.layers.conv2d(inputs, filters, kernel_size, padding='valid', data_format='channels_first', **kwargs)
  return inputs


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
  inputs = tf.nn.leaky_relu(inputs, alpha=0.1) # use leaky relu instead
  return inputs


# def conv2d(inputs,
#          filters,
#          kernel_size,
#          strides=(1, 1),
#          padding='valid',
#          data_format='channels_last',
#          dilation_rate=(1, 1),
#          activation=None,
#          use_bias=True,
#          kernel_initializer=None,
#          bias_initializer=init_ops.zeros_initializer(),
#          kernel_regularizer=None,
#          bias_regularizer=None,
#          activity_regularizer=None,
#          kernel_constraint=None,
#          bias_constraint=None,
#          trainable=True,
#          name=None,
#          reuse=None):
def model(inputs, training, **conv_args):
  shortcut_0 = inputs
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu1')
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu2')
  shortcut_1 = inputs
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu3')
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu4')
  inputs += shortcut_1
  inputs = conv2d_periodic_padding(inputs, 64, (9, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu5')
  inputs = conv2d_periodic_padding(inputs, 16, (9, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu6')
  inputs = conv2d_periodic_padding(inputs, 1, (9, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu7')
  inputs += shortcut_0
  pass


def projection_estimation_network(inputs, training):
  return model(inputs, training, kernel_initializer=tf.contrib.layers.xavier_initializer())
