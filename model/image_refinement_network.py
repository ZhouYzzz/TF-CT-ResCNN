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


def model_v2(inputs, training, **conv_args):
  shortcut_0 = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (5, 5), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu1')
  shortcut_1 = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (5, 5), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu2')
  inputs = conv2d_fixed_padding(inputs, 64, (5, 5), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu3')
  inputs += shortcut_1
  shortcut_2 = inputs
  inputs = conv2d_fixed_padding(inputs, 64, (5, 5), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu4')
  inputs = conv2d_fixed_padding(inputs, 64, (5, 5), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu5')
  inputs += shortcut_2
  inputs = conv2d_fixed_padding(inputs, 16, (5, 5), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu6')
  inputs = conv2d_fixed_padding(inputs, 1, (5, 5), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training)
  inputs = tf.identity(inputs, 'relu7')
  inputs += shortcut_0
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def image_refinement_network(inputs, training):
  return model(inputs, training, kernel_initializer=tf.contrib.layers.xavier_initializer())


def image_refinement_network_v2(inputs, training):
  return model_v2(inputs, training, kernel_initializer=tf.contrib.layers.xavier_initializer())


def image_refinement_network_v3(inputs, training):
  sc = conv2d_fixed_padding(inputs, 1, (5, 5), kernel_initializer=tf.zeros_initializer())
  return inputs + sc


def image_refinement_network_v4(inputs, training):
  """Used in the denosing paper"""
  conv_args = {

  }
  inputs = conv2d_fixed_padding(inputs, 32, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  inputs = conv2d_fixed_padding(inputs, 32, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  inputs = conv2d_fixed_padding(inputs, 32, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  inputs = conv2d_fixed_padding(inputs, 32, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  inputs = conv2d_fixed_padding(inputs, 32, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  inputs = conv2d_fixed_padding(inputs, 32, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  inputs = conv2d_fixed_padding(inputs, 32, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  inputs = conv2d_fixed_padding(inputs, 1, kernel_size=(3, 3), strides=(1, 1), **conv_args)
  inputs = batch_norm_relu(inputs, training=training, data_format='channels_first')
  return inputs


def conv2d(inputs, filters, kernel_size, strides, **kwargs):
  return tf.layers.conv2d(inputs=inputs,
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          # padding='same',
                          data_format='channels_first',
                          **kwargs)


def conv2d_transpose(inputs, filters, kernel_size, strides, **kwargs):
  return tf.layers.conv2d_transpose(inputs=inputs,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    # padding='same',
                                    data_format='channels_first',
                                    **kwargs)


def image_refinement_network_v5(inputs, training=False):
  """Implement the transform network in arXiv 1706.09:
  Perceptual Adversarial Networks for Image-to-Image Transformation"""
  shared_args = {
    'kernel_initializer': tf.contrib.layers.xavier_initializer()
  }
  conv_args = {
    'padding': 'same',
    **shared_args
  }
  conv_args_2 = {
    'padding': 'valid',
    **shared_args
  }
  raw = inputs
  inputs = conv2d(inputs, filters=16, kernel_size=(3, 3), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 1
  # print(inputs)
  inputs = conv2d(inputs, filters=32, kernel_size=(3, 3), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 2
  # print(inputs)
  shortcut_2 = inputs
  inputs = conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(2, 2), **conv_args_2)
  inputs = batch_norm_relu(inputs, training=training)  # 3
  # print(inputs)
  shortcut_3 = inputs
  inputs = conv2d(inputs, filters=128, kernel_size=(3, 3), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 4
  # print(inputs)
  shortcut_4 = inputs
  inputs = conv2d(inputs, filters=128, kernel_size=(3, 3), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 5
  # print(inputs)
  shortcut_5 = inputs
  inputs = conv2d(inputs, filters=128, kernel_size=(3, 3), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 6
  # print(inputs)
  inputs = conv2d_transpose(inputs, filters=128, kernel_size=(4, 4), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 7
  # print(inputs)
  inputs = tf.concat([inputs, shortcut_5], axis=1)
  inputs = conv2d_transpose(inputs, filters=64, kernel_size=(4, 4), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 8
  # print(inputs)
  inputs = tf.concat([inputs, shortcut_4], axis=1)
  inputs = conv2d_transpose(inputs, filters=32, kernel_size=(4, 4), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 9
  # print(inputs)
  inputs = tf.concat([inputs, shortcut_3], axis=1)
  inputs = conv2d_transpose(inputs, filters=16, kernel_size=(4, 4), strides=(2, 2), **conv_args_2)
  inputs = batch_norm_relu(inputs, training=training)  # 10
  # print(inputs)
  inputs = tf.concat([inputs, shortcut_2], axis=1)
  inputs = conv2d_transpose(inputs, filters=16, kernel_size=(4, 4), strides=(2, 2), **conv_args)
  inputs = batch_norm_relu(inputs, training=training)  # 11
  # print(inputs)
  inputs = conv2d_transpose(inputs, filters=1, kernel_size=(4, 4), strides=(2, 2), **conv_args)
  # inputs = batch_norm_relu(inputs, training=training)  # 12
  # print(inputs)
  inputs += raw
  return inputs


if __name__ == '__main__':
  x = tf.placeholder(tf.float32, shape=(None, 1, 200, 200))
  outputs = image_refinement_network_v5(inputs=x)
