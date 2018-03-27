import tensorflow as tf


def discriminator_v1(inputs, **conv_args):
  """Implementation of the discriminator used in paper
  `Low Dose CT Image Denoising Using a Generative Adversarial Network
  with Wasserstein Distance and Perceptual Loss`
  """
  inputs = tf.layers.conv2d(inputs, 64, (3, 3), (1, 1), padding='same', **conv_args)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 64, (3, 3), (2, 2), padding='same', **conv_args)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 128, (3, 3), (1, 1), padding='same', **conv_args)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 128, (3, 3), (2, 2), padding='same', **conv_args)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 256, (3, 3), (1, 1), padding='same', **conv_args)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 256, (3, 3), (2, 2), padding='same', **conv_args)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.flatten(inputs)
  inputs = tf.layers.dense(inputs, 1024)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.dense(inputs, 1)
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def discriminator(inputs, training=False):
  return discriminator_v1(inputs, data_format='channels_first', kernel_initializer=tf.random_normal_initializer(stddev=1e-3))


if __name__ == '__main__':
  x = tf.placeholder(tf.float32, shape=(None, 1, 64, 64))
  outputs = discriminator(x, training=True)
  print(outputs)
