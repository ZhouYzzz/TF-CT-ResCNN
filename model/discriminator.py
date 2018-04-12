import tensorflow as tf
from model.image_refinement_network import batch_norm_relu
from resnet.cifar10_main import Cifar10Model


def discriminator_v1(inputs, training=False, **conv_args):
  """Implementation of the discriminator used in paper
  `Low Dose CT Image Denoising Using a Generative Adversarial Network
  with Wasserstein Distance and Perceptual Loss`
  """
  inputs = tf.layers.conv2d(inputs, 32, (3, 3), (1, 1), padding='same', **conv_args)
  # inputs = tf.layers.batch_normalization(inputs, axis=1)
  inputs = batch_norm_relu(inputs, training=training)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 32, (3, 3), (2, 2), padding='same', **conv_args)
  inputs = batch_norm_relu(inputs, training=training)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 64, (3, 3), (1, 1), padding='same', **conv_args)
  inputs = batch_norm_relu(inputs, training=training)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 64, (3, 3), (2, 2), padding='same', **conv_args)
  inputs = batch_norm_relu(inputs, training=training)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 128, (3, 3), (1, 1), padding='same', **conv_args)
  inputs = batch_norm_relu(inputs, training=training)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 128, (3, 3), (2, 2), padding='same', **conv_args)
  inputs = batch_norm_relu(inputs, training=training)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 128, (3, 3), (2, 2), padding='same', **conv_args)
  inputs = batch_norm_relu(inputs, training=training)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.flatten(inputs)
  inputs = tf.layers.dense(inputs, 2048)
  # inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.dense(inputs, 1)
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def discriminator_v2(inputs, training=False):
  model = Cifar10Model(resnet_size=8, data_format='channels_first', num_classes=1)
  return model(inputs, training=training)


def discriminator(inputs, training=False):
  return discriminator_v1(inputs,
                          training=training,
                          data_format='channels_first',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                          bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4))


if __name__ == '__main__':
  x = tf.placeholder(tf.float32, shape=(None, 1, 64, 64))
  outputs = discriminator(x, training=True)
  print(outputs)
