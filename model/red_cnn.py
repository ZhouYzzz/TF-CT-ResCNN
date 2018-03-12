"""
Residual Encoder-Decoder Convolutional Neural Network (RED-CNN) implementation
"""
import tensorflow as tf


def red_cnn(inputs):
  sc0 = inputs
  # conv1
  inputs = tf.layers.conv2d(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = tf.nn.relu(inputs)
  # conv2
  inputs = tf.layers.conv2d(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = tf.nn.relu(inputs)
  sc1 = inputs
  # conv3
  inputs = tf.layers.conv2d(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = tf.nn.relu(inputs)
  # conv4
  inputs = tf.layers.conv2d(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = tf.nn.relu(inputs)
  sc2 = inputs
  # conv5
  inputs = tf.layers.conv2d(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = tf.nn.relu(inputs)
  # deconv1
  inputs = tf.layers.conv2d_transpose(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = inputs + sc2
  inputs = tf.nn.relu(inputs)
  # deconv2
  inputs = tf.layers.conv2d_transpose(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = tf.nn.relu(inputs)
  # deconv3
  inputs = tf.layers.conv2d_transpose(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = inputs + sc1
  inputs = tf.nn.relu(inputs)
  # deconv4
  inputs = tf.layers.conv2d_transpose(inputs, 96, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = tf.nn.relu(inputs)
  # deconv5
  inputs = tf.layers.conv2d_transpose(inputs, 1, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
  inputs = inputs + sc0
  inputs = tf.nn.relu(inputs)
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def main():
  inputs = tf.zeros(shape=(1, 200, 200, 1))
  inputs = red_cnn(inputs)
  print(inputs)


if __name__ == '__main__':
  main()
