import tensorflow as tf
import numpy as np
import os


tf.flags.DEFINE_string('fbp_data_dir', './data', '')
FLAGS = tf.flags.FLAGS


def _load_weights():
  W = tf.constant(np.fromfile(os.path.join(FLAGS.fbp_data_dir, 'W.bin'), np.float64).astype(np.float32),
                  dtype=tf.float32, shape=(360, 216))
  W = tf.transpose(W)
  F = tf.constant(np.fromfile(os.path.join(FLAGS.fbp_data_dir, 'F.bin'), np.float64).astype(np.float32),
                  dtype=tf.float32, shape=(216, 216))
  F = tf.transpose(F)  # actually this does nothing, since F is symmetric
  Hi = tf.constant(np.fromfile(os.path.join(FLAGS.fbp_data_dir, 'H_indices.bin'), np.int64).reshape(-1, 2),
                   dtype=tf.int64)
  Hv = tf.constant(np.fromfile(os.path.join(FLAGS.fbp_data_dir, 'H_values.bin'), np.float64).astype(np.float32),
                   dtype=tf.float32)
  H = tf.SparseTensor(Hi, Hv, dense_shape=[200 * 200, 216 * 360])
  return H, F, W


def fbp_subnet(inputs):
  H, F, W = _load_weights()
  inputs = tf.reshape(inputs, shape=(-1, 216, 360))
  inputs = tf.map_fn(lambda x: tf.multiply(W, x), inputs)             # WP
  inputs = tf.map_fn(lambda x: tf.matmul(F, x), inputs)               # FWP
  inputs = tf.transpose(inputs, perm=(0, 2, 1))
  inputs = tf.layers.flatten(inputs)
  inputs = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)   # HFWP
  inputs = tf.transpose(inputs)
  inputs = tf.reshape(inputs, shape=(-1, 1, 200, 200))
  inputs = tf.transpose(inputs, perm=(0, 1, 3, 2))
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def main(_):
  inputs = tf.placeholder(tf.float32, shape=(None, 1, 216, 360))
  inputs = fbp_subnet(inputs)


if __name__ == '__main__':
  tf.app.run()
