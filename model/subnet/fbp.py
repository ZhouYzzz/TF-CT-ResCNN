"""
The FBP subnet
"""
import tensorflow as tf
import dataset.info as info


def _load_weights():
  with tf.variable_scope('load_weights'):
    W = tf.decode_raw(tf.read_file('W.bin'), tf.float64)
    W = tf.cast(W, tf.float32)
    W = tf.reshape(W, shape=(info.PRJ_WIDTH, info.PRJ_HEIGHT))
    W = tf.transpose(W)
    F = tf.decode_raw(tf.read_file('F.bin'), tf.float64)
    F = tf.cast(F, tf.float32)
    F = tf.reshape(F, shape=(info.PRJ_HEIGHT, info.PRJ_HEIGHT))
    F = tf.transpose(F)
    Hi = tf.decode_raw(tf.read_file('H_indices.bin'), tf.int64)
    Hv = tf.decode_raw(tf.read_file('H_values.bin'), tf.float64)
    Hv = tf.cast(Hv, tf.float32)
    Hv = tf.reshape(Hv, shape=(-1,2))
  W = tf.Variable(initial_value=W, trainable=False)
  F = tf.Variable(initial_value=F, trainable=False)
  Hi = tf.Variable(initial_value=Hi, trainable=False)
  Hv = tf.Variable(initial_value=Hv, trainable=False)
  H = tf.SparseTensor(Hi, Hv, dense_shape=[info.IMG_HEIGHT * info.IMG_WIDTH, info.PRJ_HEIGHT * info.PRJ_WIDTH])
  return H, F, W


def fbp_subnet(inputs):
  H, F, W = _load_weights()
  inputs = tf.reshape(inputs, shape=(-1, info.PRJ_HEIGHT, info.PRJ_WIDTH))
  inputs = tf.map_fn(lambda x: tf.multiply(W, x), inputs)             # WP
  inputs = tf.map_fn(lambda x: tf.matmul(F, x), inputs)               # FWP
  inputs = tf.transpose(inputs, perm=(0, 2, 1))
  inputs = tf.layers.flatten(inputs)
  inputs = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)   # HFWP
  inputs = tf.transpose(inputs)
  inputs = tf.reshape(inputs, shape=(-1, info.IMG_DEPTH, info.IMG_HEIGHT, info.IMG_WIDTH))
  inputs = tf.transpose(inputs, perm=(0, 1, 3, 2))
  inputs = tf.identity(inputs, 'outputs')
  return inputs