"""
Evaluate the speed of different implementation of the FBP subnet
"""
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import dataset.info as info
from model.subnet.fbp import _load_weights


def fbp_subnet(inputs):
  """
  The default implementation
  """
  H, F, W = _load_weights()
  inputs = tf.reshape(inputs, shape=(-1, info.PRJ_HEIGHT, info.PRJ_WIDTH))
  inputs = tf.map_fn(lambda x: tf.multiply(W, x), inputs)  # WP
  inputs = tf.map_fn(lambda x: tf.matmul(F, x), inputs)  # FWP
  inputs = tf.transpose(inputs, perm=(0, 2, 1))
  inputs = tf.layers.flatten(inputs)
  inputs = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)  # HFWP
  inputs = tf.transpose(inputs)
  inputs = tf.reshape(inputs, shape=(-1, info.IMG_DEPTH, info.IMG_HEIGHT, info.IMG_WIDTH))
  inputs = tf.transpose(inputs, perm=(0, 1, 3, 2))
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def fbp_subnet_opt(inputs):
  """
  The optimized implementation
  """
  H, F, W = _load_weights()
  inputs = tf.reshape(inputs, shape=(-1, info.PRJ_HEIGHT, info.PRJ_WIDTH))

  def _fbp_op(x):
    x = tf.multiply(W, x)
    x = tf.matmul(F, x)
    x = tf.reshape(x, shape=(info.PRJ_HEIGHT * info.PRJ_WIDTH, 1))
    x = tf.sparse_tensor_dense_matmul(H, x)
    return x

  inputs = tf.map_fn(lambda x: tf.multiply(W, x), inputs)  # WP
  inputs = tf.map_fn(lambda x: tf.matmul(F, x), inputs)  # FWP


def main(_):
  inputs = tf.random_uniform(shape=(1, info.PRJ_DEPTH, info.PRJ_HEIGHT, info.PRJ_WIDTH))
  inputs = fbp_subnet(inputs)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(inputs, options=run_options, run_metadata=run_metadata)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
      f.write(ctf)

  # demo_timeline(None)


def demo_timeline(_):
  """
  Here is a stackoverflow example to measure the execution time with TF.Timeline
  url: https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow
  """
  x = tf.random_normal([1000, 1000])
  y = tf.random_normal([1000, 1000])
  res = tf.matmul(x, y)

  # Run the graph with full trace option
  with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(res, options=run_options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
      f.write(ctf)


if __name__ == '__main__':
  tf.app.run()
