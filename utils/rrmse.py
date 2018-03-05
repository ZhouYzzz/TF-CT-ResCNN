import tensorflow as tf


def create_rrmse_metric(source, target):
  source = tf.layers.flatten(source)
  target = tf.layers.flatten(target)
  diff = source - target
  rrmse = tf.norm(diff, axis=1) / tf.norm(target, axis=1)
  return tf.metrics.mean(rrmse)
