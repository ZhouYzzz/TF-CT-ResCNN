import tensorflow as tf


def train_example_spec():
  spec = dict()
  spec['sparse1'] = tf.FixedLenFeature(shape=(1, 216, 72), dtype=tf.float32)
  spec['sparse2'] = tf.FixedLenFeature(shape=(1, 216, 72), dtype=tf.float32)
  spec['sparse3'] = tf.FixedLenFeature(shape=(1, 216, 72), dtype=tf.float32)
  spec['sparse4'] = tf.FixedLenFeature(shape=(1, 216, 72), dtype=tf.float32)
  spec['sparse5'] = tf.FixedLenFeature(shape=(1, 216, 72), dtype=tf.float32)
  spec['image'] = tf.FixedLenFeature(shape=(1, 200, 200), dtype=tf.float32)
  return spec


def serve_example_spec():
  spec = dict()
  spec['sparse3'] = tf.FixedLenFeature(shape=(1, 216, 72), dtype=tf.float32)
  return spec
