"""
ExampleSpec definitions
"""
import tensorflow as tf
import dataset.info as info


def train_example_spec():
  return dict({
    'sparse1': tf.FixedLenFeature(shape=(info.PRJ_DEPTH, info.PRJ_HEIGHT, info.PRJ_SPARSE_WIDTH), dtype=tf.float32),
    'sparse2': tf.FixedLenFeature(shape=(info.PRJ_DEPTH, info.PRJ_HEIGHT, info.PRJ_SPARSE_WIDTH), dtype=tf.float32),
    'sparse3': tf.FixedLenFeature(shape=(info.PRJ_DEPTH, info.PRJ_HEIGHT, info.PRJ_SPARSE_WIDTH), dtype=tf.float32),
    'sparse4': tf.FixedLenFeature(shape=(info.PRJ_DEPTH, info.PRJ_HEIGHT, info.PRJ_SPARSE_WIDTH), dtype=tf.float32),
    'sparse5': tf.FixedLenFeature(shape=(info.PRJ_DEPTH, info.PRJ_HEIGHT, info.PRJ_SPARSE_WIDTH), dtype=tf.float32),
    'image':   tf.FixedLenFeature(shape=(info.IMG_DEPTH, info.IMG_HEIGHT, info.IMG_WIDTH),        dtype=tf.float32),
  })


if __name__ == '__main__':
  print(train_example_spec())
