import tensorflow as tf
import numpy as np

import os


def main():
  print(os.path.dirname(__file__))
  data = np.array([1,2,3,4,5])
  dataset = tf.data.Dataset.from_tensor_slices(data)  # type: tf.data.Dataset
  dataset = dataset.repeat(count=None)
  iterator = dataset.make_one_shot_iterator()
  example = iterator.get_next()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
      print(sess.run(example))


if __name__ == '__main__':
  main()
