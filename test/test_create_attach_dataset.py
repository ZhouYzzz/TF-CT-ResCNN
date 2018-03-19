import tensorflow as tf
from utils.features import feature_float
import numpy as np


def main(_):
  sess = tf.InteractiveSession()
  example = tf.placeholder(tf.string, shape=(None,))
  tf.saved_model.loader.load(sess, ['serve'],
                             '../tmpouizb6k5_projection_estimation_network/1521426198',
                             input_map={'input_example_tensor:0': example})
  print(tf.get_default_graph().get_tensor_by_name('Branch1/outputs:0'))
  print(tf.get_default_graph().get_tensor_by_name('input_example_tensor:0'))
  outputs = tf.get_default_graph().get_tensor_by_name('FBP/outputs:0')
  # print(tf.get_default_graph().get_tensor_by_name('inputs:0'))

  tf.train.Example()

  ## TRAIN
  # for i in range(10):
  #   print('dataset{}'.format(i))
  #   dataset = tf.data.TFRecordDataset(['../dataset/train_{}.tfrecords'.format(i)])
  #   dataset = dataset.repeat(1)
  #   dataset = dataset.batch(16)
  #   iterator = dataset.make_one_shot_iterator()
  #   ss = iterator.get_next()
  #
  #   writer = tf.python_io.TFRecordWriter(path='../dataset/train_{}.prerfn.tfrecords'.format(i))
  #
  #   try:
  #     count = 0
  #     while True:
  #       serialized = sess.run(ss)
  #       results = sess.run(outputs, feed_dict={example: serialized})  # type: np.ndarray
  #       results = results.reshape(-1, 200 * 200)
  #       results = np.vsplit(results, results.shape[0])
  #       count += len(results)
  #       print(count)
  #       for r in results:
  #         writer.write(tf.train.Example(features=tf.train.Features(feature={'prerfn': feature_float(r.reshape(-1).tolist())})).SerializeToString())
  #   except:
  #     pass
  #
  #   writer.close()

  # EVAL
  dataset = tf.data.TFRecordDataset(['../dataset/val.tfrecords'])
  dataset = dataset.repeat(1)
  dataset = dataset.batch(16)
  iterator = dataset.make_one_shot_iterator()
  ss = iterator.get_next()

  writer = tf.python_io.TFRecordWriter(path='../dataset/val.prerfn.tfrecords')

  try:
    count = 0
    while True:
      serialized = sess.run(ss)
      results = sess.run(outputs, feed_dict={example: serialized})  # type: np.ndarray
      results = results.reshape(-1, 200 * 200)
      results = np.vsplit(results, results.shape[0])
      count += len(results)
      print(count)
      for r in results:
        writer.write(tf.train.Example(
          features=tf.train.Features(feature={'prerfn': feature_float(r.reshape(-1).tolist())})).SerializeToString())
  except:
    pass

  writer.close()

if __name__ == '__main__':
  tf.app.run()
