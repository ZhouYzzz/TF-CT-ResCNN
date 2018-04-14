"""Ref: LOG0414
Create a sparse dataset for standard training and comparing of image refinement network
"""
import tensorflow as tf
import numpy as np
from dataset.example_spec import train_example_spec
from utils.features import feature_float


def model_fn(features, labels, mode):
  """FBP network only"""
  from model import fbp_network, slice_concat
  inputs = features['sparse3']
  inputs = slice_concat([inputs for _ in range(5)], axis=3)
  outputs = fbp_network(inputs)
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=outputs)


def main(record_name):
  dataset = tf.data.TFRecordDataset(['../dataset/{}.tfrecords'.format(record_name)])  # type: tf.data.Dataset
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  dataset = dataset.repeat(1)
  dataset = dataset.batch(16)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  model = model_fn(features, features, mode=tf.estimator.ModeKeys.PREDICT)

  writer = tf.python_io.TFRecordWriter('../dataset/{}.sparse.tfrecords'.format(record_name))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
      while True:
        results = sess.run(model.predictions)  # type: np.ndarray
        results = results.reshape(-1, 200 * 200)
        results = np.vsplit(results, results.shape[0])
        for r in results:
          writer.write(tf.train.Example(features=tf.train.Features(feature={'sparse':
                                                                            feature_float(r.reshape(-1).tolist())})).SerializeToString())
    except:
      print('End of sequence {}'.format(record_name))
      writer.close()


if __name__ == '__main__':
  for i in range(0):
    record_name = 'train_{}'.format(i)
    main(record_name=record_name)
  main(record_name='val')
