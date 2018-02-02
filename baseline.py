"""
Evaluate three baseline reconstruction method on the sparse dataset.

1. Ground Truth (Full Views).
2. Sparse Views Only.
3. Linear interpolation.

This script does not need training.
"""
import tensorflow as tf
import os
import dataset.info as info
from dataset.example_spec import train_example_spec
from model.subnet.prj_est_impl import slice_concat
from model.subnet.fbp import fbp_subnet

tf.flags.DEFINE_string('model_dir', '/tmp/TF-CT-ResCNN/baseline', '')
tf.flags.DEFINE_string('data_dir', 'dataset', '')
FLAGS = tf.flags.FLAGS


def eval_input_fn(batch_size):
  dataset = tf.data.TFRecordDataset(os.path.join(FLAGS.data_dir, 'val.tfrecords'))
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  labels = features
  return {'inputs': features['sparse3']}, labels


def create_rrmse_metric(source, target):
  source = tf.layers.flatten(source)
  target = tf.layers.flatten(target)
  diff = source - target
  rrmse = tf.norm(diff, axis=1) / tf.norm(target, axis=1)
  return tf.metrics.mean(rrmse)


def shared_model(features, labels, mode, params):
  # assert mode is tf.estimator.ModeKeys.TRAIN

  inputs = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)
  outputs = fbp_subnet(inputs)
  images = labels['image']

  train_op = tf.assign_add(tf.train.get_or_create_global_step(), 1)
  rrmse_metric_op = create_rrmse_metric(outputs, images)
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=outputs,
                                    loss=tf.constant(0),
                                    train_op=train_op,
                                    eval_metric_ops={'rrmse': rrmse_metric_op})


def ground_truth_model(features, labels, mode, params):
  tf.train.init_from_checkpoint(FLAGS.model_dir, assignment_map={'': ''})
  pass


def sparse_model(features, labels, mode, params):
  pass


def linear_interpolation_model(features, labels, mode, params):
  pass


def main(_):
  estimator = tf.estimator.Estimator(model_fn=shared_model,
                                     model_dir=FLAGS.model_dir)
  estimator.train(input_fn=lambda : eval_input_fn(batch_size=1), hooks=None, max_steps=1)
  eval_results = estimator.evaluate(input_fn=lambda : eval_input_fn(batch_size=10))
  print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
