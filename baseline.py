"""
Evaluate three baseline reconstruction method on the sparse validation dataset.

1. Ground Truth (Full Views).
2. Sparse Views Only.
3. Linear interpolation.
"""
import tensorflow as tf
import os
import dataset.info as info
from dataset.example_spec import train_example_spec
from model.subnet.prj_est_impl import slice_concat
from model.subnet.fbp import fbp_subnet
import glob

tf.flags.DEFINE_string('model_dir', '/tmp/TF-CT-ResCNN/baseline', '')
tf.flags.DEFINE_string('data_dir', 'dataset', '')
tf.flags.DEFINE_integer('batch_size', 16, '')
FLAGS = tf.flags.FLAGS


def train_input_fn(batch_size, epochs=None):
  dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(FLAGS.data_dir, 'train*.tfrecords')))
  dataset.repeat(epochs=epochs)
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  labels = features
  return {'inputs': features['sparse3']}, labels


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


def groundtruth_model(features, labels, mode):
  """
  Define the groundtruth model spec, i.e. reconstructing using FBP on full views
  :param features:
  :param labels:
  :param mode:
  :return:
  """
  inputs = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)
  outputs = fbp_subnet(inputs)
  labels = labels['image']

  loss = tf.reduce_mean(tf.nn.l2_loss(outputs - labels) / (info.PRJ_DEPTH * info.PRJ_HEIGHT * info.PRJ_WIDTH))
  train_op = tf.assign_add(tf.train.get_or_create_global_step(), 1) if mode == tf.estimator.ModeKeys.TRAIN else None
  rrmse_metric_op = create_rrmse_metric(outputs, labels)
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=outputs,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops={'rrmse': rrmse_metric_op})


def sparse_only_model(features, labels, mode):
  """
  Define the sparse only model spec, i.e. reconstructing using FBP on sparse views (repeated)
  :param features:
  :param labels:
  :param mode:
  :return:
  """
  inputs = slice_concat([features['inputs'] for _ in range(5)], axis=3)
  outputs = fbp_subnet(inputs)
  labels = labels['image']

  loss = tf.reduce_mean(tf.nn.l2_loss(outputs - labels) / (info.PRJ_DEPTH * info.PRJ_HEIGHT * info.PRJ_WIDTH))
  train_op = tf.assign_add(tf.train.get_or_create_global_step(), 1) if mode == tf.estimator.ModeKeys.TRAIN else None
  rrmse_metric_op = create_rrmse_metric(outputs, labels)
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=outputs,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops={'rrmse': rrmse_metric_op})


# def shared_model(features, labels, mode):
#   assert mode is tf.estimator.ModeKeys.TRAIN
#
#   inputs = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)
#   outputs = fbp_subnet(inputs)
#   images = labels['image']
#
#   train_op = tf.assign_add(tf.train.get_or_create_global_step(), 1)
#   rrmse_metric_op = create_rrmse_metric(outputs, images)
#   return tf.estimator.EstimatorSpec(mode=mode,
#                                     predictions=outputs,
#                                     loss=tf.constant(0),
#                                     train_op=train_op,
#                                     eval_metric_ops={'rrmse': rrmse_metric_op})
#
#
# def ground_truth_model(features, labels, mode):
#   assert mode is tf.estimator.ModeKeys.EVAL
#
#   inputs = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)
#   outputs = fbp_subnet(inputs)
#   images = labels['image']
#
#   tf.train.init_from_checkpoint(FLAGS.model_dir, assignment_map={})
#
#   rrmse_metric_op = create_rrmse_metric(outputs, images)
#   return tf.estimator.EstimatorSpec(mode=mode,
#                                     predictions=outputs,
#                                     loss=tf.constant(0),
#                                     train_op=None,
#                                     eval_metric_ops={'rrmse': rrmse_metric_op})
#
#
# def sparse_model(features, labels, mode):
#   assert mode is tf.estimator.ModeKeys.EVAL
#
#   inputs = slice_concat([labels['sparse3'] for _ in range(5)], axis=3)
#   outputs = fbp_subnet(inputs)
#   images = labels['image']
#
#   tf.train.init_from_checkpoint(FLAGS.model_dir, assignment_map={})
#
#   rrmse_metric_op = create_rrmse_metric(outputs, images)
#   return tf.estimator.EstimatorSpec(mode=mode,
#                                     predictions=outputs,
#                                     loss=tf.constant(0),
#                                     train_op=None,
#                                     eval_metric_ops={'rrmse': rrmse_metric_op})
#
#
# def linear_interpolation_model(features, labels, mode):
#   raise NotImplementedError


def main(_):
  # estimator = tf.estimator.Estimator(model_fn=shared_model,
  #                                    model_dir=FLAGS.model_dir)
  # estimator.train(input_fn=lambda : eval_input_fn(batch_size=1), hooks=None, max_steps=1)
  #
  # estimator = tf.estimator.Estimator(model_fn=ground_truth_model,
  #                                    model_dir=FLAGS.model_dir)
  # eval_results = estimator.evaluate(input_fn=lambda : eval_input_fn(batch_size=10))
  # print(eval_results)
  # estimator = tf.estimator.Estimator(model_fn=sparse_model,
  #                                    model_dir=FLAGS.model_dir)
  # eval_results = estimator.evaluate(input_fn=lambda: eval_input_fn(batch_size=10))
  # print(eval_results)
  config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                            save_summary_steps=100,
                                            keep_checkpoint_max=1)
  estimator = tf.estimator.Estimator(model_fn=groundtruth_model,
                                     model_dir=FLAGS.model_dir,
                                     config=config)
  estimator.train(input_fn=lambda: train_input_fn(FLAGS.batch_size), max_steps=1000)
  eval_results = estimator.evaluate(input_fn=lambda: eval_input_fn(FLAGS.batch_size))
  print(eval_results)

  estimator = tf.estimator.Estimator(model_fn=sparse_only_model,
                                     model_dir=FLAGS.model_dir,
                                     config=config)
  eval_results = estimator.evaluate(input_fn=lambda: eval_input_fn(FLAGS.batch_size))
  print(eval_results)
  tf.sparse_tensor_dense_matmul()



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
