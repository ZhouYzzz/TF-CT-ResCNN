"""Trains the projection estimation network for sparse-angled back-projection"""
import tensorflow as tf
import tensorflow.contrib.training as training
import tensorflow.contrib.losses as losses
import tensorflow.contrib.layers as layers

from dataset.input_fn import input_fn
from model.projection_estimation_network import projection_estimation_network_v1
from model.subnet.prj_est_impl import slice_concat
from model.subnet.fbp import fbp_subnet
from utils.summary import visualize
from utils.rrmse import create_rrmse_metric

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='proposed')
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--pretrain_steps', type=int, default=2000)
parser.add_argument('--num_epoches', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip_gradient', type=float, default=1e-4)

FLAGS, _ = parser.parse_known_args()


def model_fn(features, labels, mode, params):
  # Define model inputs and labels
  sparse_inputs = features['inputs']  # the sparse projection (sparse3)
  full_projection_labels = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)

  # Define the model
  with tf.variable_scope('Projection'):
    if FLAGS.model == 'proposed':
      projection_outputs = projection_estimation_network_v1(sparse_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    elif FLAGS.model == 'v2':  # using shared weights and skip connection
      from model.projection_estimation_network_v2 import projection_estimation_network_v2
      projection_outputs = projection_estimation_network_v2(sparse_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    else:
      raise ValueError('No recognized model named `{}`'.format(FLAGS.model))

  with tf.variable_scope('FBP'):
    image_outputs = fbp_subnet(projection_outputs)

  # Define losses
  tf.losses.mean_squared_error(projection_outputs, full_projection_labels)
  [tf.losses.add_loss(FLAGS.weight_decay * tf.nn.l2_loss(v), loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) for v in tf.trainable_variables('Projection')]
  loss = tf.losses.get_total_loss()

  # Define summaries
  visualize(tf.concat([labels['image'], image_outputs], axis=3), name='image')
  visualize(tf.concat([full_projection_labels, projection_outputs], axis=2), name='projection')

  # Define metrics
  metric = create_rrmse_metric(image_outputs, labels['image'])
  tf.summary.scalar('rrmse', tf.identity(metric[1], 'rrmse'))

  train_op = training.create_train_op(
    total_loss=loss,
    optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
    global_step=tf.train.get_or_create_global_step(),
    update_ops=None,
    variables_to_train=tf.trainable_variables(scope='Projection'),
    transform_grads_fn=training.clip_gradient_norms_fn(max_norm=FLAGS.clip_gradient))

  return tf.estimator.EstimatorSpec(
    mode,
    predictions={'projection_outputs': projection_outputs, 'image_outputs': image_outputs},
    loss=loss,
    train_op=train_op,
    eval_metric_ops={'rrmse': metric})


def main(_):
  config = tf.estimator.RunConfig(save_checkpoints_secs=1e9)
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config)
  if FLAGS.pretrain_steps > 0:
    estimator.train(lambda: input_fn('train', batch_size=1, num_epochs=1),
                    hooks=[tf.train.LoggingTensorHook(['total_loss', 'rrmse'], every_n_iter=100)],
                    max_steps=FLAGS.pretrain_steps)
    print(estimator.evaluate(lambda: input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))

  for _ in range(FLAGS.num_epoches):
    estimator.train(lambda: input_fn('train', batch_size=FLAGS.batch_size, num_epochs=1),
                    hooks=[tf.train.LoggingTensorHook(['total_loss', 'rrmse'], every_n_iter=100)])
    print(estimator.evaluate(lambda: input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))


if __name__ == '__main__':
  print(FLAGS)
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
