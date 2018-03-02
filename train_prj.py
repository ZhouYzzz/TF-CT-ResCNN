"""
Train a single branch of projection sub-network
"""
import tensorflow as tf
from model.subnet.prj_est_impl import conv2d_periodic_padding, batch_norm_relu, slice_concat
from baseline import create_rrmse_metric, train_input_fn, eval_input_fn
import dataset.info as info
import os

# tf.flags.DEFINE_string('model_dir', '/tmp/prj', '')

tf.flags.DEFINE_string('gpus', '0', '')
tf.flags.DEFINE_float('learning_rate', 0.01, '')
tf.flags.DEFINE_float('momentum', 0.9, '')
tf.flags.DEFINE_float('weight_decay', 2e-4, '')
FLAGS = tf.flags.FLAGS


def branch_network_v0(inputs, index, is_training):
  with tf.name_scope('B{}'.format(index)):
    inputs = conv2d_periodic_padding(inputs, filters=64, kernel_size=(7, 7), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training=is_training)
    inputs = conv2d_periodic_padding(inputs, filters=64, kernel_size=(7, 7), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training=is_training)
    shortcut = inputs
    inputs = conv2d_periodic_padding(inputs, filters=64, kernel_size=(7, 7), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training=is_training)
    inputs = inputs + shortcut
    inputs = conv2d_periodic_padding(inputs, filters=64, kernel_size=(7, 7), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training=is_training)
    shortcut = inputs
    inputs = conv2d_periodic_padding(inputs, filters=64, kernel_size=(7, 7), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training=is_training)
    inputs = inputs + shortcut
    inputs = conv2d_periodic_padding(inputs, filters=16, kernel_size=(7, 7), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training=is_training)
    inputs = conv2d_periodic_padding(inputs, filters=1, kernel_size=(7, 7), strides=(1, 1))
    inputs = tf.identity(inputs, 'outputs')
  return inputs


def model_fn(features, labels, mode):
  inputs = features
  branch_outputs = [branch_network_v0(inputs, i, is_training=True) for i in range(1)]
  projection_outputs = slice_concat(branch_outputs, axis=3)
  projection_labels = slice_concat([labels['sparse{}'.format(i+1)] for i in range(1)], axis=3)

  loss = tf.nn.l2_loss(projection_labels - projection_outputs) / (FLAGS.batch_size * 1 * info.PRJ_WIDTH * info.PRJ_HEIGHT)
  loss = tf.identity(loss, 'loss')

  rrmse_metric = create_rrmse_metric(projection_outputs, projection_labels)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
    train_op = optimizer.minimize(loss)
  else:
    train_op = None

  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=projection_outputs,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops={'rrmse_metric': rrmse_metric})


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                            keep_checkpoint_max=1)
  tensors_to_log = ['loss', 'rrmse_metric']
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
  estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.model_dir, config=config)
  estimator.train(lambda: train_input_fn(FLAGS.batch_size), hooks={logging_hook})
  print(estimator.evaluate(lambda: eval_input_fn(FLAGS.batch_size)))


if __name__ == '__main__':
  tf.app.run()
