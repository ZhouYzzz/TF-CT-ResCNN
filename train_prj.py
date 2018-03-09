"""
Train a single branch of projection sub-network
"""
import tensorflow as tf
from model.subnet.prj_est_impl import conv2d_periodic_padding, batch_norm_relu, slice_concat
from dataset import input_fn
from utils import create_rrmse_metric
import dataset
import os

tf.flags.DEFINE_string('model_dir', '/tmp/train_prj', '')
tf.flags.DEFINE_integer('batch_size', 10, '')

tf.flags.DEFINE_string('gpus', '0', '')

# LEARNING POLICY
tf.flags.DEFINE_float('learning_rate', 0.001, '')
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


def branch_network_v1(inputs, index, is_training):
  with tf.name_scope('B{}'.format(index)):
    inputs = conv2d_periodic_padding(inputs, filters=64, kernel_size=(9, 3), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training)
    shortcut_0 = inputs
    inputs = conv2d_periodic_padding(inputs, filters=128, kernel_size=(9, 3), strides=(2, 2))
    inputs = batch_norm_relu(inputs, is_training)
    shortcut_1 = inputs
    inputs = conv2d_periodic_padding(inputs, filters=256, kernel_size=(9, 3), strides=(2, 2))
    inputs = batch_norm_relu(inputs, is_training)
    shortcut_2 = inputs
    inputs = conv2d_periodic_padding(inputs, filters=512, kernel_size=(9, 3), strides=(1, 1))
    inputs = batch_norm_relu(inputs, is_training)

    inputs = conv2d_periodic_padding(inputs, filters=512, kernel_size=(9, 3), strides=(1, 1))  # 1/8
    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_periodic_padding(inputs, filters=256, kernel_size=(9, 3), strides=(2, 2), transpose=True, padding=False)  # 1/4
    inputs = batch_norm_relu(inputs, is_training)
    inputs += shortcut_2
    inputs = conv2d_periodic_padding(inputs, filters=128, kernel_size=(9, 3), strides=(2, 2), transpose=True, padding=False)  # 1/2
    inputs = batch_norm_relu(inputs, is_training)
    inputs += shortcut_1
    inputs = conv2d_periodic_padding(inputs, filters=64, kernel_size=(9, 3), strides=(2, 2), transpose=True, padding=False)  # 1/1
    inputs = batch_norm_relu(inputs, is_training)
    inputs += shortcut_0
    inputs = conv2d_periodic_padding(inputs, filters=1, kernel_size=(9, 3), strides=(1, 1))  # 1/1
    inputs = tf.identity(inputs, 'outputs')
  return inputs


def l2_loss(source, target):
  source = tf.layers.flatten(source)
  target = tf.layers.flatten(target)
  loss = tf.norm(source - target, axis=1)
  loss = tf.reduce_mean(loss)
  return loss


def model_fn(features, labels, mode):
  inputs = features['inputs']
  branch_outputs = [branch_network_v0(inputs, i, is_training=True) for i in range(1)]
  # projection_outputs = slice_concat(branch_outputs, axis=3)
  # projection_labels = slice_concat([labels['sparse{}'.format(i+1)] for i in range(1)], axis=3)
  projection_outputs = branch_outputs[0]
  projection_labels = labels['sparse1']

<<<<<<< HEAD
  loss = tf.reduce_mean(
    tf.nn.l2_loss(projection_labels - projection_outputs) / (info.PRJ_SPARSE_WIDTH * info.PRJ_HEIGHT))
  loss = tf.identity(loss, 'loss')

  base_loss = tf.reduce_mean(
    tf.nn.l2_loss(inputs - projection_labels) / (info.PRJ_SPARSE_WIDTH * info.PRJ_HEIGHT))
=======
  # loss = tf.nn.l2_loss(projection_labels - projection_outputs) / (FLAGS.batch_size * 1 * dataset.INFO.PRJ_SPARSE_WIDTH * dataset.INFO.PRJ_HEIGHT)
  loss = l2_loss(projection_outputs, projection_labels)
  loss = tf.identity(loss, 'loss')
  # base_loss = tf.nn.l2_loss(inputs - projection_labels) / (FLAGS.batch_size * 1 * dataset.INFO.PRJ_SPARSE_WIDTH * dataset.INFO.PRJ_HEIGHT)
  base_loss = l2_loss(inputs, projection_labels)
>>>>>>> 741ac73e5d7d1c62fcd4df669128e04af1b4a6d2
  base_loss = tf.identity(base_loss, 'base_loss')

  rrmse_metric = create_rrmse_metric(projection_outputs, projection_labels)
  tf.identity(rrmse_metric[1], 'rrmse')

  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
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
  tensors_to_log = ['loss', 'base_loss', 'rrmse']
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
  # estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.model_dir, config=config)
  estimator = tf.estimator.Estimator(model_fn, model_dir=None, config=config)

  estimator.train(lambda: input_fn('train', batch_size=1, num_epochs=1), hooks=[logging_hook])
  estimator.train(lambda: input_fn('train', batch_size=FLAGS.batch_size, num_epochs=1), hooks=[logging_hook])
  print(estimator.evaluate(lambda: input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
