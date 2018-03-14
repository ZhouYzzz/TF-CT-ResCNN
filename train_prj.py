"""
Train a single branch of projection sub-network
"""
import tensorflow as tf
from model.subnet.prj_est_impl import conv2d_periodic_padding, batch_norm_relu, slice_concat
from model.subnet.prj_est_proposed import _prj_est_subnet_branch
from dataset import input_fn
from utils import create_rrmse_metric
import dataset
import os
from model.red_cnn import red_cnn
from utils.summary import visualize

tf.flags.DEFINE_string('model_dir', '/tmp/train_prj', '')
tf.flags.DEFINE_integer('batch_size', 10, '')

tf.flags.DEFINE_string('gpus', '0', '')

# LEARNING POLICY
tf.flags.DEFINE_float('learning_rate', 1e-3, '')
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

    inputs = conv2d_periodic_padding(inputs, filters=512, kernel_size=(9, 3), strides=(2, 2))  # 1/8
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
    inputs = tf.nn.relu(inputs)
    inputs = tf.identity(inputs, 'outputs')
  return inputs


# def l2_loss(source, target):
#   source = tf.layers.flatten(source)
#   target = tf.layers.flatten(target)
#   # loss = tf.map_fn(tf.nn.l2_loss, source - target)
#   loss = tf.reduce_mean(loss)
#   return loss


def model_fn(features, labels, mode):
  inputs = features['inputs']
  # branch_outputs = [branch_network_v0(inputs, i, is_training=True) for i in range(1)]
  # branch_outputs = [_prj_est_subnet_branch(inputs, i, is_training=True) for i in range(1)]

  loss = 0
  branch_outputs = []
  for i in range(5):
    with tf.variable_scope('B{}'.format(i), reuse=tf.AUTO_REUSE):
      outputs = branch_network_v1(inputs, i, is_training=True)
      loss = loss + tf.reduce_mean(
        tf.map_fn(tf.nn.l2_loss, labels['sparse{}'.format(i+1)] - outputs))
      branch_outputs.append(outputs)
  loss = loss / (dataset.INFO.PRJ_DEPTH * dataset.INFO.PRJ_WIDTH * dataset.INFO.PRJ_HEIGHT)
  loss = tf.identity(loss, 'prj_loss')
  prj_outputs = slice_concat(branch_outputs, axis=3)
  prj_outputs = tf.identity(prj_outputs, 'prj_outputs')
  prj_labels = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)

  visualize(tf.concat([prj_labels, prj_outputs], axis=2), 'prjs')
  loss = loss + 1e-4 * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  loss = tf.identity(loss, 'total_loss')

  rrmse_metric = create_rrmse_metric(prj_outputs, prj_labels)
  tf.identity(rrmse_metric[1], 'rrmse')

  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
      # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1e-4), var)
                                for grad, var in grads_and_vars if grad is not None]
      train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=tf.train.get_or_create_global_step())
      # train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
  else:
    train_op = None

  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=prj_outputs,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops={'rrmse_metric': rrmse_metric})


# def l2_loss(t):
#   return tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, t))



def branch_network_v2(inputs):
  stddev = 0.001
  kernel_initializer = tf.contrib.layers.xavier_initializer()  # tf.random_normal_initializer(stddev=stddev)
  kernel_initializer = tf.random_normal_initializer(stddev=stddev)
  inputs = tf.layers.conv2d(inputs, 64, (5, 5), padding='same', data_format='channels_first',
                            kernel_initializer=kernel_initializer, use_bias=True)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 64, (5, 5), padding='same', data_format='channels_first',
                            kernel_initializer=kernel_initializer, use_bias=True)
  inputs = tf.nn.leaky_relu(inputs)
  sc0 = inputs
  inputs = tf.layers.conv2d(inputs, 64, (5, 5), padding='same', data_format='channels_first',
                            kernel_initializer=kernel_initializer, use_bias=True)
  inputs = tf.nn.leaky_relu(inputs)
  inputs += sc0
  inputs = tf.layers.conv2d(inputs, 64, (5, 5), padding='same', data_format='channels_first',
                            kernel_initializer=kernel_initializer, use_bias=True)
  inputs = tf.nn.leaky_relu(inputs)
  sc0 = inputs
  inputs = tf.layers.conv2d(inputs, 64, (5, 5), padding='same', data_format='channels_first',
                            kernel_initializer=kernel_initializer, use_bias=True)
  inputs = tf.nn.leaky_relu(inputs)
  inputs += sc0
  inputs = tf.layers.conv2d(inputs, 16, (5, 5), padding='same', data_format='channels_first',
                            kernel_initializer=kernel_initializer, use_bias=True)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.layers.conv2d(inputs, 1, (5, 5), padding='same', data_format='channels_first',
                            kernel_initializer=kernel_initializer, use_bias=True)
  inputs = tf.nn.leaky_relu(inputs)
  inputs = tf.identity(inputs, 'outputs')
  return inputs


def prj_est_network_v2(inputs):
  branch_outputs = list()
  for i in range(5):
    with tf.variable_scope('Branch{}'.format(i)):
      outputs = branch_network_v2(inputs)
      branch_outputs.append(outputs)
  outputs = slice_concat(branch_outputs, axis=3)
  outputs = tf.identity(outputs, 'outputs')
  return outputs


def prj_model_fn(features, labels, mode):
  inputs = features['inputs']
  sparse_outputs = slice_concat([inputs for _ in range(5)], axis=3)
  # prj_outputs = branch_network_v2(inputs)
  prj_outputs = prj_est_network_v2(inputs)
  prj_labels = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)

  loss = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (prj_outputs - prj_labels)))
  loss = loss / (dataset.INFO.PRJ_DEPTH * dataset.INFO.PRJ_WIDTH * dataset.INFO.PRJ_HEIGHT)
  loss = tf.identity(loss, 'prj_loss')

  loss += 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  loss = tf.identity(loss, 'total_loss')

  base_loss = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (sparse_outputs - prj_labels)))
  base_loss = base_loss / (dataset.INFO.PRJ_DEPTH * dataset.INFO.PRJ_WIDTH * dataset.INFO.PRJ_HEIGHT)
  base_loss = tf.identity(base_loss, 'base_loss')

  # visualize(prj_outputs, 'prj_outputs')
  visualize(tf.concat([prj_labels, prj_outputs], axis=2), 'prjs_compare')
  rrmse_metric = create_rrmse_metric(prj_outputs, prj_labels)
  tf.identity(rrmse_metric[1], 'rrmse')
  tf.summary.scalar('rrmse', rrmse_metric[1])

  base_rrmse_metric = create_rrmse_metric(sparse_outputs, prj_labels)
  tf.identity(base_rrmse_metric[1], 'base_rrmse')
  tf.summary.scalar('base_rrmse', base_rrmse_metric[1])

  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
      optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)  # FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1e-4), var)
                                for grad, var in grads_and_vars if grad is not None]
      train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=tf.train.get_or_create_global_step())

      # train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
  else:
    train_op = None
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=prj_outputs,
                                    loss=loss,
                                    train_op=train_op)


def branch_model_fn(features, labels, mode):
  inputs = features['inputs']
  prj_outputs = branch_network_v2(inputs)

  loss = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (prj_outputs - labels['sparse1'])))
  loss = loss / (dataset.INFO.PRJ_DEPTH * dataset.INFO.PRJ_SPARSE_WIDTH * dataset.INFO.PRJ_HEIGHT)
  loss = tf.identity(loss, 'prj_loss')

  loss += 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  loss = tf.identity(loss, 'total_loss')

  base_loss = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (inputs - labels['sparse1'])))
  base_loss = base_loss / (dataset.INFO.PRJ_DEPTH * dataset.INFO.PRJ_SPARSE_WIDTH * dataset.INFO.PRJ_HEIGHT)
  base_loss = tf.identity(base_loss, 'base_loss')

  visualize(prj_outputs, 'prj_outputs')
  visualize(tf.concat([labels['sparse1'], prj_outputs], axis=3), 'prjs')
  rrmse_metric = create_rrmse_metric(prj_outputs, labels['sparse1'])
  tf.identity(rrmse_metric[1], 'rrmse')

  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
      optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)#FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1e-4), var)
                                for grad, var in grads_and_vars if grad is not None]
      train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=tf.train.get_or_create_global_step())

      # train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
  else:
    train_op = None
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=prj_outputs,
                                    loss=loss,
                                    train_op=train_op)



def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                            keep_checkpoint_max=1)
  tensors_to_log = ['prj_loss', 'total_loss', 'rrmse', 'base_rrmse', 'base_loss']
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
  # estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.model_dir, config=config)
  # estimator = tf.estimator.Estimator(model_fn, model_dir=None, config=config)
  estimator = tf.estimator.Estimator(prj_model_fn, model_dir=None, config=config)

  estimator.train(lambda: input_fn('train', batch_size=1, num_epochs=1), hooks=[logging_hook], max_steps=2000)
  for _ in range(3):
    estimator.train(lambda: input_fn('train', batch_size=FLAGS.batch_size, num_epochs=1), hooks=[logging_hook])
    print(estimator.evaluate(lambda: input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))
  # print(estimator.evaluate(lambda: input_fn('val', batch_size=1, num_epochs=1)))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
