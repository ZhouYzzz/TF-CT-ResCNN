"""
Train a single image refinement network using pre-rfn dataset generated using the weights of pre-trained projection network
"""
import tensorflow as tf
from dataset.input_fn import prerfn_input_fn
from model.image_refinement_network import image_refinement_network
from model.red_cnn import red_cnn
from utils.summary import visualize
from utils.rrmse import create_rrmse_metric
from resnet.cifar10_main import Cifar10Model
import dataset

import os


tf.flags.DEFINE_string('model_dir', None, '')

tf.flags.DEFINE_integer('batch_size', 10, '')
tf.flags.DEFINE_float('learning_rate', 1e-4, '')
tf.flags.DEFINE_float('weight_decay', 1e-4, '')
tf.flags.DEFINE_float('momentum', 0.9, '')
tf.flags.DEFINE_integer('num_epochs', 10, '')

tf.flags.DEFINE_string('gpus', '0', '')
tf.flags.DEFINE_integer('verbose', False, '')

tf.flags.DEFINE_integer('use_gan', False, '')

FLAGS = tf.flags.FLAGS

def random_crop(images, size, num_crops=5):
  return [tf.map_fn(lambda i: tf.random_crop(i, size=size), images) for _ in range(num_crops)]

def discriminator(inputs, reuse=tf.AUTO_REUSE):
  # network = ImagenetModel(resnet_size=18, data_format='channels_first', num_classes=1)
  inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
  with tf.variable_scope('Discriminator', reuse=reuse):
    network = Cifar10Model(resnet_size=20, data_format='channels_first', num_classes=1)
    inputs = network(inputs, training=True)
  return inputs

def create_gan_loss_and_train_op(image_outputs, image_labels):
  samples_crops_fake = random_crop(image_outputs, size=(1, 100, 100))
  samples_crops_real = random_crop(image_labels, size=(1, 100, 100))
  ds_fake = [discriminator(s, reuse=tf.AUTO_REUSE) for s in samples_crops_fake]
  ds_real = [discriminator(s, reuse=tf.AUTO_REUSE) for s in samples_crops_real]
  d_fake = tf.add_n(ds_fake)
  d_real = tf.add_n(ds_real)
  g_loss = tf.reduce_mean(d_fake)
  d_loss = tf.reduce_mean(d_real - d_fake)
  reg_loss = FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables(scope='Discriminator')])
  g_loss += reg_loss
  d_loss += reg_loss
  d_vars = tf.trainable_variables(scope='Discriminator')
  g_vars = tf.trainable_variables(scope='RFN')
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')):
      d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
          .minimize(d_loss, var_list=d_vars)
      g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
          .minimize(g_loss, var_list=g_vars)

  d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
  g_loss = tf.identity(g_loss, 'g_loss')
  d_loss = tf.identity(d_loss, 'd_loss')
  return g_loss, d_loss, g_rmsprop, d_rmsprop

def rfn_model_fn(features, labels, mode, params):
  inputs = features['prerfn'] * 50
  with tf.variable_scope('RFN'):
    outputs = image_refinement_network(inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    #outputs = red_cnn(inputs)
  image_labels = labels['image'] * 50

  loss = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (outputs - image_labels)))
  loss = loss / (dataset.INFO.IMG_DEPTH * dataset.INFO.IMG_HEIGHT * dataset.INFO.IMG_WIDTH)
  loss = tf.identity(loss, 'image_rfn_loss')

  loss += FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables(scope='RFN')])
  loss = tf.identity(loss, 'total_loss')

  base_rrmse_metric = create_rrmse_metric(inputs, image_labels)
  tf.identity(base_rrmse_metric[1], 'base_rrmse')
  tf.summary.scalar('base_rrmse', base_rrmse_metric[1])
  image_rrmse_metric = create_rrmse_metric(outputs, image_labels)
  tf.identity(image_rrmse_metric[1], 'rrmse')
  tf.summary.scalar('rrmse', image_rrmse_metric[1])

  visualize(tf.concat([image_labels, inputs, outputs], axis=3), 'image_compare')

  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      #optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1e-4), var)
                                for grad, var in grads_and_vars if grad is not None]
      train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=tf.train.get_or_create_global_step())
  else:
    train_op = None

  if FLAGS.use_gan:
    print('USING WGAN...')
    _, _, g_train_op, d_train_op = create_gan_loss_and_train_op(outputs, image_labels)
    train_op = tf.group(train_op, g_train_op, d_train_op)

  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=outputs,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops={'rrmse': image_rrmse_metric})


def main(_):
  if FLAGS.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  config = tf.estimator.RunConfig(save_checkpoints_secs=1e9,
                                  keep_checkpoint_max=5)
  #hooks = [tf.train.LoggingTensorHook(tensors=['rrmse', 'base_rrmse', 'd_loss', 'g_loss'],every_n_iter=100)]
  hooks = [tf.train.LoggingTensorHook(tensors=['rrmse', 'base_rrmse'],every_n_iter=100)]
  estimator = tf.estimator.Estimator(rfn_model_fn, model_dir=FLAGS.model_dir, config=config)

  estimator.train(lambda: prerfn_input_fn('train', batch_size=1, num_epochs=1), hooks=hooks, steps=2000)
  print(
    estimator.evaluate(lambda: prerfn_input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))

  for _ in range(FLAGS.num_epochs):
    estimator.train(lambda: prerfn_input_fn('train', batch_size=FLAGS.batch_size, num_epochs=1), hooks=hooks)
    print(
      estimator.evaluate(lambda: prerfn_input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))


if __name__ == '__main__':
  tf.app.run()
