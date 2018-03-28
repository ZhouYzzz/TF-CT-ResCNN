import tensorflow as tf
from dataset.input_fn import prerfn_input_fn
from model.discriminator import discriminator
from model.image_refinement_network import image_refinement_network
from utils.rrmse import create_rrmse_metric
from utils.summary import visualize
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


def create_wgan(fake, real, g_vars, opt_scope):
  """
  Create the WGAN given a fake image tensor and a real image tensor
  :param fake: a 4-D image tensor which is generated from some generator
  :param real: a 4-D image tensor which is sampled from natural distribution
  :return: a `train_op` that holds all the train stuffs with a single run
  """
  cropped_fake = tf.map_fn(lambda x: tf.random_crop(x, size=(1, 100, 100)), fake)
  cropped_real = tf.map_fn(lambda x: tf.random_crop(x, size=(1, 100, 100)), real)

  with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
    outputs_fake = discriminator(cropped_fake)

  with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
    outputs_real = discriminator(cropped_real)

  reg_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables(scope='D')])
  g_loss = tf.reduce_mean(outputs_fake) + reg_loss
  d_loss = tf.reduce_mean(outputs_real - outputs_fake) + reg_loss
  g_loss = tf.identity(g_loss, 'g_loss')
  d_loss = tf.identity(d_loss, 'd_loss')

  d_vars = tf.trainable_variables('D')
  # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')):
  with tf.variable_scope(opt_scope):
    g_train_op = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(g_loss, var_list=g_vars)
    d_train_op = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(d_loss, var_list=d_vars)
  return g_train_op, d_train_op


def rfn_model_fn(features, labels, mode):
  inputs = features['prerfn'] * 50
  with tf.variable_scope('RFN'):
    outputs = image_refinement_network(inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
  image_labels = labels['image'] * 50

  g_vars = tf.trainable_variables(scope='RFN')

  loss = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (outputs - image_labels)))
  loss = loss / (dataset.INFO.IMG_DEPTH * dataset.INFO.IMG_HEIGHT * dataset.INFO.IMG_WIDTH)
  # # loss = tf.identity(loss, 'image_rfn_loss')
  #
  loss += 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in g_vars])
  loss = tf.identity(loss, 'image_rfn_loss')

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
      # optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1e-4), var)
                                for grad, var in grads_and_vars if grad is not None]
      train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=tf.train.get_or_create_global_step())
  else:
    train_op = None

  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=outputs,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops={'rrmse': image_rrmse_metric})


def gan_model_fn(features, labels, mode, params):
  """Model function with WGAN loss only"""
  inputs = features['prerfn'] * 50
  with tf.variable_scope('RFN'):
    outputs = image_refinement_network(inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
  image_labels = labels['image'] * 50

  g_vars = tf.trainable_variables(scope='RFN')

  if 'pretrained_model_dir' in params.keys():
    tf.train.init_from_checkpoint(params['pretrained_model_dir'], assignment_map={'RFN/': 'RFN/'})

  loss = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (outputs - image_labels)))
  loss = loss / (dataset.INFO.IMG_DEPTH * dataset.INFO.IMG_HEIGHT * dataset.INFO.IMG_WIDTH)
  # # loss = tf.identity(loss, 'image_rfn_loss')
  #
  # loss += 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in g_vars])
  # loss = tf.identity(loss, 'image_rfn_loss')

  ## RRMSE THINGS
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
      # define the wgan loss and train_ops
      train_op = tf.no_op()
      for i in range(5):
        with tf.control_dependencies([train_op]):
          g_train_op, d_train_op = create_wgan(outputs, image_labels, g_vars=g_vars, opt_scope='OPT_{}'.format(i))
          if i == 0:
            train_op = tf.group(train_op, g_train_op, d_train_op)
          else:
            train_op = tf.group(train_op, d_train_op)

      train_op = tf.group(train_op, tf.assign_add(tf.train.get_or_create_global_step(), 1))


      # #optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
      # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      # grads_and_vars = optimizer.compute_gradients(loss)
      # clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1e-4), var)
      #                           for grad, var in grads_and_vars if grad is not None]
      # train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=tf.train.get_or_create_global_step())
  else:
    train_op = None

  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=outputs,
                                    loss=loss,
                                    train_op=train_op)


def main(_):
  # if FLAGS.verbose:
  if True:
    tf.logging.set_verbosity(tf.logging.INFO)
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  config = tf.estimator.RunConfig(save_checkpoints_secs=1e9,
                                  keep_checkpoint_max=5)
  # hooks = [tf.train.LoggingTensorHook(tensors=['rrmse', 'base_rrmse', 'd_loss', 'g_loss'],every_n_iter=100)]
  hooks = [tf.train.LoggingTensorHook(tensors=['rrmse', 'base_rrmse'], every_n_iter=100)]
  # from train_rfn import rfn_model_fn
  estimator = tf.estimator.Estimator(rfn_model_fn, model_dir=None, config=config)

  estimator.train(lambda: prerfn_input_fn('train', batch_size=1, num_epochs=1), hooks=hooks, steps=1000)
  print(
    estimator.evaluate(lambda: prerfn_input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))

  pretrained_model_dir = estimator.model_dir

  ## GAN

  estimator = tf.estimator.Estimator(gan_model_fn, model_dir=FLAGS.model_dir, config=config, params={'pretrained_model_dir': pretrained_model_dir})
  hooks = [tf.train.LoggingTensorHook(tensors=['rrmse', 'base_rrmse', 'd_loss:0', 'g_loss:0'], every_n_iter=100)]

  for _ in range(FLAGS.num_epochs):
    estimator.train(lambda: prerfn_input_fn('train', batch_size=FLAGS.batch_size, num_epochs=1), hooks=hooks)
    print(
      estimator.evaluate(lambda: prerfn_input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))


def _main(_):
  features, labels = prerfn_input_fn()
  inputs = features['prerfn']
  # a simple model
  with tf.variable_scope('G'):
    # inputs = tf.layers.conv2d(inputs, 1, (3, 3), (1, 1), padding='same', data_format='channels_first')
    # inputs = tf.nn.relu(inputs)
    inputs = image_refinement_network(inputs, training=True)
  labels = labels['image']

  g_vars = tf.trainable_variables('G')

  train_op = tf.no_op()
  for i in range(5):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.TRAIN_OP)):
      g_train_op, d_train_op = create_wgan(inputs, labels, g_vars=g_vars, opt_scope='OPT{}'.format(i))
      train_op = tf.group(train_op, d_train_op, g_train_op)#(g_train_op if i == 0 else tf.no_op())])
      # print(train_op)

  # for train_op in tf.get_collection(tf.GraphKeys.TRAIN_OP):
  #   print(train_op)
  print(train_op)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
      # print(i)
      print(sess.run([train_op, 'g_loss:0', 'd_loss:0']))


if __name__ == '__main__':
  tf.app.run()
