"""
Train a single image refinement network using pre-rfn dataset generated using the weights of pre-trained projection network
"""
import tensorflow as tf
from dataset.input_fn import prerfn_input_fn
from model.image_refinement_network import image_refinement_network
from utils.summary import visualize
from utils.rrmse import create_rrmse_metric
import dataset

import os


tf.flags.DEFINE_string('model_dir', None, '')

tf.flags.DEFINE_integer('batch_size', 10, '')
tf.flags.DEFINE_float('learning_rate', 1e-4, '')
tf.flags.DEFINE_float('weight_decay', 1e-4, '')
tf.flags.DEFINE_float('momentum', 0.9, '')

tf.flags.DEFINE_string('gpus', '0', '')
FLAGS = tf.flags.FLAGS


def rfn_model_fn(features, labels, mode):
  inputs = features['prerfn']
  with tf.variable_scope('RFN'):
    outputs = image_refinement_network(inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
  image_labels = labels['image']

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


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  config = tf.estimator.RunConfig(save_checkpoints_secs=1e9,
                                  keep_checkpoint_max=5)
  hooks = [tf.train.LoggingTensorHook(tensors=['rrmse', 'base_rrmse'],every_n_iter=100)]
  estimator = tf.estimator.Estimator(rfn_model_fn, model_dir=FLAGS.model_dir, config=config)

  estimator.train(lambda: prerfn_input_fn('train', batch_size=1, num_epochs=1), hooks=hooks, steps=2000)
  print(
    estimator.evaluate(lambda: prerfn_input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))

  for _ in range(10):
    estimator.train(lambda: prerfn_input_fn('train', batch_size=FLAGS.batch_size, num_epochs=1), hooks=hooks)
    print(
      estimator.evaluate(lambda: prerfn_input_fn('val', batch_size=FLAGS.batch_size, num_epochs=1)))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
