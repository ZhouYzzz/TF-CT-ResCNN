#!/usr/bin/python
"""Script to train a Res-CNN for CT reconstruction (Single Stage)"""
import tensorflow as tf
import os
import glob
import dataset.info as info
from dataset.example_spec import train_example_spec
from model.res_cnn import res_cnn_model
from model.subnet.prj_est_impl import slice_concat
from utils.summary import visualize, statistics

tf.flags.DEFINE_integer('stage', 0, '')

tf.flags.DEFINE_string('data_dir', 'dataset', '')
tf.flags.DEFINE_string('model_dir', '/tmp/ResCNN', '')

tf.flags.DEFINE_integer('batch_size', 10, '')
tf.flags.DEFINE_integer('num_epochs', 5, '')
tf.flags.DEFINE_integer('epochs_per_val', 1, '')

tf.flags.DEFINE_float('learning_rate', 0.01, '')
tf.flags.DEFINE_float('momentum', 0.9, '')
tf.flags.DEFINE_float('weight_decay', 2e-4, '')
tf.flags.DEFINE_float('clip_gradient', 1e-2, '')

tf.flags.DEFINE_string('gpus', '0', '')

FLAGS = tf.flags.FLAGS


def input_fn(is_training, batch_size, num_epochs):
  def _get_filenames(is_training):
    return glob.glob(os.path.join(FLAGS.data_dir, 'train*.tfrecords' if is_training else 'val.tfrecords'))
  record_filenames = _get_filenames(is_training)
  dataset = tf.data.TFRecordDataset(record_filenames)
  if is_training:
    dataset = dataset.repeat(num_epochs)
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return {'inputs': features['sparse3']}, features


def model_fn(features, labels, mode, params):
  graph = tf.get_default_graph()
  inputs = features['inputs']
  predictions = res_cnn_model(inputs, is_training=(mode == tf.estimator.ModeKeys.TRAIN), refinement=False)

  prj_labels = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)
  prj_outputs = graph.get_tensor_by_name('PRJ/outputs:0')
  fbp_outputs = graph.get_tensor_by_name('FBP/outputs:0')
  image_labels = labels['image']
  image_outputs = predictions
  visualize_outputs = tf.zeros_like(image_outputs) if FLAGS.stage == 0 else image_outputs

  prj_tvars = tf.trainable_variables('PRJ')
  # rfn_tvars = tf.trainable_variables('RFN')

  visualize(tf.concat([prj_labels, prj_outputs], axis=2), 'projections')
  visualize(tf.concat([image_labels, fbp_outputs, visualize_outputs], axis=3), 'images')

  # loss
  prj_loss = tf.nn.l2_loss(prj_labels - prj_outputs) / (FLAGS.batch_size * info.PRJ_DEPTH * info.PRJ_HEIGHT * info.PRJ_WIDTH)
  # rfn_loss = tf.nn.l2_loss(image_labels - image_outputs) / (FLAGS.batch_size * 1 * 200 * 200)
  prj_loss += FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in prj_tvars])

  loss = prj_loss
  loss = tf.identity(loss, 'loss')

  # metrics
  image_labels_f = tf.layers.flatten(image_labels)
  image_outputs_f = tf.layers.flatten(fbp_outputs)
  rmse = tf.norm(image_labels_f - image_outputs_f, axis=1) / tf.norm(image_labels_f, axis=1)
  rmse_metrics = tf.metrics.mean(rmse)
  tf.identity(rmse_metrics[1], name='rmse')
  tf.summary.scalar('rmse', rmse_metrics[1])

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(FLAGS.learning_rate)
    learning_rate = tf.identity(learning_rate, 'learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    # train PRJ net
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='PRJ')
    with tf.control_dependencies(update_ops):
      base_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum)
      grads_and_vars = base_optimizer.compute_gradients(prj_loss)
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.clip_gradient), var)
                                for grad, var in grads_and_vars if grad is not None]
      base_train_op = base_optimizer.apply_gradients(clipped_grads_and_vars, global_step)

    train_op = base_train_op
  else:
    train_op = None

  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=predictions,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops={'rmse': rmse_metrics})


def main(unused):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  training_steps_per_epoch = info.NUM_TRAIN // FLAGS.batch_size
  maximum_training_steps = (info.NUM_TRAIN // 1) * 2 + training_steps_per_epoch * FLAGS.num_epochs
  model_dir = os.path.join(FLAGS.model_dir, 'stage{}'.format(FLAGS.stage))

  config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                            save_summary_steps=100,
                                            keep_checkpoint_max=1)
  tensors_to_log = ['loss', 'learning_rate', 'rmse']
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     config=config,
                                     params={})
  for _ in range(FLAGS.num_epochs // FLAGS.epochs_per_val):
    estimator.train(input_fn=lambda: input_fn(True, FLAGS.batch_size, FLAGS.epochs_per_val),
                    hooks=[logging_hook],
                    max_steps=maximum_training_steps)
    eval_results = estimator.evaluate(input_fn=lambda: input_fn(False, FLAGS.batch_size, 1))
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
