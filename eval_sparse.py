#!/usr/bin/python
"""Evaluate the sparse reconstruction using nearest-neiborhood interpolation"""
import tensorflow as tf
import os
from train import input_fn
import logging

tf.flags.DEFINE_integer('stage', 0, '')

tf.flags.DEFINE_string('data_dir', 'dataset', '')
tf.flags.DEFINE_string('model_dir', '/tmp/ResCNN', '')

tf.flags.DEFINE_integer('batch_size', 10, '')
# tf.flags.DEFINE_integer('num_epochs', 5, '')
# tf.flags.DEFINE_integer('epochs_per_val', 1, '')

# tf.flags.DEFINE_float('learning_rate', 0.01, '')
# tf.flags.DEFINE_float('momentum', 0.9, '')
# tf.flags.DEFINE_float('weight_decay', 2e-4, '')
# tf.flags.DEFINE_float('clip_gradient', 1e-2, '')

tf.flags.DEFINE_string('gpus', '0', '')

FLAGS = tf.flags.FLAGS


def model_fn(features, labels, mode, params):
  return None

def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                            save_summary_steps=100,
                                            keep_checkpoint_max=1)
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=FLAGS.model_dir,
                                     config=config,
                                     params={})
  estimator.train(input_fn=lambda: input_fn(False, FLAGS.batch_size, 1), max_steps=1)
  eval_results = estimator.evaluate(input_fn=lambda: input_fn(False, FLAGS.batch_size, 1))
  logging.info(eval_results)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
