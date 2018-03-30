"""Trains the image refinement network for noise deduction of reconstructed images"""
import tensorflow as tf
import tensorflow.contrib.training as training
import tensorflow.contrib.losses as losses
import tensorflow.contrib.layers as layers
import tensorflow.contrib.gan as gan

from dataset.input_fn import prerfn_input_fn as input_fn
from model.image_refinement_network import image_refinement_network
from model.discriminator import discriminator
from utils.summary import visualize
from utils.rrmse import create_rrmse_metric

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='proposed')
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--pretrain_steps', type=int, default=0)
parser.add_argument('--num_epoches', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip_gradient', type=float, default=1e-4)
parser.add_argument('--use_gan', type=bool, default=False)

FLAGS, _ = parser.parse_known_args()


def model_fn(features, labels, mode, params):
  # Define model inputs and labels
  image_inputs = features['prerfn']  # the raw reconstructed medical image
  image_labels = labels['image']

  # Define the model
  with tf.variable_scope('Refinement'):
    if FLAGS.model == 'proposed':
      image_outputs = image_refinement_network(image_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    else:
      raise ValueError('No recognized model named `{}`'.format(FLAGS.model))

  # Define losses
  tf.losses.mean_squared_error(image_outputs, image_labels)
  [tf.losses.add_loss(FLAGS.weight_decay * tf.nn.l2_loss(v), loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) for v in tf.trainable_variables('Projection')]
  loss = tf.losses.get_total_loss()

  # Define summaries
  visualize(tf.concat([image_labels, image_inputs, image_outputs], axis=3), name='image')

  # Define metrics
  metric = create_rrmse_metric(image_outputs, labels['image'])
  tf.summary.scalar('rrmse', tf.identity(metric[1], 'rrmse'))

  train_op = training.create_train_op(
    total_loss=loss,
    optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
    global_step=tf.train.get_or_create_global_step(),
    update_ops=None,
    variables_to_train=tf.trainable_variables(scope='Refinement'),
    transform_grads_fn=training.clip_gradient_norms_fn(max_norm=FLAGS.clip_gradient))

  return tf.estimator.EstimatorSpec(
    mode,
    predictions={'image_outputs': image_outputs},
    loss=loss,
    train_op=train_op,
    eval_metric_ops={'rrmse': metric})


def gan_model_fn(features, labels, mode, params):
  # Define model inputs and labels
  image_inputs = features['prerfn']  # the raw reconstructed medical image
  image_labels = labels['image']

  # Define the GAN model
  model = gan.gan_model(generator_fn=image_refinement_network,
                        discriminator_fn=discriminator,
                        real_data=image_labels,
                        generator_inputs=image_inputs,
                        generator_scope='Refinement',
                        discriminator_scope='Discriminator',
                        check_shapes=True)
  image_outputs = model.generated_data

  # Define losses
  tf.losses.mean_squared_error(image_outputs, image_labels)
  [tf.losses.add_loss(FLAGS.weight_decay * tf.nn.l2_loss(v), loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) for v in tf.trainable_variables('Projection')]
  loss = tf.losses.get_total_loss()

  # Define GAN losses and train_ops
  gan_loss = gan.gan_loss(model,
                          generator_loss_fn=gan.losses.wasserstein_generator_loss,
                          discriminator_loss_fn=gan.losses.wasserstein_discriminator_loss,
                          gradient_penalty_weight=1.0)
  gan_train_ops = gan.gan_train_ops(model,
                                    gan_loss,
                                    generator_optimizer=tf.train.AdamOptimizer(1e-5),
                                    discriminator_optimizer=tf.train.AdamOptimizer(1e-4))
  get_hook_fn = gan.get_sequential_train_hooks(gan.GANTrainSteps(1, 1))
  gan_train_hooks = get_hook_fn(gan_train_ops)

  # Define summaries
  visualize(tf.concat([image_labels, image_inputs, image_outputs], axis=3), name='image')

  # Define metrics
  metric = create_rrmse_metric(image_outputs, labels['image'])
  tf.summary.scalar('rrmse', tf.identity(metric[1], 'rrmse'))

  train_op = training.create_train_op(
    total_loss=loss,
    optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
    global_step=tf.train.get_or_create_global_step(),
    update_ops=None,
    variables_to_train=tf.trainable_variables(scope='Refinement'),
    transform_grads_fn=training.clip_gradient_norms_fn(max_norm=FLAGS.clip_gradient))

  return tf.estimator.EstimatorSpec(
    mode,
    predictions={'image_outputs': image_outputs},
    loss=loss,
    train_op=train_op,
    eval_metric_ops={'rrmse': metric},
    training_hooks=gan_train_hooks)



def main(_):
  config = tf.estimator.RunConfig(save_checkpoints_secs=1e9)
  if FLAGS.use_gan:
    model_fn = gan_model_fn
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config)

  if FLAGS.pretrain_steps > 1:  # perform pretrain first
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