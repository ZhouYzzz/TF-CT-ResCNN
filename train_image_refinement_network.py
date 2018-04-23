"""Trains the image refinement network for noise deduction of reconstructed images"""
import tensorflow as tf
import tensorflow.contrib.training as training
import tensorflow.contrib.losses as losses
import tensorflow.contrib.layers as layers
import tensorflow.contrib.gan as gan

from dataset.input_fn import prerfn_input_fn as input_fn
# from dataset.input_fn import sparse_input_fn as input_fn
from model.image_refinement_network import image_refinement_network, image_refinement_network_v2, image_refinement_network_v3
from model.discriminator import discriminator, discriminator_v2, discriminator_v5
from utils.summary import visualize
from utils.rrmse import create_rrmse_metric

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='proposed')
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--pretrain_steps', type=int, default=0)
parser.add_argument('--num_epoches', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip_gradient', type=float, default=1e-4)
parser.add_argument('--use_gan', type=bool, default=False)
parser.add_argument('--gen_lr', type=float, default=1e-4)
parser.add_argument('--diss_lr', type=float, default=1e-4)
parser.add_argument('--scale', type=float, default=50.)

FLAGS = parser.parse_args()


def model_fn(features, labels, mode, params):
  # Define model inputs and labels
  image_inputs = features['prerfn'] * FLAGS.scale  # the raw reconstructed medical image
  image_labels = labels['image'] * FLAGS.scale

  # Define the model
  with tf.variable_scope('Refinement'):
    if FLAGS.model == 'proposed':
      image_outputs = image_refinement_network(image_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    elif FLAGS.model == 'v2':
      image_outputs = image_refinement_network_v2(image_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    elif FLAGS.model == 'v4':
      from model.image_refinement_network import image_refinement_network_v4
      # this model does not work (27%)
      image_outputs = image_refinement_network_v4(image_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    elif FLAGS.model == 'v5':
      from model.image_refinement_network import image_refinement_network_v5
      image_outputs = image_refinement_network_v5(image_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    else:
      raise ValueError('No recognized model named `{}`'.format(FLAGS.model))

  # Define losses
  tf.losses.mean_squared_error(image_outputs, image_labels)
  [tf.losses.add_loss(FLAGS.weight_decay * tf.nn.l2_loss(v), loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) for v in tf.trainable_variables('Projection')]
  loss = tf.losses.get_total_loss()
  image_diff = image_outputs - image_labels

  # Define summaries
  visualize(tf.concat([image_labels, image_inputs, image_outputs], axis=3), name='image')
  visualize(tf.concat(image_diff, axis=3), name='image', use_relu=False)

  # Define metrics
  metric = create_rrmse_metric(image_outputs, image_labels)
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
  image_inputs = features['prerfn'] * 50  # the raw reconstructed medical image
  image_labels = labels['image'] * 50

  def cropped_discriminator(inputs, generator_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN)):
    inputs = tf.random_crop(inputs, size=(FLAGS.batch_size, 1, 64, 64))
    return discriminator_v5(inputs, training)

  # Define the GAN model
  if FLAGS.model == 'proposed':
    generator_fn = lambda x: image_refinement_network(x, training=(mode == tf.estimator.ModeKeys.TRAIN))
  elif FLAGS.model == 'v2':
    generator_fn = lambda x: image_refinement_network_v2(x, training=(mode == tf.estimator.ModeKeys.TRAIN))
  elif FLAGS.model == 'v3':
    # v3 is a 1-layer 0 init model for experiements only
    generator_fn = lambda x: image_refinement_network_v3(x, training=(mode == tf.estimator.ModeKeys.TRAIN))
  elif FLAGS.model == 'v5':
    from model.image_refinement_network import image_refinement_network_v5
    generator_fn = lambda x: image_refinement_network_v5(x, training=(mode == tf.estimator.ModeKeys.TRAIN))
  else:
    raise ValueError('No recognized model named `{}`'.format(FLAGS.model))

  model = gan.gan_model(generator_fn=generator_fn,
                        discriminator_fn=cropped_discriminator,
                        real_data=image_labels,
                        generator_inputs=image_inputs,
                        generator_scope='Refinement',
                        discriminator_scope='Discriminator',
                        check_shapes=True)
  image_outputs = model.generated_data
  # tf.train.init_from_checkpoint('/tmp/GAN', assignment_map={'Refinement/': 'Refinement/'})

  # Define losses
  tf.losses.mean_squared_error(image_outputs, image_labels)
  [tf.losses.add_loss(FLAGS.weight_decay * tf.nn.l2_loss(v), loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) for v in tf.trainable_variables('Refinement')]
  loss = tf.losses.get_total_loss()
  image_diff = image_outputs - image_labels

  # Define summaries
  visualize(tf.concat([image_labels, image_inputs, image_outputs], axis=3), name='image')
  visualize(tf.concat(image_diff, axis=3), name='image', use_relu=False)

  # Define metrics
  metric = create_rrmse_metric(image_outputs, image_labels)
  tf.summary.scalar('rrmse', tf.identity(metric[1], 'rrmse'))

  # train_op = training.create_train_op(
  #   total_loss=loss,
  #   optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
  #   global_step=tf.train.get_or_create_global_step(),
  #   update_ops=None,
  #   variables_to_train=tf.trainable_variables(scope='Refinement'),
  #   transform_grads_fn=training.clip_gradient_norms_fn(max_norm=FLAGS.clip_gradient))

  # Define GAN losses and train_ops
  # with tf.control_dependencies([train_op]):
    # gan.losses
  gan_loss = gan.gan_loss(model,
                          generator_loss_fn=gan.losses.wasserstein_generator_loss,
                          discriminator_loss_fn=gan.losses.wasserstein_discriminator_loss,
                          gradient_penalty_weight=None)
  gan_loss = gan.losses.combine_adversarial_loss(gan_loss,
                                                 gan_model=model,
                                                 non_adversarial_loss=loss,
                                                 weight_factor=1e-3)
  gan_train_ops = gan.gan_train_ops(model,
                                    gan_loss,
                                    generator_optimizer=tf.train.AdamOptimizer(FLAGS.gen_lr),
                                    discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.diss_lr))
  # get_hook_fn = gan.get_sequential_train_hooks(gan.GANTrainSteps(1, 1))
  # gan_train_hooks = get_hook_fn(gan_train_ops)
  # train_op = tf.group(train_op, gan_train_ops.discriminator_train_op, gan_train_ops.generator_train_op)
  #
  train_op = tf.group(gan_train_ops.generator_train_op,
                      gan_train_ops.discriminator_train_op,
                      gan_train_ops.global_step_inc_op)
  with tf.control_dependencies([train_op]):
    clip_ops = tf.group(*[tf.clip_by_value(v, -0.001, 0.001) for v in tf.trainable_variables(scope='Discriminator')])
    train_op = tf.group(train_op, clip_ops)
  return tf.estimator.EstimatorSpec(
    mode,
    predictions={'image_outputs': image_outputs},
    loss=loss,
    train_op=train_op,
    eval_metric_ops={'rrmse': metric},
    training_hooks=None)



def main(_):
  config = tf.estimator.RunConfig(save_checkpoints_secs=1e9)
  if FLAGS.use_gan:
    print('USE GAN')
    estimator = tf.estimator.Estimator(model_fn=gan_model_fn, model_dir=FLAGS.model_dir, config=config)
  else:
    print('NO GAN')
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
