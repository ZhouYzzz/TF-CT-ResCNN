import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from model.image_refinement_network import image_refinement_network
from model.discriminator import discriminator
from dataset.input_fn import prerfn_input_fn


flags = tf.flags
# tfgan = tf.contrib.gan


flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string(
    'gan_type', 'unconditional',
    'Either `unconditional`, `conditional`, or `infogan`.')

flags.DEFINE_integer(
    'grid_size', 5, 'Grid size for image visualization.')


flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector.')

FLAGS = flags.FLAGS

def _generator_fn(inputs, training=True):
  fn = image_refinement_network(inputs, training=training)
  return fn
  # inputs = tf.layers.conv2d(inputs, 1, (3, 3), (1, 1), padding='same', data_format='channels_first')
  # return inputs
def _discriminator_fn(inputs, training=True):
  CROP_SIZE = 64
  CROP_CHNL = 1
  print(inputs)
  # inputs = tf.map_fn(lambda x: tf.random_crop(x, size=(CROP_CHNL, CROP_SIZE, CROP_SIZE)), inputs)
  # tf.while_loop()
  inputs = tf.random_crop(inputs, size=(10, 1, 100, 100))
  print(inputs)
  return discriminator(inputs, training=training)


def model_fn(features, labels, mode, params):
  image_inputs = features
  image_labels = labels
  gan_model = tfgan.gan_model(generator_fn=_generator_fn, discriminator_fn=_discriminator_fn, real_data=image_labels, generator_inputs=image_inputs)
  gan_loss = tfgan.gan_loss(gan_model, generator_loss_fn=tfgan.losses.wasserstein_generator_loss, discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss, gradient_penalty_weight=1.0)
  train_ops = tfgan.gan_train_ops(gan_model, gan_loss, generator_optimizer=tf.train.AdamOptimizer(1e-4, 0.5), discriminator_optimizer=tf.train.AdamOptimizer(1e-4, 0.5))
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=gan_model.generated_data,
                                    train_op=train_ops.discriminator_train_op,
                                    loss=gan_loss.generator_loss)
# tf.estimator.EstimatorSpec()


def main(_):
  # gan_estimator = tf.estimator.Estimator(model_fn, model_dir=None)
  # tfgan.estimator.GANEstimator()
  # gan_estimator = tfgan.estimator.GANEstimator(
  #   None,
  #   generator_fn=_generator_fn,
  #   discriminator_fn=_discriminator_fn,
  #   generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
  #   discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
  #   generator_optimizer=tf.train.AdamOptimizer(1e-4, 0.5),
  #   discriminator_optimizer=tf.train.AdamOptimizer(1e-4, 0.5))
  def _input_fn():
    features, labels = prerfn_input_fn('train', batch_size=10, num_epochs=1)
    return features['prerfn'] * 50, labels['image'] * 50
  # gan_estimator.train(_input_fn)
  # tf.gradients()



  # with tf.name_scope('inputs'):
  #   features, labels = _input_fn()
  #   # print(features)
  #   # print(labels)
  # gan_model = tfgan.gan_model(generator_fn=_generator_fn, discriminator_fn=_discriminator_fn, real_data=labels,
  #                             generator_inputs=features)
  # gan_loss = tfgan.gan_loss(gan_model, generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
  #                           discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
  #                           gradient_penalty_weight=1.0)
  # # for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
  # #   print(v.name)
  # train_ops = tfgan.gan_train_ops(gan_model, gan_loss,
  #                                 generator_optimizer=tf.train.GradientDescentOptimizer(1e-4),
  #                                 discriminator_optimizer=tf.train.GradientDescentOptimizer(1e-4),
  #                                 summarize_gradients=True,
  #                                 aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
  #                                 )
  # status_message = tf.string_join(
  #     ['Starting train step: ',
  #      tf.as_string(tf.train.get_or_create_global_step())],
  #     name='status_message')
  # # if FLAGS.max_number_of_steps == 0: return
  # tfgan.gan_train(
  #     train_ops,
  #     hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
  #            tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
  #     logdir=FLAGS.train_log_dir,
  #     get_hooks_fn=tfgan.get_joint_train_hooks())

  gan_estimator = tf.estimator.Estimator(model_fn)
  gan_estimator.train(_input_fn)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
