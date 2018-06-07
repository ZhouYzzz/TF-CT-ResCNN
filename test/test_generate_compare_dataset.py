from dataset.input_fn import prerfn_input_fn as input_fn
import tensorflow as tf
import argparse
import numpy as np
import scipy.misc
import os


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--identifier', type=str, default=None)

FLAGS = parser.parse_args('--model_dir /home/zhouyz/Development/v6_L1_conv_gan_lsq_0.01 '
                          '--identifier gan001'.split(' '))


def reference_model_fn(features, labels, mode):
  outputs = features['image']
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=outputs
  )


def v6_model_fn(features, labels, mode):
  image_inputs = features['prerfn']
  with tf.variable_scope('Refinement'):
    from model.image_refinement_network import image_refinement_network_v6
    image_outputs = image_refinement_network_v6(image_inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
    image_outputs = tf.nn.relu(image_outputs)
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=image_outputs
  )


def normalize(imagef: np.ndarray):
  imagef.reshape(-1)
  imagef = imagef / np.max(imagef) * 255
  return imagef.astype(np.uint8)


def main():
  if not os.path.exists(os.path.join('database', FLAGS.identifier)):
    os.mkdir(os.path.join('database', FLAGS.identifier))
  model_fn = v6_model_fn
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)
  outputs = estimator.predict(input_fn=lambda: input_fn(mode='val', batch_size=1, num_epochs=1))
  for i, output in enumerate(outputs):
    scipy.misc.imsave(os.path.join('database', FLAGS.identifier, '{:04d}.jpg'.format(i)),
                      normalize(imagef=output).reshape(200, 200))


if __name__ == '__main__':
  main()
