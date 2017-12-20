import tensorflow as tf
from image_ref_subnet_impl import image_ref_subnet_core_v1 as image_ref_subnet_core_fn


def image_ref_subnet(inputs, is_training):
  inputs = image_ref_subnet_core_fn(inputs, is_training)
  inputs = tf.identity(inputs, 'outputs')
  return inputs
