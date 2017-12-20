import tensorflow as tf
from prj_est_subnet_impl import prj_est_subnet_core_v1 as prj_est_subnet_core_fn

def slice_concat(inputs_list, axis):
  sliced_inputs = [None for i in xrange(72 * 5)]
  for i in xrange(5):
    sliced_inputs[i::5] = tf.split(inputs_list[i], 72, axis=axis)
  inputs = tf.concat(sliced_inputs, axis=axis)
  return inputs

def prj_est_subnet(inputs, is_training):
  branch_outputs = [None for _ in range(5)]
  for i in range(5):
    branch_outputs[i] = prj_est_subnet_core_fn(inputs, i, is_training)
  inputs = slice_concat(branch_outputs, axis=3)
  inputs = tf.identity(inputs, 'outputs')
  return inputs
