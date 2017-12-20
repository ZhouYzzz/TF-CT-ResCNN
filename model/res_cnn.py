"""
Define the full res_cnn model function
"""
import tensorflow as tf
from model.subnet.fbp import fbp_subnet
from model.subnet.image_rfn import image_rfn_subnet


VERSION_PROPOSED = 'PROPOSED'


def res_cnn_model(inputs, is_training=False, refinement=True, version=VERSION_PROPOSED):
  if version == VERSION_PROPOSED:
    from model.subnet.prj_est_proposed import prj_est_subnet
  else:
    raise ValueError('Undefined Res-CNN version: {0}'.format(version))

  with tf.variable_scope('PRJ'):
    inputs = prj_est_subnet(inputs, is_training)
  with tf.variable_scope('FBP'):
    inputs = fbp_subnet(inputs)
  if refinement:
    with tf.variable_scope('RFN'):
      inputs = image_rfn_subnet(inputs, is_training)
  return inputs

