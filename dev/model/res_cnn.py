import tensorflow as tf
from prj_est_subnet import prj_est_subnet, slice_concat
from fbp_subnet import fbp_subnet
from image_ref_subnet import image_ref_subnet

def res_cnn_model(inputs, is_training):
  with tf.variable_scope('PRJ'):
    inputs = prj_est_subnet(inputs, is_training)
  with tf.variable_scope('FBP'):
    inputs = fbp_subnet(inputs)
  with tf.variable_scope('RFN'):
    inputs = image_ref_subnet(inputs, is_training)
  return inputs
