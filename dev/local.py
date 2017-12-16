import tensorflow as tf

def main(_):
  tf.reset_default_graph()
  graph = tf.get_default_graph()
  inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='images')
  inputs = tf.layers.conv2d(inputs,64,(3,3),padding='same',use_bias=False)
  print inputs.name
  with tf.name_scope('block1'):
    inputs = tf.layers.conv2d(inputs,64,(3,3),padding='same',use_bias=False)
    print inputs.name
  with tf.variable_scope('train1'):
    with tf.variable_scope('branch1'):
      inputs = tf.layers.conv2d(inputs, 64, (3, 3), padding='same', use_bias=False)
      inputs = tf.identity(inputs, 'outputs')
      print inputs.name
    with tf.variable_scope('branch2'):
      inputs = tf.layers.conv2d(inputs, 64, (3, 3), padding='same', use_bias=False)
      inputs = tf.identity(inputs, 'outputs')
      print inputs.name

  print tf.global_variables()
  print graph.get_tensor_by_name('images:0')
  print graph.get_tensor_by_name('train1/branch1/outputs:0')
  print tf.trainable_variables('train1')
  print tf.get_variable_scope().local_variables()
  # tf.estimator.EstimatorSpec()

if __name__ == '__main__':
  tf.app.run()
