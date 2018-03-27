import tensorflow as tf
from dataset.input_fn import prerfn_input_fn
from model.discriminator import discriminator
from model.image_refinement_network import image_refinement_network

# def discriminator(x):
#   return tf.layers.dense(x, 1)


def create_wgan(fake, real, g_vars, opt_scope):
  """
  Create the WGAN given a fake image tensor and a real image tensor
  :param fake: a 4-D image tensor which is generated from some generator
  :param real: a 4-D image tensor which is sampled from natural distribution
  :return: a `train_op` that holds all the train stuffs with a single run
  """
  cropped_fake = tf.map_fn(lambda x: tf.random_crop(x, size=(1, 100, 100)), fake)
  cropped_real = tf.map_fn(lambda x: tf.random_crop(x, size=(1, 100, 100)), real)

  with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
    outputs_fake = discriminator(cropped_fake)

  with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
    outputs_real = discriminator(cropped_real)

  reg_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables(scope='D')])
  g_loss = tf.reduce_mean(outputs_fake) + reg_loss
  d_loss = tf.reduce_mean(outputs_real - outputs_fake) + reg_loss
  g_loss = tf.identity(g_loss, 'g_loss')
  d_loss = tf.identity(d_loss, 'd_loss')

  d_vars = tf.trainable_variables('D')
  # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')):
  with tf.variable_scope(opt_scope):
    g_train_op = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(g_loss, var_list=g_vars)
    d_train_op = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(d_loss, var_list=d_vars)
  return g_train_op, d_train_op


def main(_):
  features, labels = prerfn_input_fn()
  inputs = features['prerfn']
  # a simple model
  with tf.variable_scope('G'):
    # inputs = tf.layers.conv2d(inputs, 1, (3, 3), (1, 1), padding='same', data_format='channels_first')
    # inputs = tf.nn.relu(inputs)
    inputs = image_refinement_network(inputs, training=True)
  labels = labels['image']

  g_vars = tf.trainable_variables('G')

  train_op = tf.no_op()
  for i in range(5):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.TRAIN_OP)):
      g_train_op, d_train_op = create_wgan(inputs, labels, g_vars=g_vars, opt_scope='OPT{}'.format(i))
      train_op = tf.group(train_op, d_train_op, g_train_op)#(g_train_op if i == 0 else tf.no_op())])
      # print(train_op)

  # for train_op in tf.get_collection(tf.GraphKeys.TRAIN_OP):
  #   print(train_op)
  print(train_op)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
      # print(i)
      print(sess.run([train_op, 'g_loss:0', 'd_loss:0']))


if __name__ == '__main__':
  tf.app.run()
