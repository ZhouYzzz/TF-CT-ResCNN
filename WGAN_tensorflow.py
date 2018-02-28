import sys, os

import tensorflow as tf
import tensorflow.contrib.layers as ly

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.flags
FLAGS = flags.FLAGS

def sample(n):
  z = tf.random_uniform([n,FLAGS.z_dim])
  h = generator(z, reuse=True)
  tf.summary.image('image', h, max_outputs=n)

def lrelu(x, leak=0.3, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
  return f1 * x + f2 * abs(x)

def generator(z, reuse=False):
  nch = 512
  with tf.variable_scope('G', reuse=reuse):
    h = tf.layers.dense(z, 3*3*nch, kernel_initializer=tf.random_normal_initializer())
    h = lrelu(h)
    h = tf.reshape(h, [-1,3,3,nch])
    h = tf.layers.conv2d_transpose(h, nch/2, 3, strides=2,
                  padding='valid', activation=lrelu)
    h = tf.layers.conv2d_transpose(h, nch/4, 3, strides=2,
                  padding='same', activation=lrelu)
    h = tf.layers.conv2d_transpose(h, nch/8, 3, strides=2,
                  padding='same')
    h = tf.layers.conv2d(h, 1, 1)
    h = tf.sigmoid(h)
  return h

def discriminator(h, reuse=False):
  with tf.variable_scope('D', reuse=reuse):
    size = 64
    h = ly.conv2d(h, num_outputs=size, kernel_size=3,
                  stride=2, activation_fn=lrelu)
    h = ly.conv2d(h, num_outputs=size * 2, kernel_size=3,
                  stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                  normalizer_params={'is_training':True})
    h = ly.conv2d(h, num_outputs=size * 4, kernel_size=3,
                  stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm, 
                  normalizer_params={'is_training':True})

    h = ly.conv2d(h, num_outputs=size * 8, kernel_size=3,
                  stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                  normalizer_params={'is_training':True})
    h = ly.fully_connected(tf.reshape(
                  h, [FLAGS.batch_size, -1]), 1, activation_fn=None)
  return h

def build_graph():
  X = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,28*28))
  X_ = tf.reshape(X, [-1,28,28,1])
  # X_ = tf.expand_dims(X, axis=-1)
  Z = tf.random_uniform([FLAGS.batch_size,FLAGS.z_dim])

  D_real = discriminator(X_)
  D_fake = discriminator(generator(Z), reuse=True)

  sample(8) # for visualization

  D_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D')
  G_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')

  D_loss = tf.reduce_mean(D_fake - D_real)
  G_loss = tf.reduce_mean(- D_fake)

  tf.summary.scalar('D_loss', D_loss)
  tf.summary.scalar('G_loss', G_loss)

  D_solver = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(D_loss, var_list=D_weights)
  G_solver = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(G_loss, var_list=G_weights)
  
  clip_weights = [tf.clip_by_value(w, -0.01, 0.01) for w in D_weights]

  return X, D_solver, G_solver, clip_weights, D_loss

def main(_):
  mnist = input_data.read_data_sets(FLAGS.input_data_dir)

  with tf.device('/%cpu:%d'%(('g' if FLAGS.gpu else 'c'), FLAGS.device)):
    X, D_s, G_s, clip, D_l = build_graph()
  
  config = tf.ConfigProto(allow_soft_placement = True)
  global_step = tf.Variable(0, trainable=False, name='global_step')
  increment_global_step_op = tf.assign(global_step, global_step+1)
  sess = tf.Session(config=config)
  summary = tf.summary.merge_all()
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

  sess.run(init)

  if FLAGS.restore:
    try:
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))
    except:
      print 'RESTORE ERROR'

  step = tf.train.global_step(sess, global_step)

  print 'START TRAINING'
  while step < FLAGS.max_steps:
    for _ in xrange(5):
      feed_dict = {X: mnist.train.next_batch(FLAGS.batch_size)[0]}
      sess.run(D_s, feed_dict=feed_dict)
      sess.run(clip)
    sess.run(G_s)

    if step % 100 == 0:
      summary_str, loss = sess.run([summary, D_l], feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()
      print 'STEP', step, '\tDLOSS', loss

    if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
      checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
      saver.save(sess, checkpoint_file, global_step=step)

    sess.run(increment_global_step_op)
    step = tf.train.global_step(sess, global_step)

if __name__ == '__main__':
  flags.DEFINE_bool(
    'gpu', False, '')
  flags.DEFINE_integer(
    'device', 0, '')
  flags.DEFINE_bool(
    'restore', False, '')
  flags.DEFINE_integer(
    'batch_size', 64, '')
  flags.DEFINE_integer(
    'z_dim', 100, '')
  flags.DEFINE_integer(
    'max_steps', 50000, '')
  flags.DEFINE_float(
    'learning_rate', 1e-4, '')
  flags.DEFINE_string(
    'input_data_dir', '/tmp/tensorflow/mnist/input_data', '')
  flags.DEFINE_string(
    'log_dir', '/tmp/tensorflow/WGAN', '')
  sys.argv.extend([])
  tf.app.run(main)
