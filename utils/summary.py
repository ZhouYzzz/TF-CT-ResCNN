import tensorflow as tf


def visualize(t, name, perm=True, max_outputs=2):
  return tf.summary.image(name, tf.transpose(t, perm=(0,2,3,1)) if perm else t, max_outputs=max_outputs)


def statistics(t, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('{}/summaries'.format(name)):
    mean = tf.reduce_mean(t)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(t - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(t))
    tf.summary.scalar('min', tf.reduce_min(t))
    tf.summary.histogram('histogram', t)
