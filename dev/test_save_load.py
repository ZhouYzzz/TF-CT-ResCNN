import tensorflow as tf


def model(inputs, is_training):
  inputs = tf.layers.batch_normalization(inputs,axis=1,training=is_training)
  inputs = tf.nn.relu(inputs)
  return inputs

def model_fn(features, labels, mode, params):
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=outputs,
    loss=loss,
    train_op=train_op
  )


def main(_):
  inputs = tf.placeholder(tf.float32, shape=(10,32,32,3))
  inputs = model(inputs, True)
  print tf.global_variables()
  print tf.trainable_variables()
  print tf.local_variables()

if __name__ == '__main__':
  tf.app.run()
