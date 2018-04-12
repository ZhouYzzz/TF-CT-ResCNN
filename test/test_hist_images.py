import tensorflow as tf
from dataset.input_fn import prerfn_input_fn


def model_fn(features, labels, mode):
  inputs = features['inputs']
  outputs = labels['image']
  tf.summary.histogram(name='projection', values=inputs)
  tf.summary.histogram(name='image', values=outputs)
  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=tf.zeros(shape=(), dtype=tf.float32),
    train_op=tf.assign_add(tf.train.get_or_create_global_step(), 1)
  )


def main(_):
  estimator = tf.estimator.Estimator(model_fn)
  estimator.train(input_fn=lambda: prerfn_input_fn(num_epochs=1), max_steps=1000)


if __name__ == '__main__':
  tf.app.run()
