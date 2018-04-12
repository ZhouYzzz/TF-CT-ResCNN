from dataset.input_fn import input_fn
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
  features, labels = input_fn(batch_size=1)
  sess = tf.InteractiveSession()
  r = (features['inputs'].eval())
  print(np.mean(r), np.max(r))
  r = (labels['image'].eval() * 50)
  print(np.mean(r), np.max(r))
  tf.summary.image()
