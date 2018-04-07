import tensorflow as tf
from model.projection_estimation_network import *
from dataset.input_fn import input_fn

sess = tf.Session()

features, _ = input_fn()

with tf.variable_scope('pad'):
  pad = projection_estimation_network(features['inputs'], training=True)

with tf.variable_scope('no_pad'):
  no_pad = projection_estimation_network_no_padding(features['inputs'], training=True)

import time

sess.run(tf.global_variables_initializer())

t = time.time()
for _ in range(100):
  sess.run(pad)
print(time.time() - t)

t = time.time()
for _ in range(100):
  sess.run(no_pad)
print(time.time() - t)
