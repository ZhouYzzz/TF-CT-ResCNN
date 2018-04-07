import tensorflow as tf
import numpy as np
from utils.rrmse import create_rrmse_metric

x = tf.placeholder(tf.float32, shape=(None, 1, 2))
y = tf.placeholder(tf.float32, shape=(None, 1, 2))

loss1 = tf.losses.mean_squared_error(x, y)

loss2 = tf.reduce_mean(tf.map_fn(tf.nn.l2_loss, (x - y)))
loss2 = loss2

rrmse1 = create_rrmse_metric(x, y)
# tf.identity(rrmse1[0], 'rrmse1')
rrmse2 = tf.metrics.root_mean_squared_error(y, x)
rrmse3 = tf.metrics.mean_squared_error(y, x)
rrmse4 = tf.metrics.mean_relative_error(y, x, y)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
for _ in range(1):
  r = sess.run([loss1, loss2, rrmse1[1], rrmse2[1], rrmse3[1], rrmse4[1]], feed_dict={x:np.array([[[4,2]]]), y:np.array([[[2,4]]])})
               # feed_dict={x: np.random.rand(100, 10, 10, 10),
               #            y: np.random.rand(100, 10, 10, 10)})
  print(r)
