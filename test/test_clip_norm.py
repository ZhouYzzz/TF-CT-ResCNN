import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

a = tf.constant(np.random.rand(10))
b = tf.constant(np.random.rand(10) * 10)
n = tf.global_norm([a])
print(n.eval())
a = tf.clip_by_norm(a, 1)
print(a)
print(a.eval())

n = tf.global_norm([a])
print(n.eval())
