import tensorflow as tf

x = tf.Variable(2)
y = tf.Variable(5)
def f1(): return tf.assign_add(x, 17)
def f2(): return tf.assign_add(y, 23)
r = tf.cond(tf.less(x, y), f1, f2)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(r.eval())
print(x.eval())
print(y.eval())

if __name__ == '__main__':
  pass