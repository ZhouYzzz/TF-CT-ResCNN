import tensorflow as tf
# from

def main(_):

  # print(example)

  with tf.Session() as sess:
    example = tf.placeholder(tf.string, shape=(None,))
    tf.saved_model.loader.load(sess, ['serve'],
                               '../tmpouizb6k5_projection_estimation_network/1521426198',
                               input_map={'input_example_tensor:0': example})
    print(tf.get_default_graph().get_tensor_by_name('Branch1/outputs:0'))
    print(tf.get_default_graph().get_tensor_by_name('input_example_tensor:0'))
    outputs = tf.get_default_graph().get_tensor_by_name('FBP/outputs:0')
    # print(tf.get_default_graph().get_tensor_by_name('inputs:0'))

    for i in range(10):
      dataset = tf.data.TFRecordDataset(['../dataset/train_{}.tfrecords'.format(i)])
      dataset = dataset.repeat(1)
      dataset = dataset.prefetch(100)
      dataset = dataset.batch(10)
      iterator = dataset.make_one_shot_iterator()
      ss = iterator.get_next()

      try:
        count = 0
        while True:
          serialized = sess.run(ss)
          results = sess.run(outputs, feed_dict={example: serialized})
          count += 1
          print(count)
      except:
        pass
      finally:
        pass


if __name__ == '__main__':
  tf.app.run()
