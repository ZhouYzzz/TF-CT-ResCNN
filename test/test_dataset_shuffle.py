"""
Test the behavior of `Dataset.shuffle`

In most cases, the `buffer_size` param should be filled with `batch_size`
"""
import tensorflow as tf

def main():
  dataset = tf.data.Dataset.from_tensor_slices(list(range(100)))  # type: tf.data.Dataset
  dataset = dataset.shuffle(10)  # `shuffle` will create a new dataset containing the next `buffer_size` elements, then randomly sample one batch from the two dataset
  dataset = dataset.batch(10)

  iterator = dataset.make_one_shot_iterator()
  example = iterator.get_next()

  tf.InteractiveSession()

  for _ in range(10):
    print(example.eval())

if __name__ == '__main__':
  main()