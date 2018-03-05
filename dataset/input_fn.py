import tensorflow as tf
import glob, os
from dataset import train_example_spec


def input_fn(mode='train',
             batch_size=16,
             num_epochs=None):
  tfrecord_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '{}*.tfrecords'.format(mode))))
  if len(tfrecord_files) == 0:
    raise(FileNotFoundError('No tfrecords files for mode `{}`'.format(mode)))

  dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)  # type: tf.data.Dataset
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  example = iterator.get_next()
  features = {'inputs': example['sparse3']}
  labels = example
  return features, labels


# if __name__ == '__main__':
#   features, labels = input_fn('train')
#   tf.InteractiveSession()
#
#   for _ in range(10):
#     features['inputs'].eval()
#     labels['image'].eval()
