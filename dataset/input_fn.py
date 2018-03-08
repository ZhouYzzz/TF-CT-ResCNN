import tensorflow as tf
import glob, os
from dataset import train_example_spec


def input_fn(mode='train',
             batch_size=16,
             num_epochs=None,
             shuffle=False):
  tfrecord_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '{}*.tfrecords'.format(mode))))
  if len(tfrecord_files) == 0:
    raise(FileNotFoundError('No tfrecords files for mode `{}`'.format(mode)))

  dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)  # type: tf.data.Dataset
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.shuffle(batch_size) if shuffle else dataset
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  example = iterator.get_next()
  features = {'inputs': example['sparse3']}
  labels = example
  return features, labels


def duo_input_fn(mode='train', batch_size=16, num_epochs=None):
  features0, labels0 = input_fn(mode=mode, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
  features1, labels1 = input_fn(mode=mode, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
  return (features0, features1), (labels0, labels1)


if __name__ == '__main__':
  features, labels = duo_input_fn('val')
  print(features)
  print(labels)
