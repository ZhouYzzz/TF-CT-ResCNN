import tensorflow as tf
import glob, os
from dataset import train_example_spec
import dataset.info as info


def input_fn(mode='train',
             batch_size=16,
             num_epochs=None,
             shuffle=False):
  if mode == 'train':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'train_{}.tfrecords'.format(i)) for i in range(10)]
  elif mode == 'val':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'val.tfrecords')]
  else:
    raise ValueError(mode)

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


def prerfn_input_fn(mode='train',
                    batch_size=16,
                    num_epochs=None,
                    shffle=False):
  if mode == 'train':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'train_{}.tfrecords'.format(i)) for i in range(10)]
    attached_tfrecord_files = [os.path.join(os.path.dirname(__file__), 'train_{}.prerfn.tfrecords'.format(i)) for i in range(10)]
  elif mode == 'val':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'val.tfrecords')]
    attached_tfrecord_files = [os.path.join(os.path.dirname(__file__), 'val.prerfn.tfrecords')]
  else:
    raise ValueError(mode)

  dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)  # type: tf.data.Dataset
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  attached_dataset = tf.data.TFRecordDataset(filenames=attached_tfrecord_files)  # type: tf.data.Dataset
  attached_dataset = attached_dataset.map(lambda s: tf.parse_single_example(s, features={'prerfn': tf.FixedLenFeature(shape=(info.IMG_DEPTH, info.IMG_HEIGHT, info.IMG_WIDTH), dtype=tf.float32)}))

  dataset = tf.data.Dataset.zip((dataset, attached_dataset))

  dataset = dataset.repeat(num_epochs)
  if shffle:
    dataset = dataset.shuffle(batch_size)

  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  example, pair = iterator.get_next()
  features = {'inputs': example['sparse3'], 'prerfn': pair['prerfn'], 'image': example['image']}
  labels = example
  return features, labels

def prerfn_input_fn_v2(mode='train',
                    batch_size=16,
                    num_epochs=None,
                    shffle=False):
  if mode == 'train':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'train_{}.tfrecords'.format(i)) for i in range(10)]
    attached_tfrecord_files = [os.path.join(os.path.dirname(__file__), 'train_{}.final.tfrecords'.format(i)) for i in range(10)]
  elif mode == 'val':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'val.tfrecords')]
    attached_tfrecord_files = [os.path.join(os.path.dirname(__file__), 'val.final.tfrecords')]
  else:
    raise ValueError(mode)

  dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)  # type: tf.data.Dataset
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  attached_dataset = tf.data.TFRecordDataset(filenames=attached_tfrecord_files)  # type: tf.data.Dataset
  attached_dataset = attached_dataset.map(lambda s: tf.parse_single_example(s, features={'final': tf.FixedLenFeature(shape=(info.IMG_DEPTH, info.IMG_HEIGHT, info.IMG_WIDTH), dtype=tf.float32)}))

  dataset = tf.data.Dataset.zip((dataset, attached_dataset))

  dataset = dataset.repeat(num_epochs)
  if shffle:
    dataset = dataset.shuffle(batch_size)

  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  example, pair = iterator.get_next()
  features = {'inputs': example['sparse3'], 'prerfn': pair['final']}
  labels = example
  return features, labels


def sparse_input_fn(mode='train',
                    batch_size=16,
                    num_epochs=None,
                    shffle=False):
  if mode == 'train':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'train_{}.tfrecords'.format(i)) for i in range(10)]
    attached_tfrecord_files = [os.path.join(os.path.dirname(__file__), 'train_{}.sparse.tfrecords'.format(i)) for i in range(10)]
  elif mode == 'val':
    tfrecord_files = [os.path.join(os.path.dirname(__file__), 'val.tfrecords')]
    attached_tfrecord_files = [os.path.join(os.path.dirname(__file__), 'val.sparse.tfrecords')]
  else:
    raise ValueError(mode)

  dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)  # type: tf.data.Dataset
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  attached_dataset = tf.data.TFRecordDataset(filenames=attached_tfrecord_files)  # type: tf.data.Dataset
  attached_dataset = attached_dataset.map(lambda s: tf.parse_single_example(s, features={'sparse': tf.FixedLenFeature(shape=(info.IMG_DEPTH, info.IMG_HEIGHT, info.IMG_WIDTH), dtype=tf.float32)}))

  dataset = tf.data.Dataset.zip((dataset, attached_dataset))

  dataset = dataset.repeat(num_epochs)
  if shffle:
    dataset = dataset.shuffle(batch_size)

  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  example, pair = iterator.get_next()
  features = {'inputs': example['sparse3'], 'prerfn': pair['sparse']}
  labels = example
  return features, labels


def duo_input_fn(mode='train', batch_size=16, num_epochs=None):
  features0, labels0 = input_fn(mode=mode, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
  features1, labels1 = input_fn(mode=mode, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
  return (features0, features1), (labels0, labels1)


if __name__ == '__main__':
  # features, labels = duo_input_fn('val')
  # print(features)
  # print(labels)
  print(prerfn_input_fn())
