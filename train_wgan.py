import tensorflow as tf
from dataset import duo_input_fn
from model.subnet.prj_est_impl import slice_concat
from model.subnet.fbp import fbp_subnet
from model.subnet.image_rfn import image_ref_subnet
from resnet.imagenet_main import ImagenetModel


def sparse2full(sparse):
  return slice_concat([sparse for _ in range(5)], axis=3)


def random_crop(images, size, num_crops=5):
  return [tf.map_fn(lambda i: tf.random_crop(i, size=size), images) for _ in range(num_crops)]


def discriminator(inputs, reuse=tf.AUTO_REUSE):
  network = ImagenetModel(resnet_size=18, data_format='channels_first', num_classes=1)
  inputs = network(inputs, training=True)



def model_fn(features, labels, mode):
  inputs = features[0]['inputs']
  inputs = sparse2full(inputs)
  inputs = fbp_subnet(inputs)
  outputs = image_ref_subnet(inputs)  # fake image
  gtimages = labels[0]['images']

  pairimages = labels[1]['images']

  samples_crops_fake = random_crop(outputs, size=(1, 100, 100))
  samples_crops_real = random_crop(pairimages, size=(1, 100, 100))
  ds_fake = [discriminator(s, reuse=tf.AUTO_REUSE) for s in samples_crops_fake]
  ds_real = [discriminator(s, reuse=tf.AUTO_REUSE) for s in samples_crops_real]
  d_losses = [(dr - df) for dr, df in zip(ds_real, ds_fake)]
  opti = tf.train.AdamOptimizer()
  train_ops = tf.group([opti.minimize(dl) for dl in d_losses])
  return tf.estimator.EstimatorSpec()


def main(_):
  x = tf.placeholder(tf.float32, shape=(None, 1, 200, 200))
  # discriminator(x)
  import numpy as np
  dataset = tf.data.Dataset.from_tensor_slices(np.arange(0, 100, 10))  # type: tf.data.Dataset
  it = dataset.make_one_shot_iterator()
  i = it.get_next()

  ii = [tf.add(i, n) for n in range(5)]

  i = i + 9

  sess = tf.InteractiveSession()
  for n in range(5):
    print(sess.run(ii[n]))


if __name__ == '__main__':
  tf.app.run()
