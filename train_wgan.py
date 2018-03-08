import tensorflow as tf
from dataset import duo_input_fn
from model.subnet.prj_est_impl import slice_concat
from model.subnet.fbp import fbp_subnet
from model.subnet.image_rfn import image_ref_subnet
# from resnet.imagenet_main import ImagenetModel
from resnet.cifar10_main import Cifar10Model

def sparse2full(sparse):
  return slice_concat([sparse for _ in range(5)], axis=3)


def random_crop(images, size, num_crops=5):
  return [tf.map_fn(lambda i: tf.random_crop(i, size=size), images) for _ in range(num_crops)]


def discriminator(inputs, reuse=tf.AUTO_REUSE):
  # network = ImagenetModel(resnet_size=18, data_format='channels_first', num_classes=1)
  inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
  with tf.variable_scope('Discriminator', reuse=reuse):
    network = Cifar10Model(resnet_size=20, data_format='channels_first', num_classes=1)
    inputs = network(inputs, training=True)
  return inputs




def model_fn(features, labels, mode):
  inputs = features[0]['inputs']
  inputs = sparse2full(inputs)
  inputs = fbp_subnet(inputs)
  outputs = image_ref_subnet(inputs, is_training=(mode == tf.estimator.ModeKeys.TRAIN))  # fake image
  gtimages = labels[0]['image']

  pairimages = labels[1]['image']

  samples_crops_fake = random_crop(outputs, size=(1, 100, 100))
  samples_crops_real = random_crop(pairimages, size=(1, 100, 100))
  ds_fake = [discriminator(s, reuse=tf.AUTO_REUSE) for s in samples_crops_fake]
  ds_real = [discriminator(s, reuse=tf.AUTO_REUSE) for s in samples_crops_real]
  d_losses = [(dr - df) for dr, df in zip(ds_real, ds_fake)]
  opti = tf.train.AdamOptimizer()
  train_ops = tf.group([opti.minimize(dl) for dl in d_losses])
  # return tf.estimator.EstimatorSpec()
  return train_ops


def main(_):
  duo_features, duo_labels = duo_input_fn('val', batch_size=16, num_epochs=None)
  train_ops = model_fn(duo_features, duo_labels, mode=tf.estimator.ModeKeys.TRAIN)
  with tf.Session() as sess:
    sess.run(train_ops)


if __name__ == '__main__':
  tf.app.run()
