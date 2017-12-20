#!/usr/bin/python
import tensorflow as tf
import os, sys, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
flags = tf.flags
flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_string('data_dir', './dataset', '')
flags.DEFINE_integer('train_epochs', 1, '')
flags.DEFINE_integer('batch_size', 10, '')
flags.DEFINE_integer('epochs_per_eval', 1, '')
FLAGS = flags.FLAGS

# training parameters
_LEARNING_RATE = 1e-2
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_PW = 72
_PH = 216
_PC = 1
_IW = 200
_IH = 200
_IC = 1
_NUM_SAMPLES = {'train': 17120, 'val': 592}

def parse_example(serialized_example):
  features = tf.parse_single_example(
      serialized_example, features={
          'sparse1': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
          'sparse2': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
          'sparse3': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
          'sparse4': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
          'sparse5': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
          'image': tf.FixedLenFeature([_IC,_IH,_IW], tf.float32),
      })
  return features#['sparse1'], features['sparse2'], features['sparse3'], features['sparse4'], features['sparse5'], features['image']

def _FBP_subnet(inputs):
  def _load_weights():
    W = tf.constant(np.fromfile('data/W.bin', np.float64).astype(np.float32), shape=(_PW*5,_PH))
    W = tf.transpose(W)
    F = tf.constant(np.fromfile('data/F.bin', np.float64).astype(np.float32), shape=(_PH,_PH))
    F = tf.transpose(F) # actually this does nothing, since F is symmetric
    Hi = tf.constant(np.fromfile('data/H_indices.bin', np.int64).reshape(-1,2))
    Hv = tf.constant(np.fromfile('data/H_values.bin', np.float64).astype(np.float32))
    H = tf.SparseTensor(Hi, Hv, dense_shape=[40000,77760])
    print H
    return H, F, W
  H, F, W = _load_weights()
  #inputs = tf.cast(inputs, tf.float64)                                # case to double precision
  inputs = tf.reshape(inputs,(-1,216,360))
  inputs = tf.map_fn(lambda x: tf.multiply(W,x), inputs)              # WP
  #tf.summary.image('WP', tf.expand_dims(inputs,axis=-1))
  inputs = tf.map_fn(lambda x: tf.matmul(F,x), inputs)                # FWP
  #tf.summary.image('FWP', tf.transpose(inputs,perm=[0,2,3,1]))
  #tf.summary.image('FWP', tf.expand_dims(inputs,axis=-1))
  #inputs = tf.reshape(tf.transpose(inputs,perm=[0,1,3,2]), shape=(-1,_PH*_PW*5))     # flatten
  inputs = tf.transpose(inputs, perm=[0,2,1])
  inputs = tf.layers.flatten(inputs)
  inputs = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)   # HFWP
  inputs = tf.transpose(inputs)
  inputs = tf.reshape(inputs, shape=(-1,1,_IH,_IW))                 # reshape to (None,1,200,200)
  inputs = tf.transpose(inputs,perm=[0,1,3,2])
  #inputs = tf.cast(inputs, tf.float32)                                # cast to single precision
  inputs = tf.identity(inputs, 'outputs')
  return inputs

def _visualize(name, image_tensor):
  tf.summary.image(name, tf.transpose(image_tensor,perm=[0,2,3,1]),max_outputs=3)
def _summaries(name, var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('{}/summaries'.format(name)):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def _slice_concat(inputs_list, axis):
  sliced_inputs = [None for i in xrange(_PW*5)]
  for i in xrange(5):
    sliced_inputs[i::5] = tf.split(inputs_list[i],_PW,axis=axis)
  inputs = tf.concat(sliced_inputs, axis=axis)
  inputs = tf.identity(inputs, 'projections')
  return inputs

def input_fn(is_training, data_dir, batch_size, num_epochs):
  def _get_filenames(is_training, data_dir):
    if is_training:
      return glob.glob(os.path.join(data_dir, 'train*.tfrecords'))
    else:
      return [os.path.join(data_dir, 'val.tfrecords')]
  record_filenames = _get_filenames(is_training, data_dir)
  dataset = tf.data.TFRecordDataset(record_filenames)
  if is_training:
      dataset = dataset.repeat(num_epochs)
  dataset = dataset.map(parse_example).prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  example = iterator.get_next()
  sparse3 = example['sparse3']
  # sparse1, sparse2, sparse3, sparse4, sparse5, image = iterator.get_next()
  return sparse3, example#{'sparse1':sparse1, 'sparse2':sparse2, 'sparse3': sparse3, 'sparse4': sparse4, 'sparse5': sparse5, 'image': image}

def serving_input_fn():
  serialized_example = tf.placeholder(dtype=tf.string, shape=[], name='input_example')
  receiver_tensors = {'examples': serialized_example}
  features = parse_example(serialized_example)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def model_fn(features, labels, mode, params):
  if mode == tf.estimator.ModeKeys.PREDICT:
    inputs = _slice_concat([features['sparse{}'.format(i+1)] for i in xrange(5)],axis=2)
    inputs = tf.expand_dims(inputs,axis=0)
    outputs = _FBP_subnet(inputs)
    export_output = tf.estimator.export.PredictOutput({'outputs':outputs})
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions={'outputs': outputs}, export_outputs={'export_output': export_output})
  inputs = _slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)],axis=3)
  outputs = _FBP_subnet(inputs)
  _visualize('outputs', outputs)
  _visualize('image', labels['image'])
  _visualize('diff', outputs-labels['image'])
  loss = tf.nn.l2_loss(outputs-labels['image'])
  images_flatten = tf.layers.flatten(labels['image'])
  outputs_flatten = tf.layers.flatten(outputs)
  rmse = tf.norm(outputs_flatten - images_flatten, axis=1) / tf.norm(images_flatten, axis=1)
  rmse_metrics = tf.metrics.mean(rmse)
  global_step = tf.train.get_or_create_global_step()
  train_op = tf.assign_add(global_step,1)
  return tf.estimator.EstimatorSpec(
      mode=mode, predictions={'outputs': outputs}, loss=loss, train_op=train_op, eval_metric_ops={'rmse':rmse_metrics})

def main(_):
  if FLAGS.model_dir is None:
    from time import time
    FLAGS.model_dir = '/tmp/CT/model_{}'.format(int(time()))
    print 'Using temp model_dir:', FLAGS.model_dir
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params={'pretrain': True})
  tensors_to_log = []
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
  #estimator.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, 1), hooks=[logging_hook])
  #estimator.evaluate(input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size, 1))
  export_dir = estimator.export_savedmodel(FLAGS.model_dir, serving_input_fn)
  with tf.Session(graph=tf.Graph()) as sess:
    tfqueue = tf.train.string_input_producer([os.path.join(FLAGS.data_dir,'val.tfrecords')])
    _, serialized_example = tf.TFRecordReader().read(tfqueue)
    tf.saved_model.loader.load(sess, ['serve'], export_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    outputs = sess.graph.get_tensor_by_name('outputs:0')
    input_example = sess.graph.get_tensor_by_name('input_example:0')
    string = sess.run(serialized_example)
    print sess.run(outputs, feed_dict={input_example: string})
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
