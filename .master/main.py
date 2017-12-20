#!/usr/bin/python
import tensorflow as tf
import os, sys, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
flags = tf.flags
flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_string('data_dir', './dataset', '')
flags.DEFINE_integer('train_epochs', 4, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_integer('epochs_per_eval', 1, '')
FLAGS = flags.FLAGS

# training parameters
_LEARNING_RATE = 1e-3
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_PC = 1; _PH = 216; _PW = 72
_IC = 1; _IH = 200; _IW = 200
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
  labels = iterator.get_next()
  #sparse1, sparse2, sparse3, sparse4, sparse5, image = iterator.get_next()
  return {'inputs': labels['sparse3']}, labels#{'sparse1':sparse1, 'sparse2':sparse2, 'sparse3': sparse3, 'sparse4': sparse4, 'sparse5': sparse5, 'image': image}

def _branch_subnet(inputs, index=0, is_training=False):
  def _batch_norm_relu(inputs, is_training):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs
  def _conv_padding(inputs, filters, kernel_size, strides):
    def _period_padding(inputs, kernel_size):
      wpad_size = (kernel_size[1] - 1) // 2
      hpad_size = (kernel_size[0] - 1) // 2
      # perform periodic padding along width
      inputs = tf.concat([inputs[:,:,:,slice(-1,-1-wpad_size,-1)], inputs, inputs[:,:,:,slice(0,wpad_size,1)]],axis=3)
      # perform zero padding along height
      inputs = tf.pad(inputs, [[0,0],[0,0],[hpad_size,hpad_size],[0,0]],)
      return inputs
    inputs = _period_padding(inputs, kernel_size)
    inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding='valid', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format='channels_first')
    return inputs
  with tf.name_scope('B{}'.format(index)):
    inputs = _conv_padding(inputs, 64, (9,3), (1,1))
    inputs = _batch_norm_relu(inputs, is_training)
    inputs = _conv_padding(inputs, 64, (9,3), (1,1))
    inputs = _batch_norm_relu(inputs, is_training)
    shortcuts = inputs
    inputs = _conv_padding(inputs, 64, (9,3), (1,1))
    inputs = _batch_norm_relu(inputs, is_training)
    inputs = shortcuts + inputs
    inputs = _conv_padding(inputs, 64, (9,3), (1,1))
    inputs = _batch_norm_relu(inputs, is_training)
    shortcuts = inputs
    inputs = _conv_padding(inputs, 64, (9,3), (1,1))
    inputs = _batch_norm_relu(inputs, is_training)
    inputs = shortcuts + inputs
    inputs = _conv_padding(inputs, 16, (9,3), (1,1))
    inputs = _batch_norm_relu(inputs, is_training)
    inputs = _conv_padding(inputs, 1, (9,3), (1,1))
    inputs = tf.identity(inputs, 'outputs')
    return inputs

def _slice_concat(inputs_list, axis):
  sliced_inputs = [None for i in xrange(_PW*5)]
  for i in xrange(5):
    sliced_inputs[i::5] = tf.split(inputs_list[i],_PW,axis=axis)
  inputs = tf.concat(sliced_inputs, axis=axis)
  inputs = tf.identity(inputs, 'projections')
  return inputs

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
  inputs = tf.reshape(inputs,(-1,216,360))
  inputs = tf.map_fn(lambda x: tf.multiply(W,x), inputs)              # WP
  inputs = tf.map_fn(lambda x: tf.matmul(F,x), inputs)                # FWP
  inputs = tf.transpose(inputs, perm=[0,2,1])
  inputs = tf.layers.flatten(inputs)
  inputs = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)   # HFWP
  inputs = tf.transpose(inputs)
  inputs = tf.reshape(inputs, shape=(-1,1,_IH,_IW))
  inputs = tf.transpose(inputs,perm=[0,1,3,2])
  inputs = tf.identity(inputs, 'outputs')
  return inputs

def _refinement_subnet(inputs, is_training=False):
  def _batch_norm_relu(inputs, is_training):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs
  def _conv_padding(inputs, filters, kernel_size, strides):
    inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding='same', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format='channels_first')
    return inputs
  inputs = _conv_padding(inputs, 64, (7,7), (1,1))
  inputs = _batch_norm_relu(inputs, is_training)
  inputs = _conv_padding(inputs, 64, (7,7), (1,1))
  inputs = _batch_norm_relu(inputs, is_training)
  shortcuts = inputs
  inputs = _conv_padding(inputs, 64, (7,7), (1,1))
  inputs = _batch_norm_relu(inputs, is_training)
  inputs = shortcuts + inputs
  inputs = _conv_padding(inputs, 64, (7,7), (1,1))
  inputs = _batch_norm_relu(inputs, is_training)
  shortcuts = inputs
  inputs = _conv_padding(inputs, 64, (7,7), (1,1))
  inputs = _batch_norm_relu(inputs, is_training)
  inputs = shortcuts + inputs
  inputs = _conv_padding(inputs, 16, (7,7), (1,1))
  inputs = _batch_norm_relu(inputs, is_training)
  inputs = _conv_padding(inputs, 1, (7,7), (1,1))
  inputs = tf.identity(inputs, 'outputs')
  return inputs

def model(inputs, is_training=False, pretrain=False):
  #inputs = tf.identity(inputs, 'inputs')
  with tf.name_scope('PRJ'), tf.variable_scope(''):
    inputs = [_branch_subnet(inputs, i, is_training=is_training) for i in range(5)]
    inputs = _slice_concat(inputs, axis=3)
  with tf.name_scope('FBP'):
    inputs = _FBP_subnet(inputs)
  with tf.name_scope('RFN'), tf.variable_scope(''):
    inputs = _refinement_subnet(inputs)
  return inputs

def _visualize(name, image_tensor):
  tf.summary.image(name, tf.transpose(image_tensor,perm=[0,2,3,1]),max_outputs=1)
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

def model_fn(features, labels, mode, params):
  # get inputs and regression targets
  inputs = features['inputs'] # sparse3
  # targets = labels  # sparse1

  # construct model
  outputs = model(inputs, is_training=True)

  if params['stage'] == 0:
    tvars = tf.trainable_variables('PRJ')
  elif params['stage'] == 1:
    tvars = tf.trainable_variables('RFN')
  elif params['stage'] == 2:
    tvars = tf.trainable_variables()
  else:
    raise ValueE

  # PREDICT SPEC
  if mode == tf.estimator.ModeKeys.PREDICT:
    export_output = tf.estimator.export.PredictOutput({'outputs':outputs})
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions={'outputs': outputs}, export_outputs={'export_output': export_output})

  graph = tf.get_default_graph()
  projections = graph.get_tensor_by_name('PRJ/projections:0')
  FBP_outputs = graph.get_tensor_by_name('FBP/outputs:0')
  # outputs: refinement outputs
  images = labels['image']

  # labels concat
  labels_inputs = _slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)
  labels_outputs = _FBP_subnet(labels_inputs)

  loss = 0
  for i in range(5):
    branch_outputs = graph.get_tensor_by_name('PRJ/B{}/outputs:0'.format(i))
    branch_targets = labels['sparse{}'.format(i+1)]
    branch_diff = branch_outputs - branch_targets
    loss += tf.nn.l2_loss(branch_diff) / (FLAGS.batch_size*_PC*_PH*_PW)
  #loss = loss / 5
  loss = tf.identity(loss, 'PRJ_loss')
  tf.summary.scalar('PRJ_loss', loss)
  if not params['pretrain']:
    images_diff = outputs - images
    loss += tf.nn.l2_loss(images_diff) / (FLAGS.batch_size*_IC*_IH*_IW)
    loss = tf.identity(loss, 'PRJ_RFN_loss')
    tf.summary.scalar('PRJ_RFN_loss', loss)

    outputs_flatten = tf.layers.flatten(outputs)
    _visualize('compare_refinement', tf.concat([images, outputs], axis=3))
  else:
    outputs_flatten = tf.layers.flatten(FBP_outputs)
    _visualize('compare_FBP_out', tf.concat([images, labels_outputs, FBP_outputs], axis=3))
    _visualize('compare_PRJ_out', tf.concat([labels_inputs, projections], axis=2))

  loss += _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tvars])
  loss = tf.identity(loss, 'loss')
  tf.summary.scalar('loss', loss)
  images_flatten = tf.layers.flatten(images)
  rmse = tf.norm(outputs_flatten - images_flatten, axis=1) / tf.norm(images_flatten, axis=1)
  rmse_metrics = tf.metrics.mean(rmse)
  # projections_labels = _slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=2)
  # rmse = tf.square(tf.layers.flatten(projections - projections_labels)) / tf.square(tf.layers.flatten(projections_labels))
  # rmse_metrics = tf.metrics.mean(rmse)
  # _visualize('projections', projections)
  # _visualize('projections_labels', projections_labels)
  # add summaries
  # _visualize('net_diff', net_diff)
  # _visualize('diff', diff)
  # _visualize('targets_vs_outputs', tf.concat([targets, outputs], axis=3))
  # _summaries('net_diff', net_diff)
  # _summaries('diff', diff)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    batches_per_epoch = _NUM_SAMPLES['train'] // FLAGS.batch_size
    print 'Batches per epoch:', batches_per_epoch
    boundaries = [batches_per_epoch * epoch for epoch in [2, 3]]
    lr_values = [_LEARNING_RATE * decay for decay in [1, 0.1, 0.01]]
    lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, lr_values)
    tf.identity(lr, name='learning_rate')
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.MomentumOptimizer(lr, _MOMENTUM)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1e-2), var) for grad, var in grads_and_vars if grad is not None]
      train_op = optimizer.apply_gradients(grads_and_vars, global_step)
  else:
    train_op = None

  return tf.estimator.EstimatorSpec(
      mode=mode, predictions={'outputs': projections}, loss=loss, train_op=train_op, eval_metric_ops={'rmse':rmse_metrics})

def main(_):
  if FLAGS.model_dir is None:
    from time import time
    FLAGS.model_dir = '/tmp/CT/model_{}'.format(int(time()))
    print 'Using temp model_dir:', FLAGS.model_dir
  # TRAINING STAGE 1
  #estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params={'pretrain': True})
  #for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
  #  tensors_to_log = ['loss', 'learning_rate']
  #  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
  #  estimator.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval), hooks=[logging_hook])
  #  estimator.evaluate(input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size, 1))
  # TRAINING STAGE 2
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params={'pretrain': False})
  for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = ['loss', 'learning_rate']
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
    tf.reset_default_graph()
    estimator.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval), hooks=[logging_hook])
    tf.reset_default_graph()
    estimator.evaluate(input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size, 1))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

