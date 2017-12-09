#!/usr/bin/python
"""
An estimator for test purpose to tune the PRJ branch network
"""
import tensorflow as tf
flags = tf.flags
flags.DEFINE_string('data_dir', './dataset', '')
flags.DEFINE_string('model_dir', './model', '')
flags.DEFINE_integer('train_epochs', 100, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('epochs_per_eval', 20, '')
# flags.DEFINE_string('data_format', 'channels_first', '')
FLAGS = flags.FLAGS
import os, sys, glob
from resnet_model import conv2d_fixed_padding, batch_norm_relu

_LEARNING_RATE = 1e-1
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

def variable_summaries(name, var):
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

def input_fn(is_training, data_dir, batch_size, num_epochs):
    def _get_filenames(is_training, data_dir):
        if is_training:
            return glob.glob(os.path.join(data_dir, 'train*.tfrecords'))
        else:
            return [os.path.join(data_dir, 'val.tfrecords')]
    def _parse_example(serialized_example):
        features = tf.parse_single_example(
                serialized_example, features={
                    'sparse1': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                    # 'sparse2': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                    'sparse3': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                    # 'sparse4': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                    # 'sparse5': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                    # 'image': tf.FixedLenFeature([_IC,_IH,_IW], tf.float32)
                })
        return features['sparse3'], features['sparse1']
    record_filenames = _get_filenames(is_training, data_dir)
    dataset = tf.data.TFRecordDataset(record_filenames)
    if is_training:
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse_example).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    sparse3, sparse1 = iterator.get_next()
    return sparse3, sparse1

# conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, transpose=False)
# batch_norm_relu(inputs, is_training, data_format)
def model(inputs, data_format='channels_first'):
    inputs = conv2d_fixed_padding(inputs, 32, (9,3), (1,1), data_format)
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = conv2d_fixed_padding(inputs, 32, (9,3), (1,1), data_format)
    inputs = batch_norm_relu(inputs, True, data_format)
    shortcut = inputs
    inputs = conv2d_fixed_padding(inputs, 32, (9,3), (1,1), data_format)
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = inputs + shortcut
    inputs = conv2d_fixed_padding(inputs, 32, (9,3), (1,1), data_format)
    inputs = batch_norm_relu(inputs, True, data_format)
    shortcut = inputs
    inputs = conv2d_fixed_padding(inputs, 32, (9,3), (1,1), data_format)
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = inputs + shortcut
    inputs = conv2d_fixed_padding(inputs, 16, (9,3), (1,1), data_format)
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = conv2d_fixed_padding(inputs, 1, (9,3), (1,1), data_format)
    return inputs

def train_model_fn(features, labels, mode, params):
    assert mode == tf.estimator.ModeKeys.TRAIN
    tf.summary.image('inputs', features)
    outputs = model(features)
    diff = labels - outputs
    tf.summary.image('diff', diff)
    variable_summaries('diff', diff)
    diff_loss = tf.nn.l2_loss(diff) / FLAGS.batch_size
    tf.summary.scalar('diff_loss', diff_loss)
    reg_loss = _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    tf.summary.scalar('reg_loss', reg_loss)
    loss = diff_loss + reg_loss
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(
            learning_rate=_LEARNING_RATE,
            momentum=_MOMENTUM)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

def main(_):
    run_config = tf.estimator.RunConfig()

    estimator = tf.estimator.Estimator(
            model_fn=train_model_fn, model_dir=FLAGS.model_dir, config=run_config, params={})
    for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        print 'CIRCLE', i
        tensors_to_log = {'prj_loss': 'prj_loss'}
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=10)
        estimator.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval), hooks=[logging_hook])

if __name__ == '__main__':
    tf.app.run()