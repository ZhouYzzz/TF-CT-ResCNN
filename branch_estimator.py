#!/usr/bin/python
"""
An estimator for test purpose to tune the PRJ branch network
"""
import tensorflow as tf
flags = tf.flags
flags.DEFINE_string('data_dir', './dataset', '')
flags.DEFINE_string('model_dir', './model', '')
flags.DEFINE_integer('train_epochs', 210, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('epochs_per_eval', 10, '')
# flags.DEFINE_string('data_format', 'channels_first', '')
FLAGS = flags.FLAGS
import os, sys, glob
from resnet_model import conv2d_fixed_padding, batch_norm_relu

_LEARNING_RATE = 1e-1
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9
_PW = 72
_PH = 216
_PC = 1
_IW = 200
_IH = 200
_IC = 1
_NUM_SAMPLES = {
        'train': 17120,
        'val': 592
        }

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
    dataset = dataset.map(_parse_example).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    sparse3, sparse1 = iterator.get_next()
    return sparse3, sparse1

# conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, transpose=False)
# batch_norm_relu(inputs, is_training, data_format)
def model(inputs, data_format='channels_first'):
    print 'Using model: 1'
    shortcut0 = inputs
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
    inputs = inputs + shortcut0
    return inputs

def model2(inputs, data_format='channels_first'):
    shortcut0 = inputs
    inputs = conv2d_fixed_padding(inputs, 32, (5,5), (2,2), data_format) # 1/2
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = conv2d_fixed_padding(inputs, 64, (3,3), (2,2), data_format) # 1/4
    inputs = batch_norm_relu(inputs, True, data_format)
    shortcut1 = inputs
    inputs = conv2d_fixed_padding(inputs, 128, (3,3), (2,2), data_format) # 1/8
    inputs = batch_norm_relu(inputs, True, data_format)
    shortcut2 = inputs
    inputs = conv2d_fixed_padding(inputs, 128, (3,3), (1,1), data_format) # 1/8
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = inputs + shortcut2
    inputs = conv2d_fixed_padding(inputs, 64, (3,3), (2,2), data_format, transpose=True) # 1/4
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = inputs + shortcut1
    inputs = conv2d_fixed_padding(inputs, 32, (3,3), (2,2), data_format, transpose=True) # 1/2
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = conv2d_fixed_padding(inputs, 16, (3,3), (2,2), data_format, transpose=True) # 1
    inputs = batch_norm_relu(inputs, True, data_format)
    inputs = conv2d_fixed_padding(inputs, 1, (5,5), (1,1), data_format) # 1
    inputs = inputs + shortcut0
    return inputs

def train_model_fn(features, labels, mode, params):
    #assert mode == tf.estimator.ModeKeys.TRAIN
    #tf.summary.image('inputs', tf.transpose(features,perm=[0,2,3,1]))
    outputs = model2(features)
    #variable_summaries('outputs', outputs)
    #variable_summaries('labels', labels)
    diff = labels - outputs
    diff_net = outputs - features
    variable_summaries('diff', diff)
    variable_summaries('diff_net', diff)
    tf.summary.image('diff', tf.transpose(diff,perm=[0,2,3,1]), max_outputs=1)
    tf.summary.image('diff_net', tf.transpose(diff_net,perm=[0,2,3,1]), max_outputs=1)
    #tf.summary.image('outputs', tf.transpose(outputs,perm=[0,2,3,1]), max_outputs=1)
    #tf.summary.image('labels', tf.transpose(labels,perm=[0,2,3,1]), max_outputs=1)
    visual = tf.concat([labels, outputs],axis=3)
    tf.summary.image('labels_outputs', tf.transpose(visual,perm=[0,2,3,1]),max_outputs=1)
    #variable_summaries('diff', diff)
    diff_loss = tf.nn.l2_loss(diff) / (FLAGS.batch_size * _PW * _PH * _PC)
    #tf.summary.scalar('diff_loss', diff_loss)
    reg_loss = _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    #tf.summary.scalar('reg_loss', reg_loss)
    loss = diff_loss + reg_loss
    tf.identity(loss, 'loss')
    tf.summary.scalar('train_loss', loss)
    ave_loss = tf.metrics.mean(loss)
    metrics = {'ave_loss': ave_loss}
    global_step = tf.train.get_or_create_global_step()
    batches_per_epoch = _NUM_SAMPLES['train'] / FLAGS.batch_size
    boundaries = [int(batches_per_epoch * epoch) for epoch in [260, 280, 290]]
    values = [_LEARNING_RATE * decay for decay in [1,1,1,1]]
    learning_rate = tf.train.piecewise_constant(
                            tf.cast(global_step, tf.int32), boundaries, values)
    optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

def eval_model_fn(features, labels, mode, params):
    outputs = model2(features)
    diff = labels - outputs
    visual = tf.concat([labels, outputs],axis=3)
    tf.summary.image('labels_outputs_eval', tf.transpose(visual,perm=[0,2,3,1]),max_outputs=1)
    diff_loss = tf.nn.l2_loss(diff) / (FLAGS.batch_size * _PW * _PH * _PC)
    reg_loss = _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss = diff_loss + reg_loss
    tf.identity(loss, 'eval_loss')
    tf.summary.scalar('eval_loss', loss)
    ave_loss = tf.metrics.mean(loss)
    metrics = {'ave_loss': ave_loss}
    return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            train_op=None,
            eval_metric_ops=metrics)

def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return train_model_fn(features, labels, mode, params)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return eval_model_fn(features, labels, mode, params)
    else:
        raise ValueError()

def main(_):
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1800)

    estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config, params={})
    for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        print 'CIRCLE', i
        tensors_to_log = {'loss': 'loss'}
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)
        estimator.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval), hooks=[logging_hook])
        eval_results = estimator.evaluate(input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size, 1))
        print eval_results

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
