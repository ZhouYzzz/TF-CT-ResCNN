#!/usr/bin/python
import tensorflow as tf
flags = tf.flags
flags.DEFINE_string('data_dir', './dataset', '')
flags.DEFINE_string('model_dir', './model', '')
flags.DEFINE_integer('train_epochs', 100, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('epochs_per_eval', 20, '')
flags.DEFINE_string('data_format', 'channels_first', '')
FLAGS = flags.FLAGS
import os, sys, glob
from model import model # def model(inputs, is_training, data_format):
#from example import parse_example # return (inputs, projection, image)
_LEARNING_RATE = 1e-1
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9
_PW = 72
_PH = 216
_PC = 1
_IW = 200
_IH = 200
_IC = 1
_NUM_SAMPLES = {'train': 17120, 'val': 592}
def _concat(sparses,axis=3):
    sliced_inputs = [None for i in xrange(_PW*5)]
    for i in xrange(5):
        sliced_inputs[i::5] = tf.split(sparses[i],_PW,axis=axis)
    projection = tf.concat(sliced_inputs, axis=axis)
    return projection
def parse_example(serialized_example):
    features = tf.parse_single_example(
            serialized_example, features={
                'sparse1': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse2': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse3': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse4': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse5': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'image': tf.FixedLenFeature([_IC,_IH,_IW], tf.float32)
            })
    #inputs = features['sparse3']
    projection = _concat([features['sparse{}'.format(i)] for i in [1,2,3,4,5]],axis=2)
    #image = features['image']
    return features['sparse3'], features['sparse1'], projection
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
    sparse3, sparse1, projection = iterator.get_next()
    return {'inputs': sparse3}, {'projection': projection, 'sparse1': sparse1}
    #return {'inputs': inputs}, {'projection': projection, 'image': image}
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
def model_fn(features, labels, mode, params):
    def _train_model_fn(features, labels, params):
        graph = tf.Graph()
        graph.as_default()
        inputs = features['inputs']
        sparse3 = inputs
        outputs = model(inputs, is_training=True)
        #projection = outputs #tf.get_default_graph().get_tensor_by_name('PRJ/outputs:0')
        sparse1 = tf.get_default_graph().get_tensor_by_name('PRJ/B0/outputs:0')
        projection = _concat([sparse1, sparse3, sparse3, sparse3, sparse3],axis=3)
        display = tf.concat([labels['projection'], projection], axis=2) # height
        tf.summary.image('U_labels_D_projection', tf.transpose(display,perm=[0,2,3,1]), max_outputs=1)
        #diff = projection - labels['projection']
        diff = sparse1 - labels['sparse1']
        variable_summaries('diff', diff)
        diff2 = sparse1 - sparse3
        variable_summaries('diff_net', diff2)
        tf.summary.image('diff_net', tf.transpose(diff2,perm=[0,2,3,1]),max_outputs=1)
        tf.summary.image('diff', tf.transpose(diff,perm=[0,2,3,1]), max_outputs=1)
        loss = tf.nn.l2_loss(diff) / (FLAGS.batch_size*_PH*_PW*_PC)
        #loss += _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        tf.summary.scalar('loss', loss)
        tf.identity(loss, 'prj_loss')
        #rmse = tf.metrics.mean_squared_error(projection, labels['projection'], weights=tf.norm(labels['projection'],axis=(-1,-2)))
        rmse = tf.metrics.mean_squared_error(tf.multiply(diff, 1/tf.norm(labels['projection'],axis=(-1,-2))), tf.zeros_like(diff))
        metrics = {'rmse': rmse}
        global_step = tf.train.get_or_create_global_step()

        batches_per_epoch = _NUM_SAMPLES['train'] / FLAGS.batch_size
        boundaries = [int(batches_per_epoch * epoch) for epoch in [60,100]]
        values = [_LEARNING_RATE * decay for decay in [1,0.1,0.01]]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
        optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=_MOMENTUM)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                eval_metric_ops=metrics,
                train_op=train_op)
    def _eval_model_fn(features, labels, params):
        graph = tf.Graph()
        graph.as_default()
        inputs = features['inputs']
        outputs = model(inputs, is_training=False)
        projection = tf.get_default_graph().get_tensor_by_name('PRJ/outputs:0')
        display = tf.concat([labels['projection'], projection], axis=2) # height
        tf.summary.image('U_labels_D_projection(eval)', tf.transpose(display,perm=[0,2,3,1]), max_outputs=1)
        diff = projection - labels['projection']
        tf.summary.image('diff(eval)', tf.transpose(diff,perm=[0,2,3,1]), max_outputs=1)
        loss = tf.nn.l2_loss(diff) / (FLAGS.batch_size*_PH*_PW*_PC)
        #loss += _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        tf.summary.scalar('loss(eval)', loss)
        tf.identity(loss, 'prj_loss(eval)')
        rmse = tf.metrics.mean_squared_error(tf.multiply(diff, 1/tf.norm(labels['projection'],axis=(-1,-2))), tf.zeros_like(diff))
        metrics = {'rmse': rmse}
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops=metrics)
    def _predict_model_fn(features, params):
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT)
    print 'Creating model for', mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        return _train_model_fn(features, labels, params)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return _eval_model_fn(features, labels, params)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return _predict_model_fn(features, params)
    else:
        raise ValueError('Undefined mode {}'.format(mode))

def main(_):
    run_config = tf.estimator.RunConfig()

    estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config, params={})
    for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        print 'CIRCLE', i
        tensors_to_log = {
                'prj_loss': 'prj_loss'
                }
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)
        estimator.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval), hooks=[logging_hook])
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

