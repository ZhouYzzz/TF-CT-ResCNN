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
from example import parse_example # return (inputs, projection, image)
_LEARNING_RATE = 1e-1
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9
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
    inputs, projection, image = iterator.get_next()
    return {'inputs': inputs}, {'projection': projection, 'image': image}
def model_fn(features, labels, mode, params):
    def _train_model_fn(features, labels, params):
        graph = tf.Graph()
        graph.as_default()
        inputs = features['inputs']
        tf.summary.image('inputs', tf.transpose(inputs,perm=[0,2,3,1]), max_outputs=2)
        outputs = model(inputs, is_training=True)
        #loss = tf.nn.l2_loss(outputs - labels['image']) / FLAGS.batch_size
        #tf.identity(loss, 'image_loss')
        projection = tf.get_default_graph().get_tensor_by_name('PRJ/outputs:0')
        #projection = tf.Print(projection, [projection])
        print projection, labels['projection']
        tf.summary.image('projection', tf.transpose(projection,perm=[0,2,3,1]), max_outputs=2)
        tf.summary.image('projection_label', tf.transpose(labels['projection'],perm=[0,2,3,1]), max_outputs=2)
        loss = tf.nn.l2_loss(projection - labels['projection']) / FLAGS.batch_size + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        tf.summary.scalar('loss', loss)
        tf.identity(loss, 'prj_loss')
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.MomentumOptimizer(
                learning_rate=_LEARNING_RATE,
                momentum=_MOMENTUM)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1e-5)
        optimizer.apply_gradients(zip(grads, tvars))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=train_op)
    def _eval_model_fn(features, labels, params):
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL)
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
                tensors=tensors_to_log, every_n_iter=10)
        estimator.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval), hooks=[logging_hook])
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

