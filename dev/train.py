import tensorflow as tf
import os, glob
from model.res_cnn import res_cnn_model
from model.prj_est_subnet import slice_concat
from utils.summary import visualize, statistics
from model.example_spec import train_example_spec, serve_example_spec

tf.flags.DEFINE_string('data_dir', './dataset', '')
tf.flags.DEFINE_string('model_dir', './tmp/model_v0', '')
tf.flags.DEFINE_integer('batch_size', 10, '')
tf.flags.DEFINE_float('base_lr', 0.1, '')
tf.flags.DEFINE_float('second_lr_ratio', 0.01, '')
tf.flags.DEFINE_float('momentum', 0.9, '')
tf.flags.DEFINE_float('clip_gradient', 0.01, '')
tf.flags.DEFINE_integer('pretrain_steps', 5000, '')
tf.flags.DEFINE_integer('num_epoches_per_stage', 5, '')
tf.flags.DEFINE_integer('epoches_per_val', 1, '')
tf.flags.DEFINE_string('gpus', '0', '')
FLAGS = tf.flags.FLAGS

_NUM_SAMPLES = {'train': 17120, 'val': 592}

def get_learning_rate():
  return tf.constant(FLAGS.base_lr)


def input_fn(is_training, batch_size, num_epochs):
  def _get_filenames(is_training):
    if is_training:
      return glob.glob(os.path.join(FLAGS.data_dir, 'train*.tfrecords'))
    else:
      return [os.path.join(FLAGS.data_dir, 'val.tfrecords')]
  record_filenames = _get_filenames(is_training)
  dataset = tf.data.TFRecordDataset(record_filenames)
  if is_training:
    dataset = dataset.repeat(num_epochs)
  dataset = dataset.map(lambda s: tf.parse_single_example(s, features=train_example_spec()))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return {'inputs': features['sparse3']}, features


def model_fn(features, labels, mode, params, config=None):
  # switch stages
  stage = params['stage'] if params.has_key('stage') else FLAGS.batch_size
  batch_size = params['batch_size']
  print(params, mode)

  graph = tf.get_default_graph()
  #print graph.get_operations()

  inputs = features['inputs']
  #if params['preload']:
  #  #predictions = res_cnn_model(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  #  with tf.Session(graph=graph) as sess:
  #    graphdef = tf.saved_model.loader.load(sess, ['serve'], './tmp/model_v0/1513566387',input_map={'ParseExample/ParseExample:0': inputs, 'global_step:0': graph.get_tensor_by_name('global_step:0')})
  #    #predictions, = tf.import_graph_def(graphdef.graph_def,input_map={'ParseExample/ParseExample:0': inputs}, return_elements=['RFN/outputs:0'])
  #    #tf.train.global_step(sess, graph.get_tensor_by_name('global_step:0'))
  #  predictions = graph.get_tensor_by_name('RFN/outputs:0')
  #  print graph.get_tensor_by_name('ParseExample/ParseExample:0')
  #  #print tf.global_variables()
  #  print tf.train.get_global_step()
  #else:
  predictions = res_cnn_model(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    export_outputs = tf.estimator.export.PredictOutput({'outputs':predictions})
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions={'outputs': predictions}, export_outputs={'export_outputs': export_outputs})

  prj_labels = slice_concat([labels['sparse{}'.format(i+1)] for i in range(5)], axis=3)
  prj_outputs = graph.get_tensor_by_name('PRJ/outputs:0')
  fbp_outputs = graph.get_tensor_by_name('FBP/outputs:0')
  image_labels = labels['image']
  image_outputs = predictions
  visualize_outputs = tf.zeros_like(image_outputs) if stage == 0 else image_outputs

  prj_tvars = tf.trainable_variables('PRJ')
  rfn_tvars = tf.trainable_variables('RFN')

  visualize(tf.concat([prj_labels, prj_outputs], axis=2), 'projections')
  visualize(tf.concat([image_labels, fbp_outputs, visualize_outputs], axis=3), 'images')

  # loss
  prj_loss = tf.nn.l2_loss(prj_labels - prj_outputs) / (batch_size * 1 * 216 * 360)
  rfn_loss = tf.nn.l2_loss(image_labels - image_outputs) / (batch_size * 1 * 200 * 200)

  loss = prj_loss if stage == 0 else rfn_loss
  loss = tf.identity(loss, 'loss')

  # metrics
  image_labels_f = tf.layers.flatten(image_labels)
  image_outputs_f = tf.layers.flatten(fbp_outputs if stage == 0 else image_outputs)
  rmse = tf.norm(image_labels_f - image_outputs_f, axis=1) / tf.norm(image_labels_f, axis=1)
  rmse_metrics = tf.metrics.mean(rmse)

  # train_op
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = graph.get_tensor_by_name('global_step:0') if params['preload'] else tf.train.get_or_create_global_step()
    learning_rate = get_learning_rate()
    learning_rate = tf.identity(learning_rate, 'learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    # train PRJ net
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='PRJ')
    with tf.control_dependencies(update_ops):
      base_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum)
      grads_and_vars = base_optimizer.compute_gradients(prj_loss if stage == 0 else rfn_loss, prj_tvars)
      clipped_grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.clip_gradient), var)
                                for grad, var in grads_and_vars if grad is not None]
      base_train_op = base_optimizer.apply_gradients(clipped_grads_and_vars, global_step)

    # train RFN net
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=None)
    with tf.control_dependencies(update_ops):
      # end to end training
      second_optimizer = tf.train.MomentumOptimizer(
        learning_rate=(FLAGS.second_lr_ratio if stage == 2 else 1) * learning_rate, momentum=FLAGS.momentum)
      second_grad_and_vars = second_optimizer.compute_gradients(rfn_loss, rfn_tvars)
      second_clipped_grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.clip_gradient), var)
                                       for grad, var in second_grad_and_vars if grad is not None]
      second_train_op = second_optimizer.apply_gradients(second_clipped_grads_and_vars, global_step)
      if stage == 0:
        train_op = base_train_op
      elif stage == 1:
        train_op = second_train_op
      elif stage == 2:
        train_op = tf.group(second_train_op, base_train_op)
  else:
    train_op = None

  #print graph.get_tensor_by_name('PRJ/Branch1/conv2d_2/kernel/Momentum:0')
  #print graph.get_tensor_by_name('RFN/conv2d_2/kernel/Momentum:0')
  #print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=None,
    export_outputs=None,
    training_chief_hooks=None,
    training_hooks=None,
    scaffold=None,
    evaluation_hooks=None
  )


def train_stage(stage, config, hooks):
  print('Training stage {}'.format(stage))
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=FLAGS.model_dir,
                                     config=config,
                                     params={'stage': stage, 'batch_size': FLAGS.batch_size, 'preload': False})
  for eval_circ in range(FLAGS.num_epoches_per_stage // FLAGS.epoches_per_val):
    MAX_STEPS = _NUM_SAMPLES['train']//FLAGS.batch_size*FLAGS.epoches_per_val*(eval_circ+1)
    estimator.train(lambda: input_fn(True, FLAGS.batch_size, FLAGS.epoches_per_val), hooks=hooks, max_steps=MAX_STEPS)
    eval_results = estimator.evaluate(lambda: input_fn(False, FLAGS.batch_size, 1))
    print(eval_results)


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  config = tf.estimator.RunConfig().replace(save_checkpoints_secs=100000,
                                            save_summary_steps=100,
                                            keep_checkpoint_max=5)
  tensors_to_log = ['loss', 'learning_rate']
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
  # stage 0
  train_stage(0, config=config, hooks=[logging_hook])
  # stage 1
  train_stage(1, config=config, hooks=[logging_hook])
  # stage 2
  train_stage(2, config=config, hooks=[logging_hook])

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
