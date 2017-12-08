import tensorflow as tf
import numpy as np
from resnet_model import *

_PW = 72
_PH = 216
_PC = 1
_IW = 200
_IH = 200
_IC = 1

def _conv_deconv(inputs, block_fn, is_training, data_format):
    shortcut = inputs
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=32, kernel_size=5, strides=1,
        data_format=data_format, transpose=True) # 1
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=2,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format) # 1/4
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=1, kernel_size=5, strides=1,
        data_format=data_format, transpose=True) # 1
    #inputs = conv2d_fixed_padding(
    #    inputs=inputs, filters=64, kernel_size=7, strides=1,
    #    data_format=data_format) # 1
    #inputs = conv2d_fixed_padding(
    #    inputs=inputs, filters=32, kernel_size=5, strides=1,
    #    data_format=data_format) # 1
    #inputs = conv2d_fixed_padding(
    #    inputs=inputs, filters=1, kernel_size=5, strides=1,
    #    data_format=data_format) # 1
    #inputs = inputs + shortcut
    #inputs = conv2d_fixed_padding(
    #    inputs=inputs, filters=16, kernel_size=7, strides=2,
    #    data_format=data_format) # 1/2
    #inputs = block_layer(
    #    inputs=inputs, filters=32, block_fn=block_fn, blocks=2,
    #    strides=2, is_training=is_training, name='block_layer1',
    #    data_format=data_format) # 1/4
    #inputs = block_layer(
    #    inputs=inputs, filters=64, block_fn=block_fn, blocks=2,
    #    strides=2, is_training=is_training, name='block_layer2',
    #    data_format=data_format) # 1/8
    #inputs = block_layer(
    #    inputs=inputs, filters=32, block_fn=block_fn, blocks=2,
    #    strides=2, is_training=is_training, name='block_layer3',
    #    data_format=data_format, transpose=True) # 1/4
    #inputs = block_layer(
    #    inputs=inputs, filters=16, block_fn=block_fn, blocks=2,
    #    strides=2, is_training=is_training, name='block_layer3',
    #    data_format=data_format, transpose=True) # 1/2
    #inputs = conv2d_fixed_padding(
    #    inputs=inputs, filters=1, kernel_size=7, strides=2,
    #    data_format=data_format, transpose=True) # 1
    #inputs = inputs + shortcut
    return inputs

def _PRJ_net(inputs, is_training, data_format):
    """ PROJECTION ESTIMATION NETWORK """
    def _PRJ_branch(inputs, index, is_training, data_format):
        with tf.variable_scope('B{}'.format(index)):
            inputs = _conv_deconv(inputs, building_block, is_training, data_format)
            inputs = tf.identity(inputs, 'outputs')
            print inputs
        return inputs
    inputs = [_PRJ_branch(inputs, i, is_training, data_format) for i in xrange(5)]
    sliced_inputs = [None for i in xrange(_PW*5)]
    for i in xrange(5):
        sliced_inputs[i::5] = tf.split(inputs[i],_PW,axis=3)
    inputs = tf.concat(sliced_inputs, axis=3)
    inputs = tf.identity(inputs, 'outputs')
    return inputs

def _FBP_net(inputs, is_training, data_format):
    """ FBP NETWORK """
    # inputs: (None,_PC,_PH,_PN*_PW)
    def _load_weights():
        W = tf.constant(np.fromfile('data/W.bin', np.float64), shape=(1,_PH,_PW*5))
        F = tf.constant(np.fromfile('data/F.bin', np.float64), shape=(1,_PH,_PH))
        Hi = tf.constant(np.fromfile('data/H_indices.bin', np.int64).reshape(-1,2))
        Hv = tf.constant(np.fromfile('data/H_values.bin', np.float64))
        H = tf.SparseTensor(Hi, Hv, dense_shape=(_IH*_IW, _PH*_PW*5))
        return H, F, W
    H, F, W = _load_weights()
    inputs = tf.cast(inputs, tf.float64)                                # case to double precision
    inputs = tf.scan(lambda a,x: tf.multiply(W,x), inputs)              # WP
    inputs = tf.scan(lambda a,x: tf.matmul(F,x), inputs)                # FWP
    inputs = tf.reshape(tf.transpose(inputs), shape=(-1,_PH*_PW*5))     # flatten
    inputs = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)   # HFWP
    inputs = tf.reshape(inputs, shape=(-1,_IC,_IH,_IW))                 # reshape to (None,1,200,200)
    inputs = tf.cast(inputs, tf.float32)                                # cast to single precision
    inputs = tf.identity(inputs, 'outputs')
    return inputs

def _RFN_net(inputs, is_training, data_format):
    """ IMAGE REFINEMENT NETWORK """
    inputs = _conv_deconv(inputs, building_block, is_training, data_format)
    inputs = tf.identity(inputs, 'outputs')
    return inputs

def model(inputs, is_training, data_format='channels_first'):
    assert data_format=='channels_first', 'NO_IMPLEMENT'
    with tf.name_scope('PRJ'):
        inputs = _PRJ_net(inputs, is_training, data_format)
        print inputs
    #with tf.name_scope('FBP'):
    #    inputs = _FBP_net(inputs, is_training, data_format)
    #    print inputs
    #with tf.name_scope('RFN'):
    #    inputs = _RFN_net(inputs, is_training, data_format)
    #    print inputs
    return inputs

def main(_):
    p = tf.placeholder(tf.float32, shape=(None,_PC,_PH,_PW))
    #inputs = tf.ones([1,_PC,_PH,_PW], tf.float32)
    inputs = model(p, False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(inputs, feed_dict={p: np.random.rand(1,_PC,_PH,_PW)})

if __name__ == '__main__':
    tf.app.run()

