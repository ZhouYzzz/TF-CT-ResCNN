#!/usr/bin/python
"""
Convert imdb.mat file to tensorflow example files (tfrecords)
"""
import tensorflow as tf
import h5py
import numpy as np

def _float_feature(array):
    value = np.reshape(array,-1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def main(_):
    print 'Loading mat file...'
    data = h5py.File('data/imdb_Lung.mat')
    set = np.squeeze(data['images']['set'].value)
    index = np.arange(set.size)
    train_index = index[set==1]
    val_index = index[set==2]
    print 'All {} examples, {} trains, {} vals'.format(index.size, train_index.size, val_index.size)
    return
    sparse1 = np.transpose(data['images']['sparse1'].value,[0,1,3,2])
    sparse2 = np.transpose(data['images']['sparse2'].value,[0,1,3,2])
    sparse3 = np.transpose(data['images']['sparse3'].value,[0,1,3,2])
    sparse4 = np.transpose(data['images']['sparse4'].value,[0,1,3,2])
    sparse5 = np.transpose(data['images']['sparse5'].value,[0,1,3,2])
    image = np.transpose(data['images']['image'].value,[0,1,3,2])
    
    train_index_split = np.array_split(train_index, 10)
    for irecord in xrange(10):
        print 'Write train record no. {}'.format(irecord)
        writer = tf.python_io.TFRecordWriter('dataset/train_{}.tfrecords'.format(irecord))
        for i in train_index_split[irecord]:
            example = tf.train.Example(features=tf.train.Features(feature={
                'sparse1': _float_feature(sparse1[i]),
                'sparse2': _float_feature(sparse2[i]),
                'sparse3': _float_feature(sparse3[i]),
                'sparse4': _float_feature(sparse4[i]),
                'sparse5': _float_feature(sparse5[i]),
                'image': _float_feature(image[i])}))
            writer.write(example.SerializeToString())
        writer.close()

    print 'Write val record'
    writer = tf.python_io.TFRecordWriter('dataset/val.tfrecords')
    for i in val_index:
        example = tf.train.Example(features=tf.train.Features(feature={
            'sparse1': _float_feature(sparse1[i]),
            'sparse2': _float_feature(sparse2[i]),
            'sparse3': _float_feature(sparse3[i]),
            'sparse4': _float_feature(sparse4[i]),
            'sparse5': _float_feature(sparse5[i]),
            'image': _float_feature(image[i])}))
        writer.write(example.SerializeToString())
    writer.close()

        


if __name__ == '__main__':
    tf.app.run()

