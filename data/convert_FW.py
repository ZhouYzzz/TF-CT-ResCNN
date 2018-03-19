#!/usr/bin/python
"""
Convert F.mat, W.mat file to tensorflow example files (tfrecords)
"""
import tensorflow as tf
import h5py
import numpy as np


def main(_):
    d = h5py.File('F.mat')
    F = d['F'].value.T
    with open('F.bin', 'wb') as f:
        f.write(F.data)
        f.close()
    d = h5py.File('W.mat')
    W = d['W'].value.T
    with open('W.bin', 'wb') as f:
        f.write(W.data)
        f.close()
    


if __name__ == '__main__':
    tf.app.run()

