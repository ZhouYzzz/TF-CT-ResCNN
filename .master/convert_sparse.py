#!/usr/bin/python
"""
Convert system metric mat file (H.mat) to sparse indices and values (binary)
"""
import tensorflow as tf
import h5py
import numpy as np 
def main(_):
    d = h5py.File('data/H.mat')
    ir = d['H']['ir'].value
    jc = d['H']['jc'].value
    v = d['H']['data'].value
    w = jc.shape[0]-1
    indices = list()
    values = list()
    count = 0
    for IC in xrange(w):
        # ic is the column number
        start = jc[IC]
        end = jc[IC+1]
        s = slice(start, end)
        irs = ir[s]
        print IC, len(irs)
        for IR in irs:
            indices.append([IR, IC])
            values.append(v[count])
            #print [IR, IC], v[count]
            count += 1
    indices = np.array(indices,np.int64)
    values = np.array(values,np.float64)
    return
    print 'Write indices'
    with open('data/H_indices.bin', 'wb') as f:
        f.write(indices.data)
        f.close()

    print 'Write values'
    with open('data/H_values.bin', 'wb') as f:
        f.write(values.data)
        f.close()

if __name__ == '__main__':
    tf.app.run()

