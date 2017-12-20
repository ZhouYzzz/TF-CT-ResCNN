"""
System metric tensor
"""
import tensorflow as tf
import numpy as np

def system_metric():
    indices = np.fromfile('data/H_indices.bin', np.int64)
    values = np.fromfile('data/H_values.bin', np.float64)
    indices = indices.reshape(-1,2)
    indices_t = tf.constant(indices)
    values_t = tf.constant(values)
    return tf.SparseTensor(indices_t, values_t, dense_shape=(40000,77760))
    #indices_raw = tf.read_file('data/H_indices.bin')
    #values_raw = tf.read_file('data/H_values.bin')
    #indices_raw = tf.Print(indices_raw, ['Load indices'])
    #values_raw = tf.Print(values_raw, ['Load values'])
    #indices = tf.decode_raw(indices_raw, tf.int64)
    #values = tf.decode_raw(values_raw, tf.float64)
    #indices = tf.reshape(indices, shape=(-1,2))
    #return tf.SparseTensor(indices, values, dense_shape=(40000,77760))

def sample_prj():
    reader = tf.TFRecordReader()
    input_producer = tf.train.string_input_producer(['dataset/val.tfrecords'])
    key, value = reader.read(input_producer)
    example = tf.parse_single_example(value,features={
            'sparse1': tf.FixedLenFeature([1,216,72], tf.float32),
            'sparse2': tf.FixedLenFeature([1,216,72], tf.float32),
            'sparse3': tf.FixedLenFeature([1,216,72], tf.float32),
            'sparse4': tf.FixedLenFeature([1,216,72], tf.float32),
            'sparse5': tf.FixedLenFeature([1,216,72], tf.float32),
            'image': tf.FixedLenFeature([1,200,200], tf.float32)
        })
    print example
    full_projection = [None for _ in range(360)]
    full_projection[0::5] = tf.split(example['sparse1'], 72, axis=2)
    full_projection[1::5] = tf.split(example['sparse2'], 72, axis=2)
    full_projection[2::5] = tf.split(example['sparse3'], 72, axis=2)
    full_projection[3::5] = tf.split(example['sparse4'], 72, axis=2)
    full_projection[4::5] = tf.split(example['sparse5'], 72, axis=2)
    full_projection = tf.concat(full_projection, axis=2)
    return full_projection, example['image']

def W_mat():
    W = np.fromfile('data/W.bin', np.float64)
    W = tf.constant(np.reshape(W,[1,216,360]))
    return W

def F_mat():
    F = np.fromfile('data/F.bin', np.float64)
    F = tf.constant(np.reshape(F,[1,216,216]))
    return F

def main(_):
    H = system_metric()
    W = W_mat()
    F = F_mat()
    print H
    #inputs = tf.ones([10,77760],tf.float64)
    #outputs = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)
    #outputs = tf.reshape(outputs, (-1,200,200))
    #outputs2 = tf.sparse_tensor_dense_matmul(H, inputs, adjoint_b=True)
    projection, image = sample_prj()
    projection = tf.cast(projection, tf.float64)
    print projection
    Wp = tf.multiply(W, projection)
    print Wp
    FWp = tf.matmul(F, Wp)
    print FWp
    FWp = tf.transpose(FWp, perm=[0,2,1])
    print FWp
    FWp = tf.reshape(FWp,shape=(-1,77760))
    print FWp
    #HFWp = tf.sparse_tensor_dense_matmul(H, tf.reshape(tf.transpose(FWp),shape=(-1,77760)), adjoint_b=True)
    HFWp = tf.sparse_tensor_dense_matmul(H, FWp, adjoint_b=True)
    print HFWp
    HFWp = tf.transpose(tf.reshape(HFWp, shape=(-1,1,200,200)))
    HFWp = tf.cast(HFWp,tf.float32)
    tf.summary.image('HFWp', tf.reshape(HFWp, shape=(-1,200,200,1)), max_outputs=1)
    tf.summary.image('Wp', tf.reshape(Wp, shape=(-1,216,360,1)), max_outputs=1)
    tf.summary.image('FWp', tf.reshape(FWp, shape=(-1,216,360,1)), max_outputs=1)
    tf.summary.image('projection', tf.reshape(projection,shape=(-1,216,360,1)), max_outputs=1)
    tf.summary.image('image', tf.reshape(image,shape=(-1,200,200,1)), max_outputs=1)
    tf.summary.image('concat', tf.concat([tf.reshape(HFWp, shape=(-1,200,200,1)), tf.reshape(image,shape=(-1,200,200,1))],axis=2), max_outputs=1)
    merged = tf.summary.merge_all()
    summary_write = tf.summary.FileWriter('/tmp/CT_test')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in xrange(1):
            print i
            summary, H1, H2 = sess.run([merged, tf.reshape(HFWp,[200,200]), tf.reshape(image,[200,200])])
            np.save('H1',H1)
            np.save('H2',H2)
            summary_write.add_summary(summary, i)

        coord.request_stop()
        coord.join(threads)

        #sess.run(outputs)
        #sess.run(outputs2)
        #print outputs.shape

if __name__ == '__main__':
    tf.app.run()

