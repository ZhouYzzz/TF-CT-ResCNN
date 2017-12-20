import tensorflow as tf

#example = tf.train.example(features=tf.train.features(feature={
#    'sparse1': _float_feature(sparse1[i]),
#    'sparse2': _float_feature(sparse2[i]),
#    'sparse3': _float_feature(sparse3[i]),
#    'sparse4': _float_feature(sparse4[i]),
#    'sparse5': _float_feature(sparse5[i]),
#    'image': _float_feature(image[i])}))

_PW = 72
_PH = 216
_PC = 1
_IW = 200
_IH = 200
_IC = 1

def parse_example(serialized_example):
    def _concat(sparses):
        sliced_inputs = [None for i in xrange(_PW*5)]
        for i in xrange(5):
            sliced_inputs[i::5] = tf.split(sparses[i],_PW,axis=2)
        projection = tf.concat(sliced_inputs, axis=2)
        return projection
    features = tf.parse_single_example(
            serialized_example, features={
                'sparse1': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse2': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse3': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse4': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'sparse5': tf.FixedLenFeature([_PC,_PH,_PW], tf.float32),
                'image': tf.FixedLenFeature([_IC,_IH,_IW], tf.float32)
            })
    inputs = features['sparse3']
    projection = _concat([features['sparse{}'.format(i)] for i in [1,2,3,4,5]])
    image = features['image']
    return inputs, projection, image

def main(_):
    reader = tf.TFRecordReader()
    input_producer = tf.train.string_input_producer(['dataset/val.tfrecords'])
    key, value = reader.read(input_producer)

    inputs, projection, image = parse_example(value)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        [A,B,C] = sess.run([inputs, projection, image])
        print A.shape
        print B.shape
        print C.shape

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()

