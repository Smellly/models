from ast import literal_eval
import json
import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
    """Wrapper for inserting an float Feature into a SequenceExample proto.
    value needs to be iterable"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# get example
def get_file():
    filename = 'mscoco-visual-concepts/raw-data/val_sorted.json'
    with open(filename, 'r') as f:
        data = json.load(f)
    for item,j in data.iteritems():
        p = [literal_eval(i.split(':')[1])[0] for i in j]
        break
    return p

with tf.device('/cpu:0'):
    filename = 'mscoco-visual-concepts/raw-data/val2014/COCO_val2014_000000100166.jpg'
    with tf.gfile.FastGFile(filename, "r") as f:
        encoded_image = f.read()

    #p = np.array([1.2, 1.3, 1.4]).astype(np.float)
    #p = [1.2, 1.4, 1.5]
    #p = tf.convert_to_tensor(p, dtype=tf.float32)
    p = get_file()
    p1 = _float_feature(p)
    #p1 = _bytes_feature(p)
    context = tf.train.Features(feature={'data': p1,})
    example = tf.train.Example(features=context)
    # decode
    s_e = example.SerializeToString()
    # features = tf.parse_single_example(s_e, features = {'data': tf.FixedLenFeature([], tf.string),})
    # dimension needs to be clear!!!
    features = tf.parse_single_example(s_e, features = {'data': tf.FixedLenFeature([1000], tf.float32),})
    feature = features['data']
    print features
    print feature
    #rr = tf.reshape(feature, [3,1])
    rr = feature
    #f1 = tf.decode_raw(feature, tf.float32)
    #rr = tf.reshape(f1, [3,1])
    #image = tf.image.decode_jpeg(feature, channels=3)
    #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #print image
    #print image.get_shape()

with tf.Session() as sess:
    #r = tf.contrib.learn.run_n(rr)
    r =  sess.run(rr)
    print len(r)


