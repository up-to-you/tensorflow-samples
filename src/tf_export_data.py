import skimage.io as io
import tensorflow as tf

from keras.utils import np_utils

import os

DATA_DIR = '/home/owner/NN_DATA/101_ObjectCategories'
TARGET_TF_RECORD = '/home/owner/target.tfrecords'


def get_subdirs_sorted():
    os_generator = os.walk(DATA_DIR)
    return sorted(next(os_generator)[1])


def names_to_categories(sorted_class_names):
    cat_num = len(sorted_class_names)
    categories = []
    for idx in range(0, cat_num):
        categories.append(idx)
    return np_utils.to_categorical(categories, cat_num)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


subfolders = get_subdirs_sorted()
data_pairs = []

for idx, subfold_class in enumerate(subfolders):
    images = io.imread_collection(DATA_DIR + '/' + subfold_class + '/*.jpg')
    for img in images:
        data_pairs.append((img, idx))

writer = tf.python_io.TFRecordWriter(TARGET_TF_RECORD)

for img, clazz in data_pairs:
    feature = {
        'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
        'clazz': _int64_feature(clazz)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.flush()
writer.close()
