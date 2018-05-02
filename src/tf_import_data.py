import skimage.io as io
import tensorflow as tf
import numpy as np

TARGET_TF_RECORD = '/home/owner/target.tfrecords'

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=TARGET_TF_RECORD)

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height'].int64_list.value[0])

    width = int(example.features.feature['width'].int64_list.value[0])

    img_string = (example.features.feature['image'].bytes_list.value[0])

    clazz = (example.features.feature['clazz'].int64_list.value[0])

    img_1d = np.fromstring(img_string, dtype=np.uint8)
    img = img_1d.reshape((height, width, -1))

    io.imshow(img)
    io.show()

