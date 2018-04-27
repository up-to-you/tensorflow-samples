import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from keras.utils import np_utils


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 32, 32, 3])

with tf.name_scope('conv_1'):
    conv1 = conv_layer(x_image, shape=[5, 5, 3, 32])
    conv1_pool = max_pool_2x2(conv1)

with tf.name_scope('conv_2'):
    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)
    conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

with tf.name_scope('full_1'):
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

with tf.name_scope('dropout'):
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

with tf.name_scope('activations'):
    y_conv = full_layer(full1_drop, 10)
    tf.summary.scalar('cross_entropy_loss', y_conv)

(X, Y), (x_test, y_test) = cifar10.load_data()

Y = np_utils.to_categorical(Y, 10)

with tf.name_scope('cross'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv), tf.argmax(y_))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def make_learning_iteration(start, total, step, sess):
    for i in range(start, total, step):
        bndr = i + step
        if i != 0 and i % total == 0:
            break

        sess.run(train_step, feed_dict={x: X[i:bndr], y_: Y[i:bndr], keep_prob: 0.5})

        if i % (step * 10) == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: X[i:bndr], y_: Y[i:bndr], keep_prob: 0.5})
            print("step {}, training accuracy {}".format(i, train_accuracy))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        make_learning_iteration(0, 50000, 10, sess)
