import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
LEARNING_RATE = 0.31

L1 = 200
L2 = 100
L3 = 60
L4 = 30
L5 = 10

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]), dtype=tf.float32)

y_true = tf.placeholder(tf.float32, [None, 10])

l1 = tf.layers.dense(x, L1, activation=tf.nn.relu, use_bias=True)
l2 = tf.layers.dense(l1, L2, activation=tf.nn.relu, use_bias=True)
l3 = tf.layers.dense(l2, L3, activation=tf.nn.relu, use_bias=True)
l4 = tf.layers.dense(l3, L4, activation=tf.nn.relu, use_bias=True)
y_pred = tf.layers.dense(l4, L5, activation=tf.nn.relu, use_bias=True)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # Train
    sess.run(tf.global_variables_initializer())
    for i in range(NUM_STEPS):
        batch_x, batch_y = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_x, y_true: batch_y})
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans * 100))
