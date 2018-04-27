import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
LEARNING_RATE = 2.7

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]), dtype=tf.float32)

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred_linear = tf.matmul(x, W) + b
y_pred_sigmoid = 1 / (1 + tf.exp(y_pred_linear))

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_sigmoid, labels=y_true)

cross_entropy = tf.reduce_mean(entropy)

gd_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred_sigmoid, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # Train
    sess.run(tf.global_variables_initializer())
    for i in range(NUM_STEPS):
        batch_x, batch_y = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_x, y_true: batch_y})
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans * 100))
