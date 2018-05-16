import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import math

def generate_x_y_data_two_freqs(isTrain, batch_size, seq_length):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        )

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2
        sig1 = amp_rand * np.cos(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        ) + sig1

        x1 = sig1[: seq_length]
        y1 = sig1[seq_length:]

        x_ = np.array([x1])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

    batch_x.append(x_)
    batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y


def generate_x_y_data_v1(isTrain, batch_size):
    seq_length = 30,
    x, y = generate_x_y_data_two_freqs(isTrain , batch_size , seq_length=seq_length),
    noise_amount = random.random() * 0.15 + 0.10,

    x = x + noise_amount * np.random.randn(seq_length, batch_size , 1),

    avg = np.average(x),
    std = np.std(x) + 0.0001,

    x = x - avg,
    y = y - avg,

    x = x / std / 2.5
    y = y / std / 2.5

    return x, y


tf.nn.seq2seq = tf.contrib.legacy_seq2seq
tf.nn.rnn_cell = tf.contrib.rnn
tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
tf.reset_default_graph()

sess = tf.InteractiveSession()

sample_x, sample_y = generate_x_y_data_v1(isTrain=True, batch_size=3)

seq_length = sample_x.shape[0]
batch_size = 5
output_dim = input_dim = sample_x.shape[-1]
hidden_dim = 12
layers_stacked_count = 2

learning_rate = 0.007
nb_iters = 150
lr_decay = 0.92
momentum = 0.5
lambda_l2_reg = 0.003

# Кодировщик:
w_in = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
b_in = tf.Variable(tf.random_normal([hidden_dim], mean=1.0))
# Переменные декодировщика
w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
b_out = tf.Variable(tf.random_normal([output_dim]))

with tf.variable_scope('Seq2seq'):
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
        for t in range(seq_length)
    ]

    reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in enc_inp]

    cells = []

    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))

    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_out".format(t))
        for t in range(seq_length)
    ]

    dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[: -1]

    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        enc_inp,
        dec_inp,
        cell
    )

    # Декодирование
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    reshaped_outputs = [output_scale_factor * (tf.matmul(i, w_out) + b_out) for i in dec_outputs]

with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=0.5)
    train_op = optimizer.minimize(loss)


def train_batch(batch_size):
    X, Y = generate_x_y_data_v1(isTrain=True, batch_size=batch_size)

    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(batch_size):
    X, Y = generate_x_y_data_v1(isTrain=False, batch_size=batch_size)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


sess.run(tf.global_variables_initializer())

nb_predictions = 1

print("Let's visualize {} predictions with our signals: ".format(nb_predictions))

X, Y = generate_x_y_data_v1(isTrain=False, batch_size=nb_predictions)

feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}

outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

for j in range(nb_predictions):
    plt.figure(figsize=(12, 3))

    for k in range(output_dim):
        past = X[:, j, k]
        expected = Y[:, j, k]
        pred = outputs[:, j, k]

    label1 = "Seen (past) values" if k == 0 else "_nolegend_"
    label2 = "True future values" if k == 0 else "_nolegend_"
    label3 = "Predictions" if k == 0 else "_nolegend_"

    plt.plot(range(len(past)), past, "o--b ", label=label1)
    plt.plot(range(len(past), len(expected) + len(past)), expected, "x--b ", label=label2)
    plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y ", label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()
