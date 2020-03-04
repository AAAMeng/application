#!/usr/bin/python
# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np
from dataSet import DataSet
import time
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from dataRead import read_data
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.disable_v2_behavior()


def discard_fiv_tupple(data):
    for i in range(10):
        # protoc
        data[:, 7 + i * 160] = 0
        # ip and port
        data[:, 10 + i * 160:22 + i * 160] = 0
    return data


def labels_transform(mlist, classes):
    batch_label = np.zeros((len(mlist), classes), dtype="i4")
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label


print("\ndata preparing ... ... ... ")
start = time.time()
Data = read_data()
x_raw = np.array(Data[:, :-1], dtype="float32")
x_raw = discard_fiv_tupple(x_raw)
y_raw = np.array(Data[:, -1], dtype="int32")

data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=0)

print("dataset prepared,cost time:%d" % (time.time() - start))

# ==========================================================================

# parameter
learning_rate = 0.0005
img_shape = 40 * 40
classes_num = 4
batch_size = tf.compat.v1.placeholder(tf.int32, [])
lstm_input_size = 160
lstm_timestep_size = 10
lstm_hidden_layers = 2
train_iter = 3000

# cnn network

_X = tf.compat.v1.placeholder(tf.float32, [None, img_shape])
y = tf.compat.v1.placeholder(tf.int32, [None, classes_num])
keep_prob = tf.compat.v1.placeholder(tf.float32)


def weight_variable(shape):
    return tf.compat.v1.Variable(tf.compat.v1.random.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def max_pool(x):
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


# 40*40*1
cnn_input = tf.compat.v1.reshape(_X, [-1, 40, 40, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
conv_1 = tf.compat.v1.nn.relu(conv2d(cnn_input, W_conv1) + b_conv1)
# 36*36*32
pool_1 = max_pool(conv_1)
# 18*18*32

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
conv_2 = tf.compat.v1.nn.relu(conv2d(pool_1, W_conv2) + b_conv2)
# 16*16*64
pool_2 = max_pool(conv_2)
# 8*8*64 = 4096

W_fc1 = weight_variable([8 * 8 * 64, 1600])
b_fc1 = bias_variable([1600])
pool_2_flat = tf.reshape(pool_2, [-1, 8 * 8 * 64])
cnn_fc1 = tf.compat.v1.matmul(pool_2_flat, W_fc1) + b_fc1
cnn_fc1_drop = tf.compat.v1.nn.dropout(cnn_fc1, keep_prob)

# LSTM network
lstm_input = tf.reshape(cnn_fc1_drop, [-1, lstm_timestep_size, lstm_input_size])

lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_layers, forget_bias=1.0,
                                              state_is_tuple=True, activation=None)

rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]

multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

init_state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)

outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=lstm_input,
                                             initial_state=init_state, dtype=tf.float32, time_major=False)

h_state = state[-1][1]

W_lstm = tf.Variable(tf.compat.v1.random.truncated_normal(shape=[256, classes_num], stddev=0.1), dtype=tf.float32)
bias_lstm = tf.Variable(tf.constant(0.15, dtype=tf.float32, shape=[classes_num]))
logits = tf.compat.v1.matmul(h_state, W_lstm) + bias_lstm

# loss and eval

predictions = {
    "classes": tf.compat.v1.argmax(input=logits, axis=1),
    "probabilities": tf.compat.v1.nn.softmax(logits, name="softmax_tensor")
}

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
loss = tf.compat.v1.losses.mean_squared_error(y, predictions["probabilities"])
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, ).minimize(loss)

correct_prediction = tf.compat.v1.equal(predictions["classes"], tf.argmax(y, axis=1))
accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_prediction, tf.float32))

TP = tf.compat.v1.metrics.true_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FP = tf.compat.v1.metrics.false_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
TN = tf.compat.v1.metrics.true_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FN = tf.compat.v1.metrics.false_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
recall = tf.compat.v1.metrics.recall(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
tf_accuracy = tf.compat.v1.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session()

# begin traing
print("\n" + "=" * 50 + "Benign Trainging" + "=" * 50)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())  # initializing
_batch_size = 128
mydata_train = DataSet(data_train, label_train)
statr = time.time()
for i in range(train_iter):
    batch = mydata_train.next_batch(_batch_size)
    labels = labels_transform(batch[1], classes_num)
    if (i + 1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})

        print("\nthe %dth loop,training accuracy:%f" % (i + 1, train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5,
                                  batch_size: _batch_size})

print("\ntraining finished cost time:%f" % (time.time() - statr))

# batch test：
test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
test_batch_size = 200
preLabel = []
mlabel = []
test_iter = len(data_test) // test_batch_size + 1

mydata_test = DataSet(data_test, label_test)
print("\n" + "=" * 50 + "Benign test" + "=" * 50)
test_start = time.time()
for i in range(test_iter):
    batch = mydata_test.next_batch(test_batch_size)
    mlabel = mlabel + list(batch[1])
    labels = labels_transform(batch[1], classes_num)

    e_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_tp, value_tp = sess.run(TP, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_fp, value_fp = sess.run(FP, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_tn, value_tn = sess.run(TN, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_fn, value_fn = sess.run(FN, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    preLabel = preLabel + list(sess.run(predictions["classes"], feed_dict={_X: batch[0], y: labels, keep_prob: 1.0,
                                                                           batch_size: test_batch_size}))

    test_accuracy = test_accuracy + e_accuracy
    true_positives = true_positives + value_tp
    false_positives = false_positives + value_fp
    true_negatives = true_negatives + value_tn
    false_negatives = false_negatives + value_fn

print("\ntest cost time :%d" % (time.time() - test_start))
print("\n" + "=" * 50 + "Test result" + "=" * 50)
print("\n test accuracy :%f" % (test_accuracy / test_iter))
print("\n true positives :%d" % true_positives)
print("\n false positives :%d" % false_positives)
print("\n true negatives :%d" % true_negatives)
print("\n false negatives :%d" % false_negatives)
print("\n" + "=" * 50 + "  DataSet Describe  " + "=" * 50)
print("\nAll DataSet Number:%s Trainging DataSet Number:%s Test DataSet Number:%s" % (
    len(x_raw), len(data_train), len(data_test)))

mP = true_positives / (true_positives + false_positives)
mR = true_positives / (true_positives + false_negatives)
mF1_score = 2 * mP * mR / (mP + mR)

print("\nPrecision:%f" % mP)
print("\nRecall:%f" % mR)
print("\nF1-Score:%f" % mF1_score)
conmat = confusion_matrix(mlabel, preLabel)
print("\nConfusion Matraics:")
print(conmat)
print(len(mlabel))
