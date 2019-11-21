#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   modelTest.py    
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/21 15:05      xm         1.0          None
"""
import tensorflow as tf
import numpy as np
import time
from dataSet import DataSet
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix

proxy_port = tuple(('7a', '31'))
# appName = 'Chrome'
# txt_file = "../dataset/raw_data_simple/" + appName + ".txt"
# csv_file = "../dataset/labeled_data_simple/" + appName + "_simple.csv"
app_label = {
    'WeChat': "1",
    'Bilibili': "2",
    'QQMusic': "3",
    'Chrome': "4",
    # 'app1': "5",
    # 'app2': "6",
    # 'app3': "7",
    # 'app4': "8",
    # 'app5': "9",
    # 'app6': "a",
    # 'app7': "b",
    # 'app8': "c"
}


# =================================DATASET PREPARE=================================
def data2feature(f_name):
    feature = f_name.to_numpy()
    feature = feature[:, 1:]  # first column, first row are index_num
    np.random.shuffle(feature)  # 将流打乱(每行)
    return feature


def discard_fiv_tuple(data):
    for i in range(10):
        # protoc
        data[:, 7 + i * 160] = 0
        # ip and port
        data[:, 10 + i * 160:22 + i * 160] = 0
    return data


def read_data():
    feature = []
    for fname in app_label.keys():
        feature.append(data2feature(pd.read_csv("../dataset/labeled_data/" + fname + ".csv")))
    return feature


start = time.time()
features = read_data()

Data = np.concatenate([features[i] for i in range(len(features))], axis=0)

np.random.shuffle(Data)

x_raw = np.array(Data[:, :-1], dtype="float32")  # data
x_raw = discard_fiv_tuple(x_raw)
y_raw = np.array(Data[:, -1], dtype="int32")  # label

data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=0)
print("\n dataset prepared,cost time:%d" % (time.time() - start))


# ==========================================================================
def labels_transform(mlist, classes):
    batch_label = np.zeros((len(mlist), classes), dtype="i4")
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label


# ============================MODEL TRAIN===================================
# parameter
learning_rate = 0.0005
img_shape = 40 * 40
classes_num = 2
batch_size = tf.placeholder(tf.int32, [])
lstm_input_size = 160
lstm_timestep_size = 10
lstm_hidden_layers = 2
train_iter = 30000

# cnn network
_X = tf.placeholder(tf.float32, [None, img_shape])
y = tf.placeholder(tf.int32, [None, classes_num])
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


# 40*40*1
cnn_input = tf.reshape(_X, [-1, 40, 40, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
conv_1 = tf.nn.relu(conv2d(cnn_input, W_conv1) + b_conv1)
# 36*36*32
pool_1 = max_pool(conv_1)
# 18*18*32

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
conv_2 = tf.nn.relu(conv2d(pool_1, W_conv2) + b_conv2)
# 16*16*64
pool_2 = max_pool(conv_2)
# 8*8*64 = 4096

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
pool_2_flat = tf.reshape(pool_2, [-1, 8 * 8 * 64])
cnn_fc1 = tf.matmul(pool_2_flat, W_fc1) + b_fc1
cnn_fc1_drop = tf.nn.dropout(cnn_fc1, keep_prob)

W_fc2 = weight_variable([1024, classes_num])
b_fc2 = bias_variable([classes_num])
logits = tf.matmul(cnn_fc1_drop, W_fc2) + b_fc2

predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
# loss = tf.losses.mean_squared_error(y,predictions["probabilities"])
y = tf.one_hot(indices=tf.argmax(input=y, axis=1), depth=classes_num, dtype="int32")
loss = tf.losses.softmax_cross_entropy(y, logits)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, ).minimize(loss)

correct_prediction = tf.equal(predictions["classes"], tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

TP = tf.metrics.true_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FP = tf.metrics.false_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
TN = tf.metrics.true_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FN = tf.metrics.false_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
recall = tf.metrics.recall(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
tf_accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

sess = tf.Session()

print("\n" + "=" * 50 + "Benign Training" + "=" * 50)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())  # 初始化局部变量
_batch_size = 128
mydata_train = DataSet(data_train, label_train)
start = time.time()
for i in range(train_iter):
    batch = mydata_train.next_batch(_batch_size)
    labels = labels_transform(batch[1], classes_num)
    if (i + 1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})

        print("\n the %dth loop,training accuracy:%f" % (i + 1, train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5,
                                  batch_size: _batch_size})

print("\n training finished cost time:%f" % (time.time() - start))

# ============================MODEL TEST===================================
test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
test_batch_size = 2000
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
print("\nAll DataSet Number:%s Training DataSet Number:%s Test DataSet Number:%s" % (
    len(x_raw), len(data_train), len(data_test)))

mP = true_positives / (true_positives + false_positives)
mR = true_positives / (true_positives + false_negatives)
mF1_score = 2 * mP * mR / (mP + mR)
print("\nPrecision:%f" % mP)
print("\nRecall:%f" % mR)
print("\nF1-Score:%f" % mF1_score)
conmat = confusion_matrix(mlabel, preLabel)
print("\nConfusion Matrix:")
print(conmat)
print(len(mlabel))
