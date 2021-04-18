# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from dataSet import DataSet
import time
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from dataRead import read_data
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.compat.v1.disable_v2_behavior()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)


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
Data = discard_fiv_tupple(Data)
np.random.shuffle(Data)

x_raw = np.array(Data[:, :-1], dtype="float32")
x_raw = discard_fiv_tupple(x_raw)
y_raw = np.array(Data[:, -1], dtype="int32")

data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=0)
print("dataset prepared,cost time:%d" % (time.time() - start))

# ==========================================================================

# parameter
lr = 0.0001
batch_size = tf.compat.v1.placeholder(tf.int32, shape=[])
input_size = 160  # 原本为164
timestep_size = 10
hidden_size = 256
layer_num = 2
class_num = 5
train_iter = 500

_X = tf.compat.v1.placeholder(tf.float32, [None, timestep_size * input_size])
y = tf.compat.v1.placeholder(tf.int32, [None, class_num])
keep_prob = tf.compat.v1.placeholder(tf.float32)

X = tf.reshape(_X, [-1, timestep_size, input_size])

lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_size, forget_bias=1.0,
                                              state_is_tuple=True, activation=None)

rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]

multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

init_state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)

outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=X,
                                             initial_state=init_state, dtype=tf.float32, time_major=False)

h_state = state[-1][1]  # 或者h_state = outputs[:,-1,:]

W = tf.Variable(tf.compat.v1.random.truncated_normal(shape=[hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.15, dtype=tf.float32, shape=[class_num]))
# [batch_size,hidden_size]*[hidden_size,class_num] + [class_num] --> [batch_size,class_num]
logits = tf.compat.v1.matmul(h_state, W) + bias

predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.compat.v1.nn.softmax(logits, name="softmax_tensor")
}

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
loss = tf.compat.v1.losses.mean_squared_error(y, predictions["probabilities"])
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, ).minimize(loss)

correct_prediction = tf.compat.v1.equal(predictions["classes"], tf.argmax(y, axis=1))
accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_prediction, tf.float32))

TP = tf.compat.v1.metrics.true_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FP = tf.compat.v1.metrics.false_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
TN = tf.compat.v1.metrics.true_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FN = tf.compat.v1.metrics.false_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
recall = tf.compat.v1.metrics.recall(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
tf_accuracy = tf.compat.v1.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])

print("\n" + "=" * 50 + "Benign Training" + "=" * 50)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())
_batch_size = 128
mydata_train = DataSet(data_train, label_train)
statr = time.time()
for i in range(train_iter):
    batch = mydata_train.next_batch(_batch_size)
    labels = labels_transform(batch[1], class_num)
    if (i + 1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})

        print("\nthe %dth loop,training accuracy:%f" % (i + 1, train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5,
                                  batch_size: _batch_size})

print("\ntraining finished cost time:%f" % (time.time() - statr))

test_accuracy = 0
preLabel = []
mlabel = []
test_batch_size = 200
test_iter = len(data_test) # test_batch_size + 1
print("\n" + "=" * 50 + "Benign test" + "=" * 50)
test_start = time.time()
begin = 0
end = test_batch_size
while begin < len(data_test):
    end = end if end < len(data_test) else len(data_test)
    data_batch = data_test[begin:end]
    label_batch = label_test[begin:end]
    begin = end
    end += test_batch_size
    mlabel = mlabel + list(label_batch)
    labels = labels_transform(label_batch, class_num)

    e_accuracy = sess.run(accuracy, feed_dict={_X: data_batch, y: labels, keep_prob: 1.0, batch_size: len(data_batch)})

    preLabel = preLabel + list(sess.run(predictions["classes"], feed_dict={_X: data_batch, y: labels, keep_prob: 1.0,
                                                                           batch_size: len(data_batch)}))

    test_accuracy = test_accuracy + e_accuracy

print("\ntest cost time :%d" % (time.time() - test_start))
print("\n" + "=" * 50 + "Test result" + "=" * 50)
print("\n test accuracy :%f" % (test_accuracy / test_iter))
print("\n" + "=" * 50 + "  DataSet Describe  " + "=" * 50)
print("\nAll DataSet Number:%s Trainging DataSet Number:%s Test DataSet Number:%s" % (
    len(x_raw), len(data_train), len(data_test)))

conf_mtx = confusion_matrix(mlabel, preLabel)
print("\nConfusion Matrix:")
print(conf_mtx)

correct_num = []
for r, w in enumerate(conf_mtx):
    correct_num.append(w[r])


def num2str(data):
    str_data = []
    for x in data:
        str_data.append(str(round(x, 4)))
    my_str = ", ".join(str_data)
    return my_str


total_num = []
label_test = label_test.tolist()
correct_num = np.asarray(correct_num)
for i in range(class_num):
    total_num.append(label_test.count(i))
total_num = np.asarray(total_num)
accuracy = accuracy_score(mlabel, preLabel)
top1_precision = precision_score(mlabel, preLabel, average=None)
top1_recall = recall_score(mlabel, preLabel, average=None)
top1_f1_score = f1_score(mlabel, preLabel, average=None)
pr = top1_precision * top1_recall
apr = (correct_num / total_num) * top1_precision * top1_recall
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
report = classification_report(mlabel, preLabel, target_names=target_names)

print("correct_num:", correct_num)
print("total_num:", total_num)
print("accuracy            :", accuracy)
print('e_accuracy		   :', num2str(correct_num / total_num))
print("precision 		   :", num2str(top1_precision))
print("recall 			   :", num2str(top1_recall))
print('f1-socre 		   :', num2str(top1_f1_score))
print('precision_recall    :', num2str(pr))
print('acc_precision_recall:', num2str(apr))
