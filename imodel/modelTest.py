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
import sys
import tensorflow as tf
import numpy as np
import time
import os
from iclass.dataSet import DataSet
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import seaborn as sn
import warnings

warnings.filterwarnings("ignore")
rootPath = os.path.abspath(os.path.dirname(__file__)).split('application')[0]  # /home/byr/xiaomeng/
tf.compat.v1.disable_v2_behavior()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
proxy_port = tuple(('7a', '31'))
# appName = 'Chrome'
# txt_file = "../dataset/raw_data_simple/" + appName + ".txt"
# csv_file = "../dataset/labeled_data_simple/" + appName + "_simple.csv"
app_label = {
    'Chrome': "0",
    'WeChat': "1",
    'Bilibili': "2",
    'QQMusic': "3",
    'SSH': "4",
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
        df = pd.read_csv(str(rootPath) + "dataset/labeled_data_TSS/" + fname + ".csv")
        feature.append(data2feature(df))
        print(fname + " count:" + str(df.shape[0]))

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


# ============================MODEL TRAIN===================================
# TensorFlow 1.x运行机制是将"定义"和"运行"分离.
# 相当于先用程序搭建一个结构(即在内存中构建一个图)，让数据(张量流)按照图中的结构顺序计算，最终运行出结果
# 在TensorFlow 2.x版本中默认使用动态图，但是也可以使用静态图
def labels_transform(mlist, classes):
    batch_label = np.zeros((len(mlist), classes), dtype="i4")
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label


# parameter
learning_rate = 0.0005
img_shape = 40 * 41  # 原本是40*41
classes_num = 5
batch_size = tf.compat.v1.placeholder(tf.int32, [])
train_iter = 1000

# cnn network
_X = tf.compat.v1.placeholder(tf.float32, [None, img_shape])
y = tf.compat.v1.placeholder(tf.int32, [None, classes_num])
keep_prob = tf.compat.v1.placeholder(tf.float32)


# weight initialization, shape is the dimension of output vector, stddev is Standard deviation of normal distribution
# tf.random.truncated_normal:从截断的正态分布中输出随机值, 即tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
# 这样保证了生成的值都在均值附近
def weight_variable(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))


# bias initialization, Constant 1-D 32 tensor populated with scalar value 0.1
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def max_pool(x):
    return tf.compat.v1.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


# Input Layer:40*40*1
# cnn_input = tf.reshape(_X, [-1, 40, 40, 1])
cnn_input = tf.reshape(_X, [-1, 40, 41, 1])
# Conv1 Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
conv_1 = tf.compat.v1.nn.relu(conv2d(cnn_input, W_conv1) + b_conv1)
# 36*36*32
pool_1 = max_pool(conv_1)
# 18*18*32

# Conv2 Layer
# 18*18*32
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
conv_2 = tf.compat.v1.nn.relu(conv2d(pool_1, W_conv2) + b_conv2)
# 16*16*64
pool_2 = max_pool(conv_2)
# 8*8*64 = 4096

# FC Layer(Include flatting and dropout)
W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
pool_2_flat = tf.reshape(pool_2, [-1, 8 * 8 * 64])
cnn_fc1 = tf.compat.v1.matmul(pool_2_flat, W_fc1) + b_fc1
cnn_fc1_drop = tf.compat.v1.nn.dropout(cnn_fc1, rate=1 - keep_prob)

W_fc2 = weight_variable([1024, classes_num])
b_fc2 = bias_variable([classes_num])
logits = tf.compat.v1.matmul(cnn_fc1_drop, W_fc2) + b_fc2

predictions = {
    "classes": tf.compat.v1.argmax(input=logits, axis=1),  # return the index withe largest value across row
    "probabilities": tf.compat.v1.nn.softmax(logits, name="softmax_tensor")
}

# y is the true label of the dataset
# y = tf.compat.v1.one_hot(indices=tf.compat.v1.argmax(input=y, axis=1), depth=classes_num, dtype="int32")
loss = tf.compat.v1.losses.softmax_cross_entropy(y, logits)
# loss = tf.compat.v1.losses.mean_squared_error(y,predictions["probabilities"]) # 均方差

train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.compat.v1.equal(predictions["classes"], tf.compat.v1.argmax(y, axis=1))
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.float32))

# TP = tf.compat.v1.metrics.true_positives(labels=tf.compat.v1.argmax(y, axis=1), predictions=predictions["classes"])
# FP = tf.compat.v1.metrics.false_positives(labels=tf.compat.v1.argmax(y, axis=1), predictions=predictions["classes"])
# TN = tf.compat.v1.metrics.true_negatives(labels=tf.compat.v1.argmax(y, axis=1), predictions=predictions["classes"])
# FN = tf.compat.v1.metrics.false_negatives(labels=tf.compat.v1.argmax(y, axis=1), predictions=predictions["classes"])
# recall = tf.compat.v1.metrics.recall(labels=tf.compat.v1.argmax(y, axis=1), predictions=predictions["classes"])
# precision = tf.compat.v1.metrics.precision_at_k(labels = ,predictions,k=1,class_id=None)
# tf_accuracy = tf.compat.v1.metrics.accuracy(labels=tf.compat.v1.argmax(y, axis=1), predictions=predictions["classes"])

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session()

print("\n" + "=" * 50 + "Begin Training" + "=" * 50)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())  # 初始化局部变量
_batch_size = 128
mydata_train = DataSet(data_train, label_train)
start = time.time()
for i in range(train_iter):
    batch = mydata_train.next_batch(_batch_size)
    labels = labels_transform(batch[1], classes_num)
    # print train_accuracy every 200 iterations
    if (i + 1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})
        print("\n the %dth loop,training accuracy:%f" % (i + 1, train_accuracy))

    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5,
                                  batch_size: _batch_size})

print("\n training finished cost time:%f" % (time.time() - start))
# ============================MODEL TEST===================================
test_accuracy = 0
test_batch_size = 200
preLabel = []
mlabel = []
test_iter = len(data_test) # test_batch_size + 1
begin = 0
end = test_batch_size
# mydata_test = DataSet(data_test, label_test)
print("\n" + "=" * 50 + "Begin test" + "=" * 50)
test_start = time.time()
# batch = mydata_test.next_batch(test_batch_size, shuffle=False)  # no need to shuffle in test period
# batch[0] = data_test
while begin < len(data_test):
    end = end if end < len(data_test) else len(data_test)
    data_batch = data_test[begin:end]
    label_batch = label_test[begin:end]
    begin = end
    end += test_batch_size
    mlabel = mlabel + list(label_batch)
    labels = labels_transform(label_batch, classes_num)

    e_accuracy = sess.run(accuracy, feed_dict={_X: data_batch, y: labels, keep_prob: 1.0, batch_size: len(data_batch)})

    preLabel = preLabel + list(sess.run(predictions["classes"], feed_dict={_X: data_batch, y: labels, keep_prob: 1.0,
                                                                           batch_size: len(data_batch)}))

    test_accuracy = test_accuracy + e_accuracy

# ============================TEST METRIC===================================
print("\n test cost time :%d" % (time.time() - test_start))

print("\n" + "=" * 50 + "Test result" + "=" * 50)
print("\n test accuracy :%f" % (test_accuracy / test_iter))
print("\n" + "=" * 50 + "  DataSet Describe  " + "=" * 50)
print("\nAll DataSet Number:%s ; Training DataSet Number:%s ; Test DataSet Number:%s" % (
    len(x_raw), len(data_train), len(data_test)))
with tf.compat.v1.Session() as sess:
    conf = tf.math.confusion_matrix(mlabel, preLabel, num_classes=classes_num)  # 计算混淆矩阵
    conf_mtx = conf.eval()  # 将 Tensor 转化为 NumPy

# conf_df = pd.DataFrame(conf_numpy, index=app_label.keys(), columns=app_label.keys())  # 将矩阵转化为 DataFrame
# conf_fig = sn.heatmap(conf_df, annot=True, fmt="d", cmap="BuPu")  # 绘制 heatmap

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
for i in range(classes_num):
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

print(report)
print("accuracy            :", accuracy)
print('e_accuracy		   :', num2str(correct_num / total_num))
print("precision 		   :", num2str(top1_precision))
print("recall 			   :", num2str(top1_recall))
print('f1-socre 		   :', num2str(top1_f1_score))
print('precision_recall    :', num2str(pr))
print('acc_precision_recall:', num2str(apr))

with open(str(rootPath) + "application/imodel/MODEL_ORG.txt", 'w', encoding='utf-8') as f:
    f.write("\n****************************ORGMODEL****************************\n")
    f.write(report)
    # f.write("\naccuracy :\n")
    # f.write(str(round(accuracy, 4)))
    # f.write("\neach accuracy:\n")
    # f.write(num2str(correct_num / total_num))
    # f.write('\nprecision :\n')
    # f.write(num2str(top1_precision))
    # f.write("\nrecall :\n")
    # f.write(num2str(top1_recall))
    # f.write('\nf1-socre :\n')
    # f.write(num2str(top1_f1_score))
    f.write("\n*****************************************************************")

print("done!")
