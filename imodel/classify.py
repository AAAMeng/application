#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   modelTest.py
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/04/01 13:01      xm         1.0          None
"""
import sys
import os
import time
import numpy as np
import seaborn as sn
import warnings
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from iclass.dataSet import DataSet
from sklearn.model_selection import train_test_split
from mymodels import ORGMODEL, TSSMODEL, naive_bayes_classifier, svm_classifier, multilayer_perceptron_classifier, \
    decision_tree_classifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, classification_report, precision_score, \
    f1_score

warnings.filterwarnings("ignore")
rootPath = os.path.abspath(os.path.dirname(__file__)).split('application')[0]  # /home/byr/xiaomeng/
# tf.compat.v1.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
proxy_port = tuple(('7a', '31'))

app_label = {
    'Chrome': "0",
    'Bilibili': "1",
    'WeChat': "2",
    'QQMusic': "3",
    # 'SSH': "4",
    'IQIYI': "4",
    'YouKu': "5",
}

class_num = len(app_label)


# =================================DATASET PREPARE=================================
def discard_fiv_tuple(data):
    for i in range(10):
        # protoc
        data[:, 807 + i * 160] = 0
        # ip and port
        data[:, 810 + i * 160:822 + i * 160] = 0
    return data


def read_data():
    # read from TSS
    feature = []
    for fname in app_label.keys():
        df = pd.read_csv(str(rootPath) + "dataset/labeled_data_TSS/" + fname + ".csv", header=None)
        # timeData=(0,399) + sptlData=(400,799) + dData=(800,2400) + label(2401)
        feature.append(df.to_numpy())
        # print(fname + " count:" + str(df.shape[0]))
    return feature


def num2str(data):
    str_data = []
    for x in data:
        str_data.append(str(round(x, 4)))
    my_str = ", ".join(str_data)
    return my_str


def num2arr(data):
    str_data = []
    for x in data:
        str_data.append(round(x, 4))
    return str_data


print("=" * 30 + "Load Data" + "=" * 30)
print('\nDataSet preparing, waiting ...... ')
start = time.time()
features = read_data()

Data = np.concatenate([features[i] for i in range(len(features))], axis=0)
np.random.shuffle(Data)
x_raw = np.array(Data[:, :-1], dtype="float32")  # data
x_raw = discard_fiv_tuple(x_raw)
y_raw = np.array(Data[:, -1], dtype="int32")  # label

x_raw = np.concatenate((x_raw, x_raw, x_raw, x_raw, x_raw, x_raw, x_raw, x_raw, x_raw, x_raw), axis=0)
y_raw = np.concatenate((y_raw, y_raw, y_raw, y_raw, y_raw, y_raw, y_raw, y_raw, y_raw, y_raw), axis=0)
print("total data:", len(x_raw))

data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=0)
data_train_TSS = np.concatenate((np.pad(data_train[:, 0:400], ((0, 0), (0, 1200)), 'constant'),
                                 np.pad(data_train[:, 400:800], ((0, 0), (0, 1200)), 'constant'), data_train[:, 800:]),
                                axis=1)
data_test_TSS = np.concatenate((np.pad(data_test[:, 0:400], ((0, 0), (0, 1200)), 'constant'),
                                np.pad(data_test[:, 400:800], ((0, 0), (0, 1200)), 'constant'), data_test[:, 800:]),
                               axis=1)
# data_test = np.concatenate((x_raw, x_raw[0:9101]), axis=0)
# label_test = np.concatenate((y_raw, y_raw[0:9101]), axis=0)

label_test = label_test.tolist()
total_num = []
for i in range(class_num):
    total_num.append(label_test.count(i))
total_num = np.asarray(total_num)
print('\nDataSet prepared !')
# print('\nDataSet prepared, cost time :', time.time() - start)
# ========================= Model Train ============================
test_classifiers = ['NB', 'SVM', 'MLP', 'DT']
# test_classifiers = ['TSS']
classifiers = {'NB': naive_bayes_classifier,
               'SVM': svm_classifier,
               'MLP': multilayer_perceptron_classifier,
               'DT': decision_tree_classifier,
               'ORG': TSSMODEL(class_num=class_num),
               'TSS': ORGMODEL(class_num=class_num)
               }
# print("=" * 50 + "Begin Training" + "=" * 50)

for classifier in test_classifiers:
    of = open(str(rootPath) + "dataset/result_evaluation/" + classifier + ".txt", 'w')
    # print('******************* %s ********************' % classifier)
    # of.write('******************* %s ********************\n' % classifier)
    start_time = time.time()
    if classifier == 'ORG':
        model, history = classifiers[classifier].train(data_train_TSS, label_train)
        print('training took %fs!' % (time.time() - start_time))
        of.write('training took %fs!\n' % (time.time() - start_time))
        classifiers[classifier].visualization(history)
        start_time = time.time()
        predict = model.predict_classes(data_test_TSS.reshape(-1, 40, 40, 3))
    elif classifier == 'TSS':
        print("\n" + "=" * 30 + "Load Model" + "=" * 30)
        model, history = classifiers[classifier].train(data_train[:, 800:], label_train)
        print('training took %fs!' % (time.time() - start_time))
        # of.write('training took %fs!\n' % (time.time() - start_time))
        # classifiers[classifier].visualization(history)
        print('\nModel preparing, waiting ...... ')
        print('\nModel prepared !')
        start_time = time.time()
        print("\n" + "=" * 30 + "Start classifying" + "=" * 30)
        predict = model.predict_classes(data_test[:, 800:].reshape(-1, 40, 40, 1))
    else:
        # for ML
        model = classifiers[classifier](data_train[:, 800:], label_train)
        print('training took %fs!' % (time.time() - start_time))
        of.write('training took %fs!\n' % (time.time() - start_time))
        start_time = time.time()
        predict = model.predict(data_test[:, 800:])

        # for ML
        # model = classifiers[classifier](data_train, label_train)
        # print('training took %fs!' % (time.time() - start_time))
        # start_time = time.time()
        # predict = model.predict(data_test)
    # print('classify on %d samples took %fs!' % (len(data_test), time.time() - start_time))
    print('classify on %d samples took %fs!' % (len(data_test), time.time() - start_time))
    # target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    # classify_report = metrics.classification_report(label_test, predict, target_names=target_names)  # 使用这种模式

    conf_mtx = metrics.confusion_matrix(label_test, predict)  # 计算混淆矩阵
    print("confusion_matrix:")
    print(conf_mtx)

    correct_num = []
    for r, w in enumerate(conf_mtx):
        correct_num.append(w[r])

    correct_num = np.asarray(correct_num)

    a_accuracy = accuracy_score(label_test, predict)
    c_accuracy = correct_num / total_num
    c_precision = precision_score(label_test, predict, average=None)
    c_recall = recall_score(label_test, predict, average=None)
    c_f1measure = f1_score(label_test, predict, average=None)
    c_pr = c_precision * c_recall
    c_apr = c_accuracy * c_precision * c_recall

    # c_accuracy = c_recall
    result = [num2arr(c_accuracy), num2arr(c_precision), num2arr(c_recall), num2arr(c_f1measure)]

    pd.DataFrame(result).to_csv(str(rootPath) + "dataset/result_evaluation/" + classifier + ".csv", header=None,
                                index=False)

    # print(report)
    # print("correct_num:", correct_num)
    # print("total_num:", total_num)
    #
    # print('category_accuracy		   :', num2str(c_accuracy))
    # print("category_precision 		   :", num2str(c_precision))
    # print("category_recall 			 :", num2str(c_recall))
    # print('category_f1-measure 		 :', num2str(c_f1measure))
    # print('category_precision_recall   :', num2str(c_pr))
    # print('category_acc_pre_recall     :', num2str(c_apr))
    print("overall_accuracy:", a_accuracy)
    # print("Accuracy:%f" % metrics.accuracy_score(label_test, predict))
    # print("Precision:%f" % metrics.precision_score(label_test, predict, average="micro"))
    # print("Recall:%f" % metrics.recall_score(label_test, predict, average="micro"))
    # print("F1-score:%f" % metrics.f1_score(label_test, predict, average="micro"))
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    # target_names = ['QQ', 'WeChat', 'Chrome', 'QQMusic', 'WangYiYun', 'KuGou', 'QQVideo', 'IQIYI', 'YouKu', 'Bilibili']
    classify_report = metrics.classification_report(label_test, predict, target_names=target_names)  # 使用这种模式
    of.write(classify_report)
    of.close()
