#!/usr/bin/python
# -*- encoding:utf-8 -*-
# Linear Kernel:
# K(x1, x2) = t(x1) * x2
#
# Gaussian Kernel (RBF):
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
import warnings
import os
import time
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import plot_model
from sklearn import datasets, metrics
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, classification_report, precision_score, \
    f1_score
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")
ops.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# reading data
rootPath = os.path.abspath(os.path.dirname(__file__)).split('application')[0]  # /home/byr/xiaomeng/
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


def read_data():
    feature = []
    for fname in app_label.keys():
        df = pd.read_csv(str(rootPath) + "dataset/labeled_data_S2I/" + fname + ".csv", header=None)
        feature.append(df.to_numpy())
        print(fname + " count:" + str(df.shape[0]))
    Data = np.concatenate([feature[i] for i in range(len(feature))], axis=0)
    np.random.shuffle(Data)
    return Data


# # Gaussian (RBF) kernel
# # 该核函数用矩阵操作来表示
# # 在sq_dists中应用广播加法和减法操作
# # 线性核函数可以表示为：my_kernel=tf.matmul（x_data，tf.transpose（x_data）
# gamma = tf.constant(-50.0)
# dist = tf.reduce_sum(tf.square(x_data), 1)
# dist = tf.reshape(dist, [-1, 1])
# sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
# my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

def mrg_kernel(x):
    # 此处直接将超参数 γ 设定为 1.0；
    gamma = 0.1
    mean = np.mean(x)
    x_marginal = np.zeros((10, 10))
    for i, data in enumerate(x):
        x_marginal[i, i] = np.exp(-gamma * (data - mean) ** 2)

    return x_marginal.flatten()


def cdt_kernel(x):
    # 此处直接将超参数 γ 设定为 1.0；
    gamma = 0.1
    x_conditional = np.zeros((10, 10))
    for i in range(len(x)):
        for j in range(len(x)):
            x_conditional[i, j] = np.exp(-gamma * (x[i] - x[j]) ** 2)
    return x_conditional.flatten()


class S2IMODEL(object):
    def __init__(self, class_num):
        self.input_shape = (10, 10, 6)  # 原本是40*41
        self.class_num = class_num
        self.train_iter = 1000
        self.epochs = 100
        self.batch_size = 100

    def train(self, data_train, label_train):
        data_train = data_train.reshape(-1, 10, 10, 6)
        label_train = keras.utils.to_categorical(label_train, num_classes=self.class_num)
        # data_test = x_test.reshape(-1, 40, 40, 1)
        # label_test = keras.utils.to_categorical(y_test, num_classes=self.class_num)

        # 建立模型
        model = Sequential()
        # 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME, 激活函数relu,输入张量的大小
        model.add(
            Conv2D(32, (5, 5), activation='relu', padding='same', name='layer1_con1', input_shape=self.input_shape))
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', name='layer1_con2'))
        # 池化层,池化核大小２x2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='layer1_pool'))
        # 随机丢弃四分之一的网络连接，防止过拟合
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='layer2_con1'))
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='valid', name='layer2_con2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='layer2_pool'))
        model.add(Dropout(0.25))
        # 全连接层,展开操作,添加隐藏层神经元的数量和激活函数
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.class_num, activation='softmax'))

        # 定义损失值、优化器, 编译模型
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # optimizer = RMSprop(lr=0.001, decay=0.0)  # lr :学习效率，　decay :lr的衰减值
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print('------------ Start Training ------------')
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, mode='auto',
                                                    verbose=1, factor=0.5, min_lr=0.00001)

        # 保存训练参数
        Checkpoint = ModelCheckpoint(filepath='./cnn_model', monitor='val_acc', mode='auto', save_best_only=True)

        # 图片增强
        data_augment = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                                          width_shift_range=0.1, height_shift_range=0.1,
                                          horizontal_flip=False, vertical_flip=False)
        # 训练历史可视化
        history = model.fit(x=data_train, y=label_train, validation_split=0.25, batch_size=self.batch_size,
                            epochs=self.epochs, verbose=1)
        # 模型可视化
        plot_model(model, to_file=str(rootPath) + "figset/S2I/model.png")

        return model, history

    def visualization(self, history):
        # 绘制训练 & 验证的准确率值
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(str(rootPath) + "figset/S2I/Accuracy.png", bbox_inches='tight')
        plt.show()

        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(str(rootPath) + "figset/S2I/Loss.png", bbox_inches='tight')
        plt.show()


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


if __name__ == '__main__':
    print("\ndata preparing ... ... ... ")
    start = time.time()
    Data = read_data()
    x_raw = np.array([np.arange(0, 600)])
    for d in Data:
        x_row = np.concatenate(
            (mrg_kernel(d[0:10]), mrg_kernel(d[10: 20]), mrg_kernel(d[20: 30]), cdt_kernel(d[0: 10]),
             cdt_kernel(d[10: 20]),
             cdt_kernel(d[20: 30])))
        x_raw = np.concatenate((x_raw, x_row.reshape(-1, 600)), axis=0)

    x_raw = np.array(x_raw[1:], dtype="float32")
    y_raw = np.array(Data[:, -1], dtype="int32")
    data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=0)
    label_test = label_test.tolist()
    total_num = []
    for i in range(class_num):
        total_num.append(label_test.count(i))
    total_num = np.asarray(total_num)
    of = open(str(rootPath) + "dataset/result_evaluation/S2I.txt", 'w')
    of.write('******************* S2I ********************\n')
    print("\ndataset prepared, cost time:%d" % (time.time() - start))

    start_time = time.time()
    model, history = S2IMODEL(class_num=class_num).train(data_train, label_train)

    print('training took %fs!' % (time.time() - start_time))
    of.write('training took %fs!\n' % (time.time() - start_time))
    start_time = time.time()

    predict = model.predict_classes(data_test.reshape(-1, 10, 10, 6))

    print('testing took %fs!' % (time.time() - start_time))
    of.write('training took %fs!\n' % (time.time() - start_time))

    S2IMODEL(class_num=class_num).visualization(history)
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    classify_report = metrics.classification_report(label_test, predict, target_names=target_names)  # 使用这种模式
    of.write('classify_report\n')
    of.write(classify_report)
    of.close()
    conf_mtx = metrics.confusion_matrix(label_test, predict)  # 计算混淆矩阵
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
    # print(result)
    pd.DataFrame(result).to_csv(str(rootPath) + "dataset/result_evaluation/S2I.csv", header=None, index=False)