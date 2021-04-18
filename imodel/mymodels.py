#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   modelTest.py
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/04/01 13:53      xm         1.0          None
"""
import os
import warnings
import numpy as np
import keras
from sklearn import metrics
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, classification_report, precision_score, \
    f1_score
from keras.callbacks import LambdaCallback

# 忽略硬件加速的警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

rootPath = os.path.abspath(os.path.dirname(__file__)).split('application')[0]  # /home/byr/xiaomeng/


def knn_classifier(feature, label):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(feature, label)
    return model


def logistic_regression_classifier(feature, label):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='auto')
    model.fit(feature, label)
    return model


def random_forest_classifier(feature, label):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(feature, label)
    return model


def gradient_boosting_classifier(feature, label):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(feature, label)
    return model


# NB
def naive_bayes_classifier(feature, label):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB(priors=None, var_smoothing=1e-09)
    model.fit(feature, label)
    return model


# SVM
def svm_classifier(feature, label):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', gamma='scale', probability=True)
    model.fit(feature, label)
    return model


# MLP
def multilayer_perceptron_classifier(feature, label):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(max_iter=10, hidden_layer_sizes=(100, 50), random_state=1)
    model.fit(feature, label)
    return model


# DT
def decision_tree_classifier(feature, label):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(feature, label)
    return model


class ORGMODEL(object):
    def __init__(self, class_num):
        self.input_shape = (40, 40, 1)  # 原本是40*41
        self.class_num = class_num
        self.train_iter = 1000
        self.epochs = 30
        self.batch_size = 500
        self.learning_rate = 0.0001

    def train(self, data_train, label_train):
        data_train = data_train.reshape(-1, 40, 40, 1)
        label_train = keras.utils.to_categorical(label_train, num_classes=self.class_num)
        # data_test = x_test.reshape(-1, 40, 40, 1)
        # label_test = keras.utils.to_categorical(y_test, num_classes=self.class_num)

        # 建立模型
        model = Sequential()
        # 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
        model.add(
            Conv2D(16, (5, 5), activation='relu', padding='valid', name='layer1_con1', input_shape=self.input_shape))
        # 池化层,池化核大小２x2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='layer1_pool'))
        # 随机丢弃四分之一的网络连接，防止过拟合
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', name='layer2_con1'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='layer2_pool'))
        model.add(Dropout(0.25))
        # 全连接层,展开操作,添加隐藏层神经元的数量和激活函数
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.class_num, activation='softmax'))

        # 定义损失值、优化器, 编译模型
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # optimizer = RMSprop(lr=0.001, decay=0.0)  # lr :学习效率，　decay :lr的衰减值
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # print('------------ Start Training ------------')
        # learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, mode='auto', verbose=1, factor=0.5, min_lr=0.00001)
        # TensorBoard可视化
        # TensorBoard = TensorBoard(log_dir='./log', write_images=1, histogram_freq=1)

        # 保存训练参数
        # Checkpoint = ModelCheckpoint(filepath='cnn_model', monitor='val_acc', mode='auto', save_best_only=True)

        # 图片增强
        # data_augment = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
        #                                   width_shift_range=0.1, height_shift_range=0.1,
        #                                   horizontal_flip=False, vertical_flip=False)

        # model.summary()
        # 训练历史可视化
        history = model.fit(x=data_train, y=label_train, validation_split=0.25, batch_size=self.batch_size,
                            epochs=self.epochs, verbose=0)
        # 模型可视化
        # plot_model(model, to_file=str(rootPath) + "figset/ORG/model.png")

        # print("Train finished! Model is saved in \'my_model.h5\'")
        # model.save("./my_model.h5")
        return model, history

    def visualization(self, history):
        # 绘制训练 & 验证的准确率值

        plt.plot(history.history['accuracy'], '-^')
        plt.plot(history.history['val_accuracy'], '-.')
        plt.title('Training and validation accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['acc', 'val_acc'])

        plt.annotate('(20, ' + str(history.history['accuracy'][19]) + ')', xy=(20, history.history['accuracy'][19]),
                     xycoords='data', xytext=(-15, -15), textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="arc3,rad=.2"))
        plt.scatter([20], [history.history['accuracy'][19]], 10, 'black')

        plt.savefig(str(rootPath) + "figset/ORG/Accuracy.png", bbox_inches='tight')
        plt.show()

        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'], '-^')
        plt.plot(history.history['val_loss'], '-.')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['loss', 'val_loss'])

        plt.savefig(str(rootPath) + "figset/ORG/Loss.png", bbox_inches='tight')
        plt.show()


class TSSMODEL(object):
    def __init__(self, class_num):
        self.input_shape = (40, 40, 3)  # 原本是40*41
        self.class_num = class_num
        self.train_iter = 1000
        self.epochs = 100
        self.batch_size = 200

    def train(self, data_train, label_train):
        data_train = data_train.reshape(-1, 40, 40, 3)
        label_train = keras.utils.to_categorical(label_train, num_classes=self.class_num)

        # 建立模型
        model = Sequential()
        # 第一个卷积层，32个卷积核，大小5*5，卷积模式SAME,激活函数relu,输入张量的大小
        model.add(
            Conv2D(32, (5, 5), activation='relu', padding='valid', name='layer1_con1', input_shape=self.input_shape))
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', name='layer1_con2'))
        # 池化层,池化核大2x2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='layer1_pool'))
        # 随机丢弃四分之一的网络连接，防止过拟合
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid', name='layer2_con1'))
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='valid', name='layer2_con2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='layer2_pool'))
        model.add(Dropout(0.25))
        # 全连接层,展开操作,添加隐藏层神经元的数量和激活函数
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.class_num, activation='softmax'))

        # 定义损失值、优化器, 编译模型
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # optimizer = RMSprop(lr=0.001, decay=0.0)  # lr :学习效率，　decay :lr的衰减值
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # print('------------ Start Training ------------')
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, mode='auto',
                                                    verbose=1, factor=0.5, min_lr=0.00001)
        # TensorBoard可视化
        # TensorBoard = TensorBoard(log_dir='./log', write_images=1, histogram_freq=1)

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
        plot_model(model, to_file=str(rootPath) + "figset/TSS/model.png")
        return model, history

    def visualization(self, history):
        # 绘制训练 & 验证的准确率值
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig(str(rootPath) + "figset/TSS/Accuracy.png", bbox_inches='tight')
        plt.show()

        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig(str(rootPath) + "figset/TSS/Loss.png", bbox_inches='tight')
        plt.show()
