#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   result_show.py    
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/12/4 13:49      xm         1.0          None
"""

# 绘制列之间的相似度热度图

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
import time
from iclass.dataSet import DataSet
from sklearn.model_selection import train_test_split
import os
import pylab
import io

rootPath = os.path.abspath(os.path.dirname(__file__)).split('application')[0]  # /home/byr/xiaomeng/

sample_num = 20
app_label = {
    'Chrome': "0",
    'WeChat': "1",
    'Bilibili': "2",
    'QQMusic': "3",
    # 'app1': "5",
    # 'app2': "6",
    # 'app3': "7",
    # 'app4': "8",
    # 'app5': "9",
    # 'app6': "a",
    # 'app7': "b",
    # 'app8': "c"
}


# function:随机选取每个文件4个样本绘制灰度图
def data2feature(f_name):
    feature = f_name.to_numpy()
    feature = feature[:, 1:]  # first row is index_num
    np.random.shuffle(feature)  # 将流打乱(每行)
    feature = np.mean(feature[:sample_num], axis=0)  # select the first 20 packet to calculate mean
    return feature


def read_data_mean():
    feature = []
    for fname in app_label.keys():
        feature.append(data2feature(pd.read_csv("../../dataset/labeled_data/" + fname + ".csv")))
    return feature


def image_grid():
    """Return a labelGroup*maxnumOfGroup grid of the falsePrediction images as a matplotlib figure."""
    fig = plt.figure(dpi=600)
    # read the labeled_data
    features = read_data_mean()
    error_df = pd.read_csv("../../dataset/false_clf/1128.csv")  # .to_numpy()[:, 1:]
    grouped_by_label = error_df.groupby(['1600', '1601'])  # sort by (truelabel, prelabel)
    row = 0
    for name, groups in grouped_by_label:
        error_feature = groups.to_numpy()
        for i in range(len(error_feature)):
            plt.subplot(grouped_by_label.__len__(), grouped_by_label.size().max() + 2,
                            row * (grouped_by_label.size().max() + 2) + (i + 3))  # (row, col, sub_pos)
            data = error_feature[i, 1:-2]
            img = data.reshape([40, 40])
            plt.title(int(error_feature[i, -2]), fontsize=5, pad=5)  # true label
            plt.axis("off")
            plt.imshow(img, cmap='gray')

        # Label image
        plt.subplot(grouped_by_label.__len__(), grouped_by_label.size().max() + 2,
                        row * (grouped_by_label.size().max() + 2) + 1)
        data = features[name[0]][0:-1]
        img = data.reshape([40, 40])
        plt.title(int(name[0]), fontsize=5, pad=5)  # name[0] is the true label
        plt.axis("off")
        plt.imshow(img, cmap='gray')

        # PreLabel image
        plt.subplot(grouped_by_label.__len__(), grouped_by_label.size().max() + 2,
                    row * (grouped_by_label.size().max() + 2) + 2)
        data = features[name[1]][0:-1]
        img = data.reshape([40, 40])
        plt.title(int(name[1]), fontsize=5, pad=5)  # name[1] is the predict label
        plt.axis("off")
        plt.imshow(img, cmap='gray')
        row = row + 1
    plt.tight_layout()
    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=1)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


if __name__ == "__main__":
    # 绘制误分类样本的灰度图并与相关分类对比
    logdir = str(rootPath) + "application/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    # Prepare the plot
    figure = image_grid()
    # Convert to image and log
    with file_writer.as_default():
        tf.summary.image("False prediction", plot_to_image(figure), step=0)
