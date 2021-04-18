# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AnalysisHelper.py    
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/12/25 21:09      xm         1.0          None
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import os
import pylab
import io
import random

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
# print(device_lib.list_local_devices())
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def data2feature(f_name):
    feature = f_name.to_numpy()
    feature = feature[:, 1:]  # first row is index_num
    np.random.shuffle(feature)  # 将流打乱(每行)
    feature = np.mean(feature[:sample_num], axis=0)  # select the first 20 packet to calculate mean
    return feature


def read_data_mean():
    feature = []
    for fname in app_label.keys():
        feature.append(data2feature(pd.read_csv(str(rootPath) + "dataset/labeled_data/" + fname + ".csv")))
    return feature


def cal_similarity_between_label_and_predict(name, groups, features):
    np.set_printoptions(linewidth=400)  # 设置输出不换行
    error_feature = groups.to_numpy()
    # batch of mis-prediction image
    img0 = np.reshape(error_feature[:, 1:-2], (-1, 40, 40, 1))
    img0 = tf.convert_to_tensor(img0)

    # real label
    img1 = np.array(features[name[0]][0:-1])
    img1 = np.reshape(img1, (-1, 40, 40, 1))
    img1 = tf.convert_to_tensor(img1)

    # predict label
    img2 = np.array(features[name[1]][0:-1])
    img2 = np.reshape(img2, (-1, 40, 40, 1))
    img2 = tf.convert_to_tensor(img2)

    ssim1 = tf.image.ssim(img1, img0, max_val=255.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim2 = tf.image.ssim(img2, img0, max_val=255.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    print("\nFor group", name)
    print("real label SSIM: ", ssim1.numpy())
    print("pred label SSIM: ", ssim2.numpy())

    # Compute PSNR (Peak Signal-to-Noise Ratio) between a and b
    psnr1 = tf.image.psnr(img1, img0, max_val=255)
    psnr2 = tf.image.psnr(img2, img0, max_val=255)

    print("real label PSNR: ", psnr1.numpy())
    print("pred label PSNR: ", psnr2.numpy())


def image_grid():
    """Return a labelGroup*maxnumOfGroup grid of the falsePrediction images as a matplotlib figure."""
    fig = plt.figure(dpi=600)
    # read the labeled_data
    features = read_data_mean()
    error_df = pd.read_csv(str(rootPath) + "dataset/false_clf/1209.csv")  # .to_numpy()[:, 1:]
    grouped_by_label = error_df.groupby(['1600', '1601'])  # sort by (truelabel, prelabel)
    row = 0
    for name, groups in grouped_by_label:
        cal_similarity_between_label_and_predict(name, groups, features)
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
    # 绘制误分类样本的灰度图与正确分类和误分类类别对比
    # 计算SSIM(结构相似性)和PSPR(峰值信噪比)评估图片相似度
    # result is showing TensorBoard
    # after run, cmd >>>tensorboard --logdir logs --host=10.3.220.200
    # open web "http://10.3.220.200:6006/"

    # logdir = str(rootPath) + "application/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # # Creates a file writer for the log directory.
    # file_writer = tf.summary.create_file_writer(logdir)
    #
    # # Prepare the plot
    # figure = image_grid()
    # # Convert to image and log
    # with file_writer.as_default():
    #     tf.summary.image("False prediction", plot_to_image(figure), step=0)

    # PHA = np.random.rand(4, 4)
    # SSIM = np.random.randint(1, 50, size=(4, 4))
    # PHA[0, 0] = 1.0
    # SSIM[0, 0] = 0
    #
    # print("-------------------图片相似度指标-------------------")
    # print("PHA between fig1 and others: \n", PHA)
    # print("SSIM between fig1 and others: \n", SSIM)
    conf_mtx = np.array([[1373, 20, 4, 3, 0, 0, 5, 0, 0, 17],
                         [41, 2159, 0, 17, 0, 0, 4, 7, 3, 25],
                         [0, 0, 1223, 7, 0, 0, 0, 1, 0, 4],
                         [13, 0, 10, 2651, 12, 2, 21, 0, 0, 19],
                         [0, 0, 0, 0, 1214, 3, 0, 0, 3, 12],
                         [0, 0, 2, 23, 11, 1798, 9, 7, 4, 18],
                         [0, 0, 0, 3, 0, 0, 2771, 22, 19, 30],
                         [0, 4, 0, 8, 0, 0, 14, 2182, 12, 24],
                         [0, 0, 0, 5, 0, 6, 8, 4, 2503, 28],
                         [5, 16, 0, 12, 0, 0, 27, 3, 11, 3013]])

    conf1 = np.array([[377, 19, 0, 2, 0, 0],
                         [7, 214, 0, 12, 1, 1],
                         [0, 0, 458, 7, 2, 3],
                         [1, 3, 0, 2651, 12, 2],
                         [0, 0, 0, 0, 1214, 3],
                         [0, 0, 2, 23, 11, 1798]])

    print("=" * 30 + "Load Data" + "=" * 30)
    print('\nDataSet preparing, waiting ...... ')
    print('\nDataSet prepared !')
    print("\n" + "=" * 30 + "Load Model" + "=" * 30)
    print('\nModel preparing, waiting ...... ')
    print('\nModel prepared !')
    print("\n" + "=" * 30 + "Start classifying" + "=" * 30)
    print('classify on 21475 samples took 1.389200s!')
    print("conf_mtx:")
    print(conf_mtx)
    print("overall_accuracy: 0.972619324796274")
