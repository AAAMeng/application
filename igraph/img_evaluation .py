#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   compare_img.py    
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/12/5 10:38      xm         1.0          None
"""
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from iclass.dataSet import DataSet
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt
import pylab

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
        feature.append(data2feature(pd.read_csv(str(rootPath) + "dataset/labeled_data/" + fname + ".csv")))
    return feature


if __name__ == "__main__":
    # error_feature = pd.read_csv(str(rootPath) + "dataset/false_clf/1128.csv").to_numpy()  # [:, 1:-2]
    # convert ndarray to Tensors
    # dataset = tf.data.Dataset.from_tensor_slices({
    #     "img": features[:, 1:-2],
    #     "label": features[:, -2],
    #     "prelabel": features[:, -1]
    # })

    # read the labeled_data
    features = read_data_mean()
    error_df = pd.read_csv(str(rootPath) + "dataset/false_clf/1209.csv")  # .to_numpy()[:, 1:]
    grouped_by_label = error_df.groupby(['1600', '1601'])  # sort by (truelabel, prelabel)
    row = 0
    for name, groups in grouped_by_label:
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



