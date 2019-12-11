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

features = pd.read_csv(str(rootPath) + "dataset/false_clf/1128.csv").to_numpy()  # [:, 1:-2]
# for d in features:
#     print(d.reshape([40, 40]))
dataset = tf.data.Dataset.from_tensor_slices({
    "img": features[:, 1:-2],
    "label": features[:, -2],
    "prelabel": features[:, -1]
})
for d in dataset.take(1):
    # dataset = tf.data.Dataset.from_tensor_slices(d.reshape([40, 40]))
    rgb_grayscale = tf.image.grayscale_to_rgb(tf.reshape(d["img"], [40, 40,-1]))
# tf.image.grayscale_to_rgb(tf.reshape(d["img"], [40, 40]))
    rgb_grayscale = rgb_grayscale / 255.
    plt.imshow(rgb_grayscale)

writer = tf.summary.create_file_writer("/tmp/mylogs")
with writer.as_default():
  for step in range(100):
    # other model code would go here
    tf.summary.scalar("my_metric", 0.5, step=step)
    writer.flush()