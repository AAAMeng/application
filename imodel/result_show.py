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

# import time
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


# feature = [
#         data2feature(pd.read_csv("labeledData/test1.csv"))[:sample_num], # [行，列]左闭右开 取0，1，2，3行
# ]

# imgs = np.concatenate((type_0, type_1, type_2, type_3, type_4, type_5, type_6), axis=0)
# imgs = np.concatenate([features[i] for i in range(len(features))], axis=0)


if __name__ == "__main__":
    # 绘制误分类样本的灰度图并与相关分类对比
    fig = plt.figure(dpi=600)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # read the labeled_data
    features = read_data_mean()
    error_feature = []
    error_df = pd.read_csv("../../dataset/false_clf/1128.csv")  # .to_numpy()[:, 1:]
    grouped_by_label = error_df.groupby(['1600', '1601'])
    # print(grouped_by_label.size().max())
    row = 0
    for name, groups in grouped_by_label:
        # print(type(name))
        error_feature = groups.to_numpy()
        for i in range(len(error_feature)):
            ax = fig.add_subplot(grouped_by_label.__len__(), grouped_by_label.size().max() + 2,
                                 row * (grouped_by_label.size().max() + 2) + (i + 3))  # (row, col, sub_pos)
            data = error_feature[i, 1:-2]
            img = data.reshape([40, 40])
            img_title = error_feature[i, -2]  # true label
            plt.axis("off")
            plt.title(int(img_title))
            ax.imshow(img, cmap='gray')

        # Label image
        ax = fig.add_subplot(grouped_by_label.__len__(), grouped_by_label.size().max() + 2,
                             row * (grouped_by_label.size().max() + 2) + 1)
        data = features[name[0]][0:-1]  # name[0] is the true label
        img = data.reshape([40, 40])
        img_title = name[0]
        plt.title(int(img_title))
        plt.axis("off")
        ax.imshow(img, cmap='gray')

        # PreLabel image
        ax = fig.add_subplot(grouped_by_label.__len__(), grouped_by_label.size().max() + 2,
                             row * (grouped_by_label.size().max() + 2) + 2)
        data = features[name[1]][0:-1]  # name[1] is the predict label
        img = data.reshape([40, 40])
        img_title = name[1]
        plt.title(int(img_title))
        ax.imshow(img, cmap='gray')
        plt.axis("off")
        row = row + 1
plt.tight_layout()
plt.savefig('fig.png', bbox_inches='tight')
plt.show()
