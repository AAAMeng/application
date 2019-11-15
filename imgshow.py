import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sample_num = 2
# function:随机选取每个文件4个样本绘制灰度图
def data2feature(f_name):
    file_value = f_name.to_numpy()
    # file_value[:, -1] = cla already been labeled
    feature = file_value
    feature = feature[:, 1:]  # first column, first row are index_num
    np.random.shuffle(feature)  # 将流打乱(每行)
    return feature


# type = [
#     pd.read_csv("labeledData/WeChat.csv"),
#     pd.read_csv("labeledData/Bilibili.csv"),
#     pd.read_csv("labeledData/QQMusic.csv"),
#     pd.read_csv("labeledData/Chrome.csv")
# ]

feature = [
        data2feature(pd.read_csv("labeledData/test1.csv"))[:sample_num], # [行，列]左闭右开 取0，1，2，3行
        data2feature(pd.read_csv("labeledData/test2.csv"))[:sample_num],
        data2feature(pd.read_csv("labeledData/test3.csv"))[:sample_num],
]

# imgs = np.concatenate((type_0, type_1, type_2, type_3, type_4, type_5, type_6), axis=0)
imgs = np.concatenate([feature[i] for i in range(len(feature))], axis=0)

fig = plt.figure()

# for i in range(4):
#     ax = fig.add_subplot(4, 1, i + 1)  # (row, col, sub_pos)
#     cla = (type_0[i])[:-1]
#     img_title = (type_0[i])[-1]
#     img = cla.reshape([40, 40])
#     ax.imshow(img, cmap='gray')
#     plt.axis("off")
#     plt.title(img_title)


for i in range(len(feature)):
    for j in range(sample_num):
        ax = fig.add_subplot(len(feature), sample_num, i * sample_num + (j + 1))  # (row, col, sub_pos)
        cla = (imgs[i * sample_num + j])[:-1]
        img_title = (imgs[i * sample_num + j])[-1]
        img = cla.reshape([40, 40])
        ax.imshow(img, cmap='gray')
        plt.axis("off")
        plt.title(img_title)

# plt.savefig('visualization.png')
plt.show()
