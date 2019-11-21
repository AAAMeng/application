import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sample_num = 4
app_label = {
    'WeChat': "1",
    'Bilibili': "2",
    'QQMusic': "3",
    'Chrome': "4",
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
    file_value = f_name.to_numpy()
    feature = file_value
    feature = feature[:, 1:]  # first column, first row are index_num
    np.random.shuffle(feature)  # 将流打乱(每行)
    return feature


def read_data():
    feature = []
    for fname in app_label.keys():
        feature.append(data2feature(pd.read_csv("../dataset/labeled_data/" + fname + ".csv"))[:sample_num])
    return feature


# feature = [
#         data2feature(pd.read_csv("labeledData/test1.csv"))[:sample_num], # [行，列]左闭右开 取0，1，2，3行
# ]

# imgs = np.concatenate((type_0, type_1, type_2, type_3, type_4, type_5, type_6), axis=0)
# imgs = np.concatenate([features[i] for i in range(len(features))], axis=0)


if __name__ == "__main__":
    fig = plt.figure()
    features = read_data()
    imgs = np.concatenate([features[i] for i in range(len(features))], axis=0)
    for i in range(len(features)):
        for j in range(sample_num):
            ax = fig.add_subplot(len(features), sample_num, i * sample_num + (j + 1))  # (row, col, sub_pos)
            cla = (imgs[i * sample_num + j])[:-1]
            img_title = (imgs[i * sample_num + j])[-1]
            img = cla.reshape([40, 40])
            ax.imshow(img, cmap='gray')
            plt.axis("off")
            plt.title(img_title)

    # plt.savefig('visualization.png')
    plt.show()
