import os
import numpy as np
import pandas as pd

# reading data
rootPath = os.path.abspath(os.path.dirname(__file__)).split('application')[0]  # /home/byr/xiaomeng/
app_label = {
    'Chrome': "0",
    'WeChat': "1",
    'Bilibili': "2",
    'QQMusic': "3",
}


def data2feature(f_name):
    feature = f_name.to_numpy()
    feature = feature[:, 1:]  # first column, first row are index_num
    np.random.shuffle(feature)  # 将流打乱(每行)
    return feature


def read_data():
    feature = []
    for fname in app_label.keys():
        df = pd.read_csv(str(rootPath) + "dataset/labeled_data/" + fname + ".csv")
        feature.append(data2feature(df))
        print(fname + " count:" + str(df.shape[0]))
    Data = np.concatenate([feature[i] for i in range(len(feature))], axis=0)
    np.random.shuffle(Data)
    return Data







