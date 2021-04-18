import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

rootPath = os.path.abspath(os.path.dirname(__file__)).split('application')[0]  # /home/byr/xiaomeng/
# test_classifiers = ['SVM', 'MLP', 'ORG', 'TSS']
test_classifiers = ['ml-SVM', 'ml-MLP', 'ml-NB', 'ml-DT', 'ORG', 'TSS']
model_num = len(test_classifiers)
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

feature = []
for classifier in test_classifiers:
    df = pd.read_csv(str(rootPath) + "dataset/result_evaluation/" + classifier + ".csv", header=None)
    feature.append(df.to_numpy())
Data = np.concatenate([feature[i] for i in range(len(feature))], axis=0)
evaluations = ['Accuracy', 'Precision', 'Recall', 'F1-measure']

for i, e in enumerate(evaluations):
    x_data = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']

    plt.figure()

    ln1, = plt.plot(x_data, Data[0 + i], 'b-o')
    ln2, = plt.plot(x_data, Data[4 + i], 'g-o')
    ln3, = plt.plot(x_data, Data[8 + i], 'r-o')
    ln4, = plt.plot(x_data, Data[12 + i], 'c-o')
    ln5, = plt.plot(x_data, Data[16 + i], 'm-o')
    ln6, = plt.plot(x_data, Data[20 + i], 'y-o')
    #
    plt.title(e)  # 设置标题及字体

    # plots = [plt.plot(x_data, Data[0 + i], 'b-o'), plt.plot(x_data, Data[4 + i], 'g-o'),
    #          plt.plot(x_data, Data[8 + i], 'r-o'), plt.plot(x_data, Data[12 + i], 'c-o'),
    #          plt.plot(x_data, Data[16 + i], 'm-o'),plt.plot(x_data, Data[20 + i], 'y-o')]
    plt.legend(handles=[ln1, ln2, ln3, ln4, ln5, ln6], labels=test_classifiers)
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示

    plt.tight_layout()
    plt.savefig(str(rootPath) + "figset/" + e + ".png")
