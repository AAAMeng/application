from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale
import numpy as np
import time
from dataRead import read_data
import warnings
import os

warnings.filterwarnings("ignore")
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


def data2feature(f_name, cla):
    file_value = f_name.values
    file_value[:, -1] = cla
    feature = file_value
    feature = feature[:, 1:]
    np.random.shuffle(feature)
    return feature


def num2str(data):
    str_data = []
    for x in data:
        str_data.append(str(round(x, 4)))
    my_str = ", ".join(str_data)
    return my_str


if __name__ == '__main__':
    classes_num = 6
    test_classifiers = ['NB', 'SVM', 'MLP', 'DT']
    classifiers = {'NB': naive_bayes_classifier,
                   'SVM': svm_classifier,
                   'MLP': multilayer_perceptron_classifier,
                   'DT': decision_tree_classifier,
                   }
    print("\ndata preparing ... ... ... ")
    start = time.time()
    Data = read_data()
    np.random.shuffle(Data)

    x_raw = np.array(Data[:, 1:-1], dtype="float32")
    x_raw = scale(x_raw)
    y_raw = np.array(Data[:, -1], dtype="int32")

    data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=0)
    label_test = label_test.tolist()

    total_num = []
    for i in range(classes_num):
        total_num.append(label_test.count(i))
    total_num = np.asarray(total_num)
    of = open(str(rootPath) + "application/imodel/tradition.txt", 'w')
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        of.write('******************* %s ********************\n' % classifier)
        start_time = time.time()
        model = classifiers[classifier](data_train, label_train)
        print('training took %fs!' % (time.time() - start_time))
        of.write('training took %fs!\n' % (time.time() - start_time))
        start_time = time.time()
        predict = model.predict(data_test)
        print(predict)
        print('testing took %fs!' % (time.time() - start_time))
        of.write('classify_report\n')
        target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
        classify_report = metrics.classification_report(label_test, predict, target_names=target_names)  # 使用这种模式

        conf_mtx = metrics.confusion_matrix(label_test, predict)  # 计算混淆矩阵
        # conf_mtx = conf.eval()  # 将 Tensor 转化为 NumPy
        print(conf_mtx)

        correct_num = []
        for r, w in enumerate(conf_mtx):
            correct_num.append(w[r])

        correct_num = np.asarray(correct_num)

        accuracy = accuracy_score(label_test, predict)
        top1_precision = precision_score(label_test, predict, average=None)
        top1_recall = recall_score(label_test, predict, average=None)
        top1_f1_score = f1_score(label_test, predict, average=None)
        pr = top1_precision * top1_recall
        apr = (correct_num / total_num) * top1_precision * top1_recall

        # print(report)
        print("correct_num:", correct_num)
        print("total_num:", total_num)
        print("accuracy            :", accuracy)
        print('e_accuracy		   :', num2str(correct_num / total_num))
        print("precision 		   :", num2str(top1_precision))
        print("recall 			   :", num2str(top1_recall))
        print('f1-socre 		   :', num2str(top1_f1_score))
        print('precision_recall    :', num2str(pr))
        print('acc_precision_recall:', num2str(apr))
        # print("\nAccuracy:%f" % metrics.accuracy_score(label_test, predict))
        # print("\nPrecision:%f" % metrics.precision_score(label_test, predict, average="macro"))
        # print("\nRecall:%f" % metrics.recall_score(label_test, predict, average="macro"))
        # print("\nF1-score:%f" % metrics.f1_score(label_test, predict, average="macro"))
        # print("\nconfusion matrix:")
        # print("\n%s" % metrics.confusion_matrix(label_test, predict))
        # print(classify_report)
        of.write(classify_report)
    of.close()
