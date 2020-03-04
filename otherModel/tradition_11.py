from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
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


def decision_tree_classifier(feature, label):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(feature, label)
    return model


def gradient_boosting_classifier(feature, label):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(feature, label)
    return model


def svm_classifier(feature, label):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', gamma='scale', probability=True)
    model.fit(feature, label)
    return model


def data2feature(f_name, cla):
    file_value = f_name.values
    file_value[:, -1] = cla
    feature = file_value
    feature = feature[:, 1:]
    np.random.shuffle(feature)
    return feature


if __name__ == '__main__':
    test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    classifiers = {'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'GBDT': gradient_boosting_classifier
                   }

    print("\ndata preparing ... ... ... ")
    start = time.time()
    Data = read_data()
    np.random.shuffle(Data)

    x_raw = np.array(Data[:, 1:-1], dtype="float32")
    y_raw = np.array(Data[:, -1], dtype="int32")

    data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=0)

    of = open(str(rootPath) + "application/otherModel/tradition.txt", 'w')
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        of.write('******************* %s ********************\n' % classifier)
        start_time = time.time()
        model = classifiers[classifier](data_train, label_train)
        print('training took %fs!' % (time.time() - start_time))
        of.write('training took %fs!\n' % (time.time() - start_time))
        predict = model.predict(data_test)
        of.write('classify_report\n')
        classify_report = metrics.classification_report(label_test, predict)  # 使用这种模式
        print("\nAccuracy:%f" % metrics.accuracy_score(label_test, predict))
        print("\nPrecision:%f" % metrics.precision_score(label_test, predict, average="macro"))
        print("\nRecall:%f" % metrics.recall_score(label_test, predict, average="macro"))
        print("\nF1-score:%f" % metrics.f1_score(label_test, predict, average="macro"))
        print("\nconfusion matrix:")
        print("\n%s" % metrics.confusion_matrix(label_test, predict))
        print(classify_report)
        of.write(classify_report)
    of.close()
