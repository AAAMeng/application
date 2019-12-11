# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd

proxy_port = tuple(('7a', '31'))
# appName = 'Chrome'
# txt_file = "../dataset/raw_data_simple/" + appName + ".txt"
# csv_file = "../dataset/labeled_data_simple/" + appName + "_simple.csv"
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


def read_from_txt(fillna=True):
    """
    Function: read raw data from txt.file of each Application which obtained from Wireshark and filter unnecessary record
    :param fillna: whether or not fill the packet NULL with 0
    :return: a Dict contain all DataFrame of each Application. s.t. data={'app1':app1_df, ...}
    """
    # DataFrame Initialization
    data = {}
    for k, v in app_label.items():
        df1 = pd.read_csv("../../dataset/raw_data_simple/" + k + ".txt", sep="\n", header=None, engine='python',
                          skiprows=lambda x: x % 4 != 2, dtype='str')  # read from csv data
        df1 = df1[0].str.split('  ', expand=True)
        df1 = df1[1].str.split('|', expand=True).drop([0], axis=1)
        if fillna:
            df1.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)  # fill 0 into packet
            df1 = df1.fillna('0')
        data[k] = df1
    return data


def read_from_csv():
    data = {}
    for k, v in app_label.items():
        tmp = pd.read_csv(
            "../../dataset/labeled_data_simple/" + k + "_simple.csv")  # read from csv data and construct ndarray
        data[k] = tmp.reindex(np.random.permutation(tmp.index))  # random sort
    return data


def write_into_csv(data):
    """
    Function: convert hex into dec and write the sessions bytes of each Application into csv.file
    :param data: a Dict contain all df of each Application data={'app1':app1_df, ...}
    """
    for fname, df1 in data.items():
        hex_list = df1.to_numpy()
        dec_list = [[int(hex_list[i][j], 16) for j in range(len(hex_list[i]))] for i in range(len(hex_list))]

        df1 = pd.DataFrame(dec_list)
        df1.to_csv("../../dataset/labeled_data_simple/" + fname + "_simple.csv")


if __name__ == "__main__":
    pd.set_option('mode.chained_assignment', None)
    start = time.time()
    # data file operation
    # mydata = read_from_txt()
    # write_into_csv(mydata)
    # ---------------------------------------------------------------------------------

    # data = read_from_txt(fillna=False)
    # print(data['Chrome'].count(axis=1))

    #     print(mydata.get('Bilibili').iloc[0:500, 20])

    print("\nPreprocessed finished cost time:%f" % (time.time() - start))
