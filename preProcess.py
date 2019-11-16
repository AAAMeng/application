# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd

proxy_port = tuple(('7a', '31'))
appName = 'Chrome'
txt_file = "raw_data/" + appName + ".txt"
csv_file = "labeled_data/" + appName + ".csv"
app_label = {
    'WeChat': "1",
    'Bilibili': "2",
    'QQMusic': "3",
    'Chrome': "4",
    'app1': "5",
    'app2': "6",
    'app3': "7",
    'app4': "8",
    'app5': "9",
    'app6': "a",
    'app7': "b",
    'app8': "c"
}


def read_from_txt():
    """
    Function: read raw data from txt.file which obtained from Wireshark and filter unnecessary record
    :return: a DataFrame of raw data
    """
    # DataFrame Initialization
    df1 = pd.read_csv(txt_file, sep="\n", header=None, engine='python', skiprows=lambda x: x % 4 != 2, dtype='str')
    df1 = df1[0].str.split('  ', expand=True)
    df1 = df1[1].str.split('|', expand=True).drop([0], axis=1)

    return df1


def session_merge(dict1, dict2):
    """
    Function: merge the packets which belong to same session
    :param dict1: packet index sort by src.port
    :param dict2: packet index sort by dst.port
    :return: merge result stored in dict1
    """
    for k, v in dict1.items():
        if dict2.__contains__(k):
            dict1[k] = np.sort(np.unique(np.concatenate((dict1[k], dict2[k]), axis=0)))


def rawdata_construct(df1, list1, sess_size=10, pck_str=16, pck_len=160):
    """
    Function: transform raw data into intended format(1 row = 1 session with 10 packets and 160bytes/packet)
    :param df1: raw data
    :param list1: array of the index which belongs to same session
    :param sess_size: number of packet selected from one session
    :param pck_str: packet start position
    :param pck_len: packet length(fixed)
    :return: sorted packets in session in hex
    """
    # slice 16th~175th, column index keep same
    df1 = df1.iloc[:, pck_str:pck_str + pck_len]
    df1.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)    # fill 0 into packet
    df1 = df1.fillna('0')
    sessions = pd.DataFrame()
    for k, v in list1.items():
        if k == proxy_port:
            continue
        else:
            s = pd.concat([df1.iloc[i] for i in v[0:sess_size]], axis=0, ignore_index=True)
            sessions = pd.concat([sessions, s], axis=1, ignore_index=True)

    return label_data(sessions.T, sess_size, pck_len)


def label_data(df1, sess_size, pck_len):
    df1.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)  # fill 0 into session
    df1 = df1.fillna('0')
    df1[sess_size * pck_len] = app_label[appName]
    return df1


def write_into_csv(df1):
    """
    Function: convert hex into dec and write the sessions bytes of one Application into csv.file
    :param df1: a DataFrame of sessions in hex
    """
    hex_list = df1.to_numpy()
    dec_list = [[int(hex_list[i][j], 16) for j in range(len(hex_list[i]))] for i in range(len(hex_list))]

    del df1
    df1 = pd.DataFrame(dec_list)
    df1.to_csv(csv_file)


if __name__ == "__main__":
    pd.set_option('mode.chained_assignment', None)
    start = time.time()

    df = read_from_txt()
    groupedBySrc = df.groupby([35, 36])  # return GroupBy Object
    groupedByDst = df.groupby([37, 38])
    listBySrc = groupedBySrc.indices
    listByDst = groupedByDst.indices

    session_merge(listBySrc, listByDst)
    appDF = rawdata_construct(df, listBySrc)
    write_into_csv(appDF)
    print("\nPreprocessed finished cost time:%f" % (time.time() - start))
