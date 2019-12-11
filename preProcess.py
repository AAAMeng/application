# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd

proxy_port = tuple(('7a', '31'))
# txt_file = "../dataset/raw_data/"+appName+".txt"
# csv_file = "../dataset/labeled_data/"+appName+".csv"
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
        df1 = pd.read_csv("../dataset/raw_data/" + k + ".txt", sep="\n", header=None, engine='python',
                          skiprows=lambda x: x % 4 != 2, dtype='str')  # read from csv data
        df1 = df1[0].str.split('  ', expand=True)
        df1 = df1[1].str.split('|', expand=True).drop([0], axis=1)
        if fillna:
            df1.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)  # fill 0 into packet
            df1 = df1.fillna('0')
        data[k] = df1
        print(k+' ... [Done]')
    return data


def packet_list(dict1, dict2):
    for k, v in dict1.items():
        if dict2.__contains__(k):
            dict1[k] = np.sort(np.unique(np.concatenate((dict1[k], dict2[k]), axis=0)))
    return dict1


def session_merge(data):
    """
    Function: merge the packets which belong to same session
    :param data: a Dict contain all DataFrame of each Application. s.t. data={'app1':app1_df, ...}
    :return: lists[Chrome] = {port1:[1,2,10,33,....], port2:[3,4,5,6,....],...}
    """
    lists = {}
    for aName, df in data.items():
        grouped_by_src = df.groupby([35, 36])  # return GroupBy Object
        grouped_by_dst = df.groupby([37, 38])
        list_by_src = grouped_by_src.indices
        list_by_dst = grouped_by_dst.indices
        lists[aName] = packet_list(list_by_src, list_by_dst)
        print(aName + ' ... [Done]')
    return lists


def rawdata_format(data, lists, sess_size=10, pck_str=16, pck_len=160):
    """
    Function: transform raw data into intended format(1 row = 1 session with 10 packets and 160bytes/packet)
    :param data: a Dict contain all DataFrame of each Application. s.t. data={'app1':app1_df, ...}
    :param lists: lists[Chrome] = {port1:[1,2,10,33,....], port2:[3,4,5,6,....],...}
    :param sess_size: number of packet selected from one session
    :param pck_str: packet start position
    :param pck_len: packet length(fixed)
    :return: sorted packets in session in hex
    """
    # slice 16th~175th, column index keep same
    for aName, df in data.items():
        df = df.iloc[:, pck_str:pck_str + pck_len]
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)  # fill 0 into packet
        df = df.fillna('0')
        sessions = pd.DataFrame()
        for port, l in lists.get(aName).items():
            if port == proxy_port:
                continue
            else:
                s = pd.concat([df.iloc[i] for i in l[0:sess_size]], axis=0, ignore_index=True)
                sessions = pd.concat([sessions, s], axis=1, ignore_index=True)
        data[aName] = label_data(aName, sessions.T, sess_size, pck_len)
        print(aName + ' ... [Done]')
    return data


def label_data(aname, df1, sess_size, pck_len):
    df1.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)  # fill 0 into session
    df1 = df1.fillna('0')
    df1[sess_size * pck_len] = app_label[aname]
    return df1


def write_into_csv(data):
    """
    Function: convert hex into dec and write the sessions bytes of each Application into csv.file
    :param data: a Dict contain all df of each Application data={'app1':app1_df, ...}
    """
    for fname, df1 in data.items():
        hex_list = df1.to_numpy()
        dec_list = [[int(hex_list[i][j], 16) for j in range(len(hex_list[i]))] for i in range(len(hex_list))]

        df1 = pd.DataFrame(dec_list)
        df1.to_csv("../dataset/labeled_data/" + fname + ".csv")
        print(fname + ' ... [Done]')


if __name__ == "__main__":
    pd.set_option('mode.chained_assignment', None)
    start = time.time()
    print("--------------------START--------------------")
    print("1. Read from txt:")
    pData = read_from_txt()
    print('---------------------------------------------')
    print("2. Merge session:")
    pList = session_merge(pData)
    print('---------------------------------------------')
    print("3. Format session:")
    sData = rawdata_format(pData, pList)
    print('---------------------------------------------')
    print("4. Write into csv:")
    write_into_csv(sData)
    print("\nPreprocessed finished cost time:%f" % (time.time() - start))
