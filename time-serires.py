import numpy as np
import pandas as pd


def create_dataset(data, days_for_train=5, feature_nums=3, regression=True) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        feature_nums:输入的特征数量 默认最后一列为y
        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        x = data.iloc[i:i+days_for_train, :feature_nums]
        dataset_x.append(x)
    dataset_y = data.iloc[days_for_train:, data.shape[1]-1:data.shape[1]]
    # for i in range(len(data) - days_for_train):
    #     _x = data[i:(i + days_for_train)]
    #     dataset_x.append(_x)
    #     dataset_y.append(data[i + days_for_train])
    return np.array(dataset_x), np.array(dataset_y)


if __name__ == '__main__':
    data = pd.read_csv("./data/SP500.csv")
    create_dataset(data)
