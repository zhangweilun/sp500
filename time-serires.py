import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_dataset(data: pd.DataFrame, days_for_train=5, feature_nums=3, regression=True) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        feature_nums:输入的特征数量 默认最后一列为y,并且列名为y
        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    columes = data.columns.values
    y_name = data.columns.values[len(columes) - 1]
    data.rename(columns={str(y_name): 'y'})
    dataset_x, dataset_y = [], []
    if regression:
        for i in range(len(data) - days_for_train):
            x = data.iloc[i:i + days_for_train, :feature_nums]
            dataset_x.append(x)
        dataset_y = data.iloc[days_for_train:, data.shape[1] - 1:data.shape[1]]
        return np.array(dataset_x), np.array(dataset_y)
    else:
        for i in range(len(data) - days_for_train):
            x = data.iloc[i:i + days_for_train, :feature_nums]
            dataset_x.append(x)
        y = data.iloc[days_for_train - 1:, data.shape[1] - 1:data.shape[1]]
        # 相邻两行相减
        y["tump"] = y["y"].shift(1)
        gap_y = y["y"] - y["tump"]
        for i, v in gap_y.items():
            print(i, v)
            if pd.isna(v):
                continue
            if v >= 0:
                dataset_y.append(1)
            else:
                dataset_y.append(0)
        return np.array(dataset_x), np.array(dataset_y)


if __name__ == '__main__':
    data = pd.read_csv("./data/SP500.csv")

    # train_x, train_y = create_dataset(data, regression=False)
    # # displot 分布图 y轴为数量
    # sns.displot(train_y)
    # lineplot 折线图
    sns.lineplot(x=data['Date'], y=data['Close'])
    plt.show()

