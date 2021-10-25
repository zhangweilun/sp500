import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import dataset
import util.constant


def create_dataset(data: pd.DataFrame, windows_size=5, feature_nums=3, regression=True) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        feature_nums:输入的特征数量 默认最后一列为y,并且列名为y
        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    data.drop('Date', axis=1, inplace=True)
    columes = data.columns.values
    y_name = data.columns.values[len(columes) - 1]
    data.rename(columns={str(y_name): 'y'})
    dataset_x, dataset_y = [], []
    if regression:
        for i in range(len(data) - windows_size):
            x = data.iloc[i:i + windows_size, :feature_nums]
            dataset_x.append(x)
        dataset_y = data.iloc[windows_size:, data.shape[1] - 1:data.shape[1]]
        return np.array(dataset_x), np.array(dataset_y)
    else:
        for i in range(len(data) - windows_size):
            x = data.iloc[i:i + windows_size, :feature_nums]
            dataset_x.append(x)
        y = data.iloc[windows_size - 1:, data.shape[1] - 1:data.shape[1]]
        # 相邻两行相减
        y["tump"] = y["y"].shift(1)
        gap_y = y["y"] - y["tump"]
        for i, v in gap_y.items():
            # print(i, v)
            if pd.isna(v):
                continue
            if v >= 0:
                dataset_y.append(1)
            else:
                dataset_y.append(0)
        return np.array(dataset_x), np.array(dataset_y)


class SpDataset(Dataset):
    """
      TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
      实现将一组Tensor数据对封装成Tensor数据集
      能够通过index得到数据集的数据，能够通过len，得到数据集大小
      """

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # [batch, time_step, input_dim]
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=3, num_layers=5, batch_first=True)
        self.linear = torch.nn.Linear(3, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x_input):
        # [batch time feature] [:, -1, :] 去最后一天作为输出
        h0 = torch.randn(5, 10, 3)
        c0 = torch.randn(5, 10, 3)
        x = self.lstm(x_input, (h0, c0))[0]
        x = x[:, -1, :]
        x = self.linear(x)
        x = self.relu(x)
        y = F.softmax(x, dim=1)
        return y


if __name__ == '__main__':
    # 用某日前8天窗口数据作为输入预测该日数据
    WINDOW_SIZE = util.constant.WINDOW_SIZE
    epochs = util.constant.EPOCHS
    feature_nums = util.constant.FEATURE_NUMS

    train_loader, valid_loader, test_loader = dataset.input_data(r"F:\project\sp500\data")

    data = pd.read_csv("./data/SP500.csv")
    train_x, train_y = create_dataset(data, windows_size=WINDOW_SIZE, feature_nums=feature_nums, regression=False)
    # # displot 分布图 y轴为数量
    # sns.displot(train_y)
    # lineplot 折线图
    # sns.lineplot(x=data['Date'], y=data['Close'])
    # plt.show()
    # 生成数据
    data_tensor = torch.from_numpy(train_x.astype(np.float32))
    # data_tensor = torch.from_numpy(train_x)
    target_tensor = torch.from_numpy(train_y.astype(np.float32))
    # 将数据封装成Dataset
    tensor_dataset = SpDataset(data_tensor, target_tensor)

    # 数据较小，可以将全部训练数据放入到一个batch中，提升性能
    train_loader = DataLoader(tensor_dataset, batch_size=10)

    model = Net()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_val_all = []
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        model.train()
        corrects = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):
            # input:[batch, time_step, input_dim]
            out = model(x)
            loss = loss_function(out, y)
            print('loss:', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
