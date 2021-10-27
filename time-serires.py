import json

import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import dataset
import util.constant as constant


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

    gpu = torch.device('cuda')
    epochs = constant.EPOCHS
    learning_rate = constant.LEARNING_RATE
    weight_decay = constant.WEIGHT_DECAY
    epsilon = constant.EPSILON
    max_acc = constant.MAX_ACC
    model = Net()
    train_datum, valid_datum, test_datum = dataset.get_dataset(r"F:\project\sp500\data")
    train_loader, valid_loader, test_loader = dataset.get_data_loader(r"F:\project\sp500\data")
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.cuda()
    print("模型现在位于:{}".format(model.device))
    for epoch in range(epochs):
        print('*' * 30, 'epoch {}'.format(epoch + 1), '*' * 30)
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            print("imgaes现在位于:{},labels现在位于:{}".format(images.device, labels.device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs + epsilon, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            num_correct = (preds == labels).sum()
            running_acc += num_correct.item()
            loss.backward()
            optimizer.step()
        print('Finish {} epoch\nLoss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss * 100 / len(train_datum),
                                                                  running_acc * 100 / len(train_datum)))
        train_acc.append(running_acc * 100 / len(train_datum))
        train_loss.append(running_loss * 100 / len(train_datum))

        model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        for data in valid_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss = eval_loss + loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            num_correct = (preds == labels).sum()
            eval_acc = eval_acc + num_correct.item()
        print('Valid Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss * 100 / len(valid_datum),
                                                       eval_acc * 100 / len(valid_datum)))
        valid_acc.append(eval_acc * 100 / len(valid_datum))
        valid_loss.append(eval_loss * 100 / len(valid_datum))
        if (eval_acc * 100 / len(valid_datum)) > max_acc:
            max_acc = eval_acc * 100 / len(valid_datum)
            torch.save(model.state_dict(), '.\\trained_model\\best_model_basic_CBAM_31_mat.pth')
    print(max_acc)
    with open('.\\loss_acc\\train_basic_CBAM_31_mat', 'a+') as f1:
        json.dump(train_loss, f1)
        json.dump(train_acc, f1)
    with open('.\\loss_acc\\valid_basic_CBAM_31_mat', 'a+') as f2:
        json.dump(valid_loss, f2)
        json.dump(valid_acc, f2)
    torch.save(model.state_dict(), '.\\trained_model\\final_model_basic_CBAM_31_mat.pth')
