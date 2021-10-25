import torch
# import cv2
import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


# train_path = r'..\dataset\font style\train'
# valid_path = r'..\dataset\font style\valid'

# data = sio.loadmat("./CV_1_801.mat")
# train_data = np.reshape(data["train"], [2400, 64, 64, 1])
# train_label = data["train_label"]
# valid_data = np.reshape(data["test"], [800, 64, 64, 1])
# valid_label = data["test_label"]


# print(len(train_data), len(valid_data))


class StockData(Dataset):
    """
    股票数据集，data_set
    包含输入的数据和分类结果label
    """

    def __init__(self, stock_data, font_label, transform=None):
        self.stock_data = stock_data
        self.transform = transform
        self.font_label = font_label

    def __len__(self):
        return len(self.font_data)

    def __getitem__(self, index):
        img = self.stock_data[index]
        label = np.argmax(self.font_label[index])
        if self.transform:
            img = self.transform(img)
        return img, label


# train_datum = FontData(train_data, train_label, my_trans)
# valid_datum = FontData(valid_data, valid_label, my_trans)


def input_data(data_dir: str) -> (DataLoader, DataLoader, DataLoader):
    """
    返回data_loader
    :param data_dir: 训练数据所在的文件路径
    :return: data_loader
    """
    train_path = data_dir + r'\train'
    valid_path = data_dir + r'\valid'
    test_path = data_dir + r'\test'

    my_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((96, 96)),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_datum = datasets.ImageFolder(train_path, transform=my_trans)
    valid_datum = datasets.ImageFolder(valid_path, transform=my_trans)
    test_datum = datasets.ImageFolder(test_path, transform=my_trans)

    train_loader = DataLoader(train_datum, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_datum, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datum, batch_size=64, shuffle=False)

    return train_loader, valid_loader, test_loader
