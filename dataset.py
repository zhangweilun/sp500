from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder


def get_dataset(data_dir: str) -> (ImageFolder, ImageFolder, ImageFolder):
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

    train_datum = ImageFolder(train_path, transform=my_trans)
    valid_datum = ImageFolder(valid_path, transform=my_trans)
    test_datum = ImageFolder(test_path, transform=my_trans)
    return train_datum, valid_datum, test_datum


def get_data_loader(data_dir: str) -> (DataLoader, DataLoader, DataLoader):
    """
    返回data_loader
    :param data_dir: 训练数据所在的文件路径
    :return: data_loader
    """
    train_datum, valid_datum, test_datum = get_dataset(data_dir)

    train_loader = DataLoader(train_datum, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_datum, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datum, batch_size=64, shuffle=False)

    return train_loader, valid_loader, test_loader
