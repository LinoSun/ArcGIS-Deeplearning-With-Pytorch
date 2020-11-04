import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from PIL import Image

from DataloaderUtils.dataset_utils import Classified_Tiles_Dataset
from models.Unet.unet_model import Unet


def unet_transforms():
    hp = random.choice([0, 1])
    vp = random.choice([0, 1])
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(hp),
         torchvision.transforms.RandomVerticalFlip(vp),
         torchvision.transforms.ToTensor()]
    )
    print(hp,vp)
    return transforms


def plot(image, label):
    print(image.size,label.size)
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image[0])
    print(image.size)
    label = unloader(label[0])
    label = np.asarray(label)
    label_copy = label.copy()
    label_copy[label_copy==1] = 255
    label = Image.fromarray(label_copy)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(label)


def unet_train(model, data_path, batch_size=4, lr=0.001, split=0.2):
    # 加载训练集
    unet_dataset = Classified_Tiles_Dataset(data_path, transforms=unet_transforms())
    # 切分数据集
    val_size = int(split * len(unet_dataset))
    train_size = int(len(unet_dataset) - val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(unet_dataset, [train_size, val_size])
    data_loader = DataLoader(train_dataset,batch_size=batch_size)
    for image,label in data_loader:
        plot(image, label)


if __name__ == '__main__':
    data_path = r'F:\deepLearning\data\building]Data\trainingData\building3000_Classfied_Tile'
    unet_train(Unet, data_path)
