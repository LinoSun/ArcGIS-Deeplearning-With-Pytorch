import copy
import math
import os

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
from PIL import Image

from datasets.dataset_utils import Classified_Tiles_Dataset
from models.Unet.unet_model import Unet

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def unet_transforms():
    '''返回一个数据增强的列表'''
    transforms = [
        torchvision.transforms.RandomHorizontalFlip(1),
        torchvision.transforms.RandomVerticalFlip(1)
    ]
    return transforms


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot(image, label):
    print(image.size, label.size)
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image[0])
    print(image.size)
    label = unloader(label[0])
    label = np.asarray(label)
    label_copy = label.copy()
    label_copy[label_copy == 1] = 255
    label = Image.fromarray(label_copy)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(label)


def unet_train(model, data_path, epochs=40, batch_size=1, lr=0.001, split=0.2, step_size=5,
               save_inter=1, disp_inter=1, checkpoint_dir=None, find_lr=False):
    # 模型保存地址
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(data_path, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # 加载训练集
    unet_dataset = Classified_Tiles_Dataset(data_path, transforms=unet_transforms())
    # 切分数据集
    val_size = int(split * len(unet_dataset))
    train_size = int(len(unet_dataset) - val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(unet_dataset, [train_size, val_size])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据集
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # 学习率调整策略
    scheduler = StepLR(optimizer, step_size=step_size)
    # 损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    # 查找学习率
    if find_lr:
        logs, losses = lr_find(model, train_data_loader, optimizer, criterion)
        plt.plot(logs[10:-5], losses[10:-5])
        return [logs,losses]

    # 主循环
    train_loss_total_epochs, val_loss_total_epochs, epoch_lr = [], [], []
    best_loss = 1e50
    best_mode = copy.deepcopy(model)
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss_per_epoch = 0
        for image, label in train_data_loader:
            image, label = image.to(device), label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            pred = model(image)
            # 因为损失函数，所以降维
            label = torch.squeeze(label, 0).long()
            # print(pred.size(),label.size())
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss.item()
        # 验证阶段
        model.eval()
        val_loss_per_epoch = 0
        with torch.no_grad():
            for image, label in val_data_loader:
                data, label = image.to(device), label.to(device)
                pred = model(data)
                # 损失函数
                label = torch.squeeze(label, 0).long()
                loss = criterion(pred, label)
                val_loss_per_epoch += loss.item()

        # 一个epoch结束
        train_loss_per_epoch = train_loss_per_epoch / train_size
        val_loss_per_epoch = val_loss_per_epoch / val_size
        train_loss_total_epochs.append(train_loss_per_epoch)
        val_loss_total_epochs.append(val_loss_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
        # 保存最优模型
        if val_loss_per_epoch < best_loss:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = val_loss_per_epoch
            best_mode = copy.deepcopy(model)
        scheduler.step()
        # 显示loss
        if epoch % disp_inter == 0:
            print('Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}'.format(epoch, train_loss_per_epoch,
                                                                                  val_loss_per_epoch))

    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='训练集loss')
        ax.plot(x, smooth(val_loss_total_epochs, 0.6), label='验证集loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title(f'训练曲线', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr, label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'学习率变化曲线', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    return best_mode, model


def model_pred(model):
    pass


def lr_find(model, trn_loader, optimizer, criterion, init_value=1e-8, final_value=10., beta=0.98):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in trn_loader:
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        labels = torch.squeeze(labels, 0).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # print(loss.item(),loss.data)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


if __name__ == '__main__':
    data_path = r'F:\deepLearning\data\building]Data\trainingData\building3000_Classfied_Tile'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(3, 2).to(device)
    # best_model, model = unet_train(model, data_path, epochs=2, find_lr=True)
    logs, losses = unet_train(model, data_path, epochs=2, find_lr=True)
    plt.plot(logs,losses)
    plt.show()
