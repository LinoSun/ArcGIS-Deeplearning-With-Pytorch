import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import gdal

from datasets.transforms import train_transform


class DXMDataset(Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.rootp = Path(root)
        map_txt = self.rootp / 'map.txt'
        if not os.path.exists(map_txt):
            raise Exception('没有map.txt，无法推断数据后缀')

        with open(map_txt) as f:
            right = f.readline().replace('\n', '').strip()

        if not right:
            raise Exception('map.txt中未读取到任何信息，请检查数据')
        image_right = right.split('  ')[0].split('\\')[1].split('.')[1]
        label_right = right.split('  ')[1].split('\\')[1].split('.')[1]

        self.images = [image for image in os.listdir(self.rootp / 'images') if image.endswith(image_right)]
        self.masks = [image.replace('.' + image_right, '.' + label_right) for image in self.images]

    def __getitem__(self, idx):
        # load image and masks
        image_path = self.rootp / 'images' / f'{self.images[idx]}'
        mask_path = self.rootp / 'labels' / f'{self.masks[idx]}'

        image = gdal.Open(str(image_path)).ReadAsArray().astype(np.uint8)
        image = image.transpose([1, 2, 0])
        mask = gdal.Open(str(mask_path)).ReadAsArray().astype(np.uint8)

        # 变换中随机挑选一个变换
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return {
            'image': image,
            'label': mask.long()
        }

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data_path = r'F:\deepLearning\data\dongxiaomai\xiaomai_samplearea_tra30_4band'
    dataset = DXMDataset(data_path, transforms=train_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, pin_memory=False)
    for image, label in data_loader:
        print(image.shape, label.shape)
