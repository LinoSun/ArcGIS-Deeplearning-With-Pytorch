import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image


class Classified_Tiles_Dataset(Dataset):

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

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        # 变换中随机挑选一个变换
        if self.transforms is not None:
            transform_index = random.randint(0, len(self.transforms)-1)
            print(transform_index)
            image = self.transforms[transform_index](image)
            mask = self.transforms[transform_index](mask)
        # totensor中就已经做了归一化处理以及纬度变换
        totensor = torchvision.transforms.ToTensor()
        image = totensor(image)
        mask = totensor(mask)
        return image, mask

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data_path = r'F:\deepLearning\data\building]Data\trainingData\building3000_Classfied_Tile'
    dataset = Classified_Tiles_Dataset(data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, pin_memory=False)
    for image, label in data_loader:
        print(image.shape)
