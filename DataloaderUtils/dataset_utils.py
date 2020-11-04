import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image


def preprocess(pil_image):
    image_nd = np.array(pil_image)
    if len(image_nd.shape) == 2:
        image_nd = np.expand_dims(image_nd, axis=2)
    try:
        image_trans = image_nd.transpose(2, 0, 1)
    except:
        print(image_nd.shape)
    if image_trans.max() > 1: image_trans = image_trans / 255
    return image_trans


class Classified_Tiles_Dataset(Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.rootp = Path(root)
        map_txt = self.rootp / 'map.txt'
        if not os.path.exists(map_txt):
            raise Exception('没有map.txt，无法推断数据后缀')

        with open(map_txt) as f:
            right = f.readline().replace('\n','').strip()

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

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # 归一化处理
        # image = preprocess(image)
        image = np.array(image)
        mask = np.array(mask)
        return image, mask

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data_path = r'F:\deepLearning\data\building]Data\trainingData\building3000_Classfied_Tile'
    dataset = Classified_Tiles_Dataset(data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, pin_memory=False)
    for image, label in data_loader:
        print(image.shape)
