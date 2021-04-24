import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class LaserDataset(Dataset):

    def __init__(self, csv_file='labels.csv', root_dir='laser_images/', transform=None, kind='classifier'):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Filter corrupt images ahead of time
        print('Filtering corrupt images: ')
        bad = []
        for idx in range(len(self.labels)):
            print(f'Filtering: {idx}\r', end='')
            try:
                self.__getitem__(idx)
            except ValueError:
                bad.append(idx)
        for b in bad:
            self.labels.drop(b, inplace=True)
        print(f'Filtering complete. Dropped {len(bad)} images')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0].strip())
        image = torch.from_numpy(io.imread(image_name))
        image = torch.movedim(image, 2, 0).float()
        laser = self.labels.iloc[idx, 4:5]
        laser = torch.from_numpy(np.array(laser, dtype=np.int16))
        laser = laser.float()
        reg = self.labels.iloc[idx, 2:4]
        reg = torch.from_numpy(np.array(reg, dtype=np.float32))
        reg = reg.float()
        sample = dict(image=image, laser=laser, reg=reg)

        if self.transform:
            sample = self.transform(sample)
        return image, laser

if __name__ == '__main__':
    dataset = LaserDataset()
    for item in dataset:
        print(item)



