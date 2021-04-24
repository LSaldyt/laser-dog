import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class LaserDataset(Dataset):

    def __init__(self, csv_file='labels.csv', root_dir='laser_images/', transform=None, kind='classifier'):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0].strip())
        print(image_name)
        try:
            image = io.imread(image_name)
            laser = self.labels.iloc[idx, 4:5]
            laser = np.array(laser)
            print(laser)
            reg = self.labels.iloc[idx, 2:4]
            reg = np.array(reg)
            print(reg)
            sample = dict(image=image, laser=laser, reg=reg)

            if self.transform:
                sample = self.transform(sample)
            return sample
        except:
            return None


if __name__ == '__main__':
    dataset = LaserDataset()
    for item in dataset:
        print(item)



