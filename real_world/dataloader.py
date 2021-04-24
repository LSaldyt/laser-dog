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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0].strip())
        try:
            image = torch.from_numpy(io.imread(image_name))
            image = image.unsqueeze(0)
            image = torch.movedim(image, 3, 1).float()
            laser = self.labels.iloc[idx, 4:5]
            laser = torch.from_numpy(np.array(laser, dtype=np.int16))
            laser = laser.unsqueeze(0).float()
            reg = self.labels.iloc[idx, 2:4]
            reg = torch.from_numpy(np.array(reg, dtype=np.float32))
            reg = reg.unsqueeze(0).float()
            sample = dict(image=image, laser=laser, reg=reg)

            if self.transform:
                sample = self.transform(sample)
            return sample
        except ValueError:
            return None


if __name__ == '__main__':
    dataset = LaserDataset()
    for item in dataset:
        print(item)



