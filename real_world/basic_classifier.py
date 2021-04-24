import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from torch.utils.data import DataLoader
from dataset import LaserDataset

# Copy the tutorial CIFAR-10 classifier, but w/ binary classification
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1886544, 120) # 1886544 established based on image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) # 1 output for binary classification task

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # For binary classification only
        return x

def main():
    net = Net()

    # For binary classification, use BCELoss (needs replacing for regression or multi-class)
    criterion = nn.BCELoss()
    # For batch training
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    dataset = LaserDataset()
    train_size = int(len(dataset) * 0.8)
    train, test = torch.utils.data.random_split(dataset, [train_size,
                                                          len(dataset) - train_size])
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
    test_dataloader  = DataLoader(test,  batch_size=64, shuffle=True)

    for epoch in range(100):
        total_loss = 0.0
        for i, (inputs, label) in enumerate(train_dataloader):
            print(f'Batch: {i}\r', end='')

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'{epoch:<5} loss: {total_loss/len(dataset):6.2f}')
        torch.save(net.state_dict(), f'net_params/epoch_{epoch:04d}.pth')

if __name__ == '__main__':
    main()
