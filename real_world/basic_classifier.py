import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from dataloader import LaserDataset

# Copy the tutorial CIFAR-10 classifier, but w/ binary classification
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1886544, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def main():
    net = Net()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    dataset = LaserDataset()

    for epoch in range(100):
        total_loss = 0.0
        for i, data in enumerate(dataset, 0):
            print(f'Item: {i}\r', end='')
            if data is None: # If images are corrupt, skip training on them.
                continue
            inputs = data['image']
            label  = data['laser']

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
