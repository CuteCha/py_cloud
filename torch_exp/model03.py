import torch
import numpy as np
from torch.utils.data import Dataset

from torch import nn
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader


def read_data():
    dataFrame = pd.read_csv("./data/age_gender.gz", compression='gzip')

    age_bins = [0, 10, 15, 20, 25, 30, 40, 50, 60, 120]
    age_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    dataFrame['bins'] = pd.cut(dataFrame.age, bins=age_bins, labels=age_labels)

    train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2)

    class_nums = {'age_num': len(dataFrame['bins'].unique()), 'eth_num': len(dataFrame['ethnicity'].unique()),
                  'gen_num': len(dataFrame['gender'].unique())}

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    train_set = UTKDataset(train_dataFrame, transform=train_transform)
    test_set = UTKDataset(test_dataFrame, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    for X, y in train_loader:
        print(f'Shape of training X: {X.shape}')
        print(f'Shape of y: {y.shape}')
        break

    return train_loader, test_loader, class_nums


def train(trainloader, model, opt, num_epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    age_loss = nn.CrossEntropyLoss()
    gen_loss = nn.CrossEntropyLoss()
    eth_loss = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        gen_correct, eth_correct, age_correct, total = 0, 0, 0, 0
        for _, (X, y) in loop:
            age, gen, eth = y[:, 0].to(device), y[:, 1].to(device), y[:, 2].to(device)
            X = X.to(device)

            pred = model(X)  # Forward pass
            loss = age_loss(pred[0], age) + gen_loss(pred[1], gen) + eth_loss(pred[2], eth)  # Loss calculation

            # Backpropagation
            opt.zero_grad()  # Zero the gradient
            loss.backward()  # Calculate updates

            # Gradient Descent
            opt.step()  # Apply updates

            age_correct += (pred[1].argmax(1) == age).type(torch.float).sum().item()
            gen_correct += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
            eth_correct += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

            total += len(y)

            # Update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epoch}]")
            loop.set_postfix(loss=loss.item())

    gen_acc, eth_acc, age_acc = gen_correct / total, eth_correct / total, age_correct / total
    print(f'Epoch: {epoch + 1}/{num_epoch}, Age Accuracy: {age_acc * 100}, '
          f'Gender Accuracy: {gen_acc * 100}, Ethnicity Accuracy : {eth_acc * 100}\n')


def evaluate(testloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    size = len(testloader.dataset)
    model.eval()

    age_acc, gen_acc, eth_acc = 0, 0, 0  # capital L on age to not get confused with loss function

    with torch.no_grad():
        for X, y in testloader:
            age, gen, eth = y[:, 0].to(device), y[:, 1].to(device), y[:, 2].to(device)
            X = X.to(device)
            pred = model(X)

            age_acc += (pred[0].argmax(1) == age).type(torch.float).sum().item()
            gen_acc += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
            eth_acc += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

    age_acc /= size
    gen_acc /= size
    eth_acc /= size

    print(f"Age Accuracy: {age_acc * 100}%, Gender Accuracy: {gen_acc * 100}, Ethnicity Accuracy : {eth_acc * 100}\n")


class UTKDataset(Dataset):
    '''
        Inputs:
            dataFrame : Pandas dataFrame
            transform : The transform to apply to the dataset
    '''

    def __init__(self, dataFrame, transform=None):
        self.transform = transform
        data_holder = dataFrame.pixels.apply(lambda x: np.array(x.split(" "), dtype=float))
        arr = np.stack(data_holder)
        arr = arr / 255.0
        arr = arr.astype('float32')
        arr = arr.reshape(arr.shape[0], 48, 48, 1)
        self.data = arr

        self.age_label = np.array(dataFrame.bins[:])
        self.gender_label = np.array(dataFrame.gender[:])
        self.eth_label = np.array(dataFrame.ethnicity[:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        data = self.transform(data)
        labels = torch.tensor((self.age_label[index], self.gender_label[index], self.eth_label[index]))

        return data, labels


class highLevelNN(nn.Module):
    def __init__(self):
        super(highLevelNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.CNN(x)

        return out


class lowLevelNN(nn.Module):
    def __init__(self, num_out):
        super(lowLevelNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=num_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2, padding=1))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class TridentNN(nn.Module):
    def __init__(self, num_age, num_gen, num_eth):
        super(TridentNN, self).__init__()
        self.CNN = highLevelNN()
        self.ageNN = lowLevelNN(num_out=num_age)
        self.genNN = lowLevelNN(num_out=num_gen)
        self.ethNN = lowLevelNN(num_out=num_eth)

    def forward(self, x):
        x = self.CNN(x)
        age = self.ageNN(x)
        gen = self.genNN(x)
        eth = self.ethNN(x)

        return age, gen, eth


def model_debug01():
    print('Testing out Multi-Label NN')
    mlNN = TridentNN(10, 10, 10)
    input = torch.randn(64, 1, 48, 48)
    y = mlNN(input)
    print(y[0].shape)


def main():
    train_loader, test_loader, class_nums = read_data()

    tridentNN = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])
    opt = torch.optim.Adam(tridentNN.parameters(), lr=0.001)

    train(train_loader, tridentNN, opt, 2)
    print('Finished training, running the valid script...')
    evaluate(test_loader, tridentNN)
    print("Finished valid")

    torch.save({'epoch': 2, 'state_dict': tridentNN.state_dict(), 'optimizer': opt.state_dict()},
               './data/tridentNN_epoch2.pth.tar')
    print("model checkpoint success")


def predict():
    train_loader, test_loader, class_nums = read_data()

    tridentNN = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])
    # Load and test the trained model
    checkpoint = torch.load('./data/tridentNN_epoch2.pth.tar')
    tridentNN.load_state_dict(checkpoint['state_dict'])
    print("load checkpoint success, start evaluation")
    evaluate(test_loader, tridentNN)


if __name__ == '__main__':
    # main()
    predict()
