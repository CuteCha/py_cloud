import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


def debug_net():
    net = Net()
    print(net)
    x = torch.ones((64, 3, 32, 32))
    y = net(x)
    print(y.shape)


def run():
    debug_net()


def make_dataset():
    train_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        "./data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=64
    )

    train_data_size = len(train_data)
    test_data_size = len(test_data)

    return train_dataloader, test_dataloader, train_data_size, test_data_size


def train():
    pass


def save_model(net, dump_dir, epoch, max_dump_num=5):
    torch.save(net, f"{dump_dir}/net_{epoch}.pth")
    if epoch >= max_dump_num and os.path.exists(f"{dump_dir}/net_{epoch - max_dump_num}.pth"):
        os.remove(f"{dump_dir}/net_{epoch - max_dump_num}.pth")


def load_model():
    net = torch.load("./logs/checkpoint/net_9.pth")
    print(net)
    return net


def read_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])

    image = transform(image)
    image = torch.reshape(image, (1, 3, 32, 32))

    return image


def inference():
    net = load_model()
    image = read_image("./data/backup/airplane.png")
    net.eval()
    with torch.no_grad():
        pred = net(image)

    print(pred)
    print(pred.argmax(1))


def main():
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_dataloader, test_dataloader, train_data_size, test_data_size = make_dataset()
    net = Net()
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    total_train_step = 0
    total_test_step = 0
    epoch = 10
    writer = SummaryWriter("./logs/tensorboard")
    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i + 1))

        net.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        net.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = net(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step = total_test_step + 1

        save_model(net, "./logs/checkpoint/", i)
        print("模型已保存")

    writer.close()


if __name__ == '__main__':
    main()
    # inference()
