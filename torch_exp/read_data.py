import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def load_data_from_torchvision():
    test_data = torchvision.datasets.CIFAR10(
        "./data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    img, target = test_data[0]
    print(img.shape)
    print(target)

    writer = SummaryWriter("./logs")
    for epoch in range(2):
        step = 0
        for data in test_loader:
            imgs, targets = data
            writer.add_images("Epoch: {}".format(epoch), imgs, step)
            step = step + 1

    writer.close()


class SsdData(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)

        with open(label_item_path, 'r') as f:
            label = f.readline()

        # img = np.array(img)
        img = self.transform(img)
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


def load_data_from_ssd():
    writer = SummaryWriter("logs")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    root_dir = "./data/pictures/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = SsdData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = SsdData(root_dir, image_bees, label_bees, transform)
    train_dataset = ants_dataset + bees_dataset

    print(train_dataset[0])

    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)
    for i, j in enumerate(dataloader):
        # imgs, labels = j
        print(type(j))
        print(i, j['img'].shape)
        # writer.add_image("train_data_b2", make_grid(j['img']), i)

    writer.close()


if __name__ == '__main__':
    load_data_from_torchvision()
