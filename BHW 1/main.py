import torch
import numpy as np
import torchvision
import tqdm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import shutil
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from IPython.display import clear_output
from torchvision.models import resnet18, resnet152


PATH = 'bhw1/'
TEST_SIZE = 0.5
SPLIT_RANDOM_SEED = 42

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

device = torch.device('mps')


class ImageDataset(Dataset):
    TEST_SIZE = 0.5
    SPLIT_RANDOM_SEED = 42

    def __init__(self, root, data, test_size = 0.5, train=True, load_to_ram=True, transform=None, trainval=True):
        super().__init__()
        self.TEST_SIZE = test_size
        self.root = root
        self.train = train
        self.trainval = trainval
        self.load_to_ram = load_to_ram
        self.transform = transform
        self.to_tensor = T.ToTensor()
        self.all_files = []
        self.all_labels = []
        self.images = []

        if trainval:
            self.classes = sorted(data['Category'].unique())
            for i, class_name in tqdm(enumerate(self.classes), total=len(self.classes)):
                files = data[data['Category'] == class_name]['Id']
                train_files, test_files = train_test_split(files.to_list(), random_state=self.SPLIT_RANDOM_SEED + i, test_size=self.TEST_SIZE)
                if self.train:
                    self.all_files += train_files
                    self.all_labels += [class_name] * len(train_files)
                    if self.load_to_ram:
                        self.images += self._load_images(train_files)

                else:
                    self.all_files += test_files
                    self.all_labels += [class_name] * len(test_files)
                    if self.load_to_ram:
                        self.images += self._load_images(test_files)
        else:
            for file_name in tqdm(data['Id'].to_list(), total=data.shape[0]):
                self.all_labels += [file_name]
                self.all_files += [file_name]
                image = Image.open(os.path.join(self.root, 'test/', file_name)).convert('RGB')
                self.images += [image]

    def _load_images(self, image_files):
        images = []
        for filename in image_files:
            add = 'trainval/' if self.trainval else 'test/'
            image = Image.open(os.path.join(self.root, add, filename)).convert('RGB')
            #if self.transform is not None:
            #    image = self.transform(image)
            images += [image]

        return images

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, item):
        label = self.all_labels[item]
        if self.load_to_ram:
            image = self.images[item]
        else:
            filename = self.all_files[item]
            add = 'trainval/' if self.trainval else 'test/'
            image = Image.open(os.path.join(self.root, add, filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model, optimizer, criterion, train_loader, tqdm_desc):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()
    for images, labels in tqdm(train_loader, total=len(train_loader), desc=tqdm_desc, leave=True):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size

        optimizer.zero_grad()
        logits = model(images)  # logits: batch_size x num_classes
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        train_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


@torch.no_grad()
def validation_epoch(model, criterion, test_loader, tqdm_desc):
    test_loss, test_accuracy = 0.0, 0.0
    model.eval()
    for images, labels in tqdm(test_loader, total=len(test_loader), desc=tqdm_desc):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size
        logits = model(images)  # logits: batch_size x num_classes
        loss = criterion(logits, labels)

        test_loss += loss.item() * images.shape[0]
        test_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy


def train(model, optimizer, scheduler, criterion, train_loader, test_loader, num_epochs):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        if test_loader is not None:
            test_loss, test_accuracy = validation_epoch(
                model, criterion, test_loader,
                tqdm_desc=f'Validating {epoch}/{num_epochs}'
            )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        test_losses += [test_loss]
        test_accuracies += [test_accuracy]
        plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)
        print(test_accuracies[-1])
#         torch.save(model.state_dict(), "weights.pt")

    return train_losses, test_losses, train_accuracies, test_accuracies


class ResBlock(nn.Module):
    def __init__(self, in_count, out_count):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_count, out_count, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_count),
            nn.ReLU(),
            nn.Conv2d(out_count, out_count, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_count),
        )
        torch.nn.init.kaiming_normal_(self.encoder[0].weight, mode="fan_out",
                                      nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.encoder[3].weight, mode="fan_out",
                                      nonlinearity="relu")
        torch.nn.init.constant_(self.encoder[1].weight, 1)
        torch.nn.init.constant_(self.encoder[1].bias, 0)
        torch.nn.init.constant_(self.encoder[4].weight, 1)
        torch.nn.init.constant_(self.encoder[4].bias, 0)

        self.added_lay = nn.Conv2d(in_channels=in_count,
                                   out_channels=out_count, kernel_size=1)
        torch.nn.init.kaiming_normal_(self.added_lay.weight, mode="fan_out",
                                      nonlinearity="relu")

        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        out = self.encoder(x)
        x = self.added_lay(x)
        x = self.relu(x + out)
        return self.drop(x)


class FourteenVrNN(nn.Module):
    def __init__(self, num_blocks=2, n_classes=200):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # (b, 64, 20, 20)
        )
        torch.nn.init.kaiming_normal_(self.encoder[0].weight, mode="fan_out",
                                      nonlinearity="relu")
        torch.nn.init.constant_(self.encoder[1].weight, 1)
        torch.nn.init.constant_(self.encoder[1].bias, 0)

        arr = []
        for i in range(num_blocks):
            arr += [ResBlock(64, 64)]
        self.resblock1 = nn.Sequential(*arr)

        arr2 = []
        arr2 += [ResBlock(64, 128)]
        for i in range(num_blocks - 1):
            arr2 += [ResBlock(128, 128)]
        self.resblock2 = nn.Sequential(*arr2)

        arr3 = []
        arr3 += [ResBlock(128, 256)]
        for i in range(num_blocks - 1):
            arr3 += [ResBlock(256, 256)]
        self.resblock3 = nn.Sequential(*arr3)

        arr4 = []
        arr4 += [ResBlock(256, 512)]
        for i in range(num_blocks - 1):
            arr4 += [ResBlock(512, 512)]
        self.resblock4 = nn.Sequential(*arr4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        #         x = self.resblock3(x)
        #         x = self.resblock4(x)
        x = self.avgpool(x)
        x = x.reshape((-1, 128))
        return self.linear(x)


def main():
    data = pd.read_csv(PATH + 'labels.csv')

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = T.Compose([
        T.RandomVerticalFlip(),
        T.ElasticTransform(),
        T.RandomGrayscale(),
        T.RandomInvert(),
        T.RandomEqualize(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_train_dataset = ImageDataset(root=PATH, data=data, test_size=0.5,
                                      train=True, load_to_ram=True,
                                      transform=test_transform)
    aug_train_dataset = ImageDataset(root=PATH, data=data, test_size=0.5,
                                     train=True, load_to_ram=True,
                                     transform=train_transform)

    train_dataset = base_train_dataset + aug_train_dataset
    test_dataset = ImageDataset(root=PATH, data=data, train=False, load_to_ram=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

    num_epochs = 30
    torch.autograd.set_detect_anomaly(True)
    model = FourteenVrNN()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    train_losses, test_losses, train_accuracies, test_accuracies = train(
        model, optimizer, scheduler, criterion, train_loader, test_loader,
        num_epochs
    )


if __name__ == '__main__':
    main()
