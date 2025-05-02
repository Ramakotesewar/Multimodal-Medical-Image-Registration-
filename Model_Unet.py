import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Evaluate_Error import evaluat_error


class ImageRegistrationDataset(Dataset):
    def __init__(self, image_pairs, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1, img2, transformation = self.image_pairs[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(transformation, dtype=torch.float32)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = self.up_conv_block(512, 256)
        self.dec3 = self.up_conv_block(256, 128)
        self.dec2 = self.up_conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        dec4 = self.dec4(self.pool(enc4))
        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True))
        return dec1


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, target in dataloader:
            optimizer.zero_grad()
            outputs = model(img1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img1.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss


def Model_UNET(image1, image2, Target):
    images = []
    for a in range(1000):  # Adjust this according to your data size
        image_pairs = [(image1[a], image2[a])]
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ImageRegistrationDataset(image_pairs, transform=transform)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = UNet()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 25
        train_model(model, train_loader, criterion, optimizer, num_epochs)
        predict = model.predict(image_pairs)
        images.append(predict)
    Eval = evaluat_error(images, Target)
    return Eval
