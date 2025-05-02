import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from Evaluate_Error import evaluat_error
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ImageRegistrationDataset(Dataset):
    def __init__(self, image_pairs, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image_pair = self.image_pairs[idx]
        img1, img2, transformation = image_pair

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.Tensor(transformation)


class AdaptiveTransResUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(1, 2, 4)):
        super(AdaptiveTransResUNetBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate) for rate in dilation_rates
        ])
        self.relu = nn.ReLU()
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.res_conv(x)
        for conv in self.convs:
            x = self.relu(conv(x))
        x += residual
        return x


class ATRUNet(nn.Module):
    def __init__(self, in_channels, out_channels, sol):
        super(ATRUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.transresunet_blocks = nn.Sequential(
            AdaptiveTransResUNetBlock(sol[0], 64),
            AdaptiveTransResUNetBlock(64, 64)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transresunet_blocks(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x


def train_image_registration(model, train_loader, criterion, optimizer, num_epochs, Steps_per_Epoch):
    model.train()
    for epoch in range(num_epochs):
        for img1, img2, target in train_loader:
            optimizer.zero_grad()
            outputs = model(img1)
            loss = criterion(outputs, target, Steps_per_Epoch)
            loss.backward()
            optimizer.step()
        return model

def Model_ATRUNet(image1, image2, Target, sol=None):
    if sol is None:
        sol = [5, 5, 100]
    images = []
    for a in range(1000):  # Adjust this according to your data size
        image_pairs = [(image1[a], image2[a])]
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ImageRegistrationDataset(image_pairs, transform=transform)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = ATRUNet(in_channels=1, out_channels=1, sol=sol)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = sol[1]
        Steps_per_Epoch = sol[2]
        train_image_registration(model, train_loader, criterion, optimizer, num_epochs, Steps_per_Epoch)
        model.eval()

        with torch.no_grad():
            for img1, img2, target in train_loader:
                predict = model(img1)
                images.append(predict)
    Eval = evaluat_error(images, Target)
    return Eval, predict
