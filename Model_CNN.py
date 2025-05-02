import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Evaluate_Error import evaluat_error


# Define a simple CNN model for image registration
class ImageRegistrationCNN(nn.Module):

    def __init__(self):
        super(ImageRegistrationCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        image_size = 128
        self.fc = nn.Sequential(
            nn.Linear(64 * (image_size // 4) * (image_size // 4), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)  # Output 6 parameters for affine transformation
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Custom dataset for image pairs and corresponding transformations
class ImageRegistrationDataset():
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


# Training function
def train_image_registration(model, train_loader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            img1, img2, target = batch

            optimizer.zero_grad()
            outputs = model(img1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        return model

def Model_CNN(image1, image2, Target):
    images = []
    for a in range(1000):  # Because of large data
        image_pairs = [image1[a], image2[a]]  # List of tuples (image1, image2, transformation)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ImageRegistrationDataset(image_pairs, transform=transform)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = ImageRegistrationCNN()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        image = train_image_registration(model, train_loader, criterion, optimizer, num_epochs)
        predict = model.predict(image)
        images.append(predict)
    Eval = evaluat_error(images, Target)
    return Eval
