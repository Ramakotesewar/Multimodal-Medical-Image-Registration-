
import numpy as np
import torch
from nibabel import spaces
from skimage.transform import AffineTransform, warp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Evaluate_Error import evaluat_error


class ImageRegistrationEnv():
    def __init__(self, image_pairs):
        super(ImageRegistrationEnv, self).__init__()
        self.image_pairs = image_pairs
        self.current_index = 0
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # Affine transform parameters
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 256, 256), dtype=np.float32)  # Two images

    def reset(self):
        self.current_index = np.random.randint(len(self.image_pairs))
        img1, img2 = self.image_pairs[self.current_index]
        self.img1 = self.transform(img1).numpy()
        self.img2 = self.transform(img2).numpy()
        return np.stack([self.img1, self.img2], axis=0)

    def step(self, action):
        transformation = AffineTransform(translation=action[:2], rotation=action[2], shear=action[3], scale=action[4:6])
        transformed_img1 = warp(self.img1, transformation.inverse, output_shape=self.img2.shape)
        reward = -np.mean((transformed_img1 - self.img2) ** 2)
        done = True
        return np.stack([transformed_img1, self.img2], axis=0), reward, done, {}

    def PPO(self, mode='human'):
        pass

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



def Model_DRL(image1, image2, Target):
    images = []
    for a in range(1000):  # Because of large data
        image_pairs = [image1[a], image2[a]]
        dataset = ImageRegistrationEnv(image_pairs)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = (dataset)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        image = train_image_registration(model, train_loader, criterion, optimizer, num_epochs)
        predict = model.predict(image)
        images.append(predict)
    Eval = evaluat_error(images, Target)
    return Eval

