import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

train_list_file = "train_list.txt"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class FacadesDataset(data.Dataset):
    def __init__(self, list_file, transform=None):
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    ##分割数据集中的语义图片与真实图片
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        img_color_semantic = cv2.cvtColor(img_color_semantic, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic

train_dataset = FacadesDataset(train_list_file, transform=transform)

BATCHSIZE = 32
train_dataloader = data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)

annos_batch, imgs_batch = next(iter(train_dataloader))

def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    if image.shape[0] != 3:
        raise ValueError(f"Expected tensor with 3 channels, but got {image.shape[0]}.")
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = image[..., ::-1]
    return image

##储存结果图片
def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        input_img = tensor_to_image(inputs[i])
        target_img = tensor_to_image(targets[i])
        output_img = tensor_to_image(outputs[i])
        comparison = np.hstack((input_img, target_img, output_img))
        comparison_path = f'{folder_name}/epoch_{epoch}/result_{i + 1}.png'
        cv2.imwrite(comparison_path, comparison)

#下采样
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_norm=True):
        super(Downsample, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#上采样
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super(Upsample, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#生成器结构
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = Downsample(3, 64, use_norm=False)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 256)
        self.down5 = Downsample(256, 256)
        self.down6 = Downsample(256, 256)
        self.bottleneck = Downsample(256, 256, use_norm=False)
        self.up1 = Upsample(256, 256, use_dropout=True)
        self.up2 = Upsample(512, 256, use_dropout=True)
        self.up3 = Upsample(512, 256)
        self.up4 = Upsample(512, 128)
        self.up5 = Upsample(384, 64)
        self.up6 = Upsample(192, 64)
        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        bottleneck = self.bottleneck(x6)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, x6], dim=1))
        u3 = self.up3(torch.cat([u2, x5], dim=1))
        u4 = self.up4(torch.cat([u3, x4], dim=1))
        u5 = self.up5(torch.cat([u4, x3], dim=1))
        u6 = self.up6(torch.cat([u5, x2], dim=1))
        out = self.final(torch.cat([u6, x1], dim=1))
        return torch.tanh(out)

#判别器结构
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = Downsample(6, 64)
        self.down2 = Downsample(64, 128)
        self.conv1 = nn.Conv2d(128, 256, 3)
        self.bn = nn.BatchNorm2d(256)
        self.last = nn.Conv2d(256, 1, 3)

    def forward(self, anno, img):
        x = torch.cat([anno, img], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = F.dropout2d(self.bn(F.leaky_relu_(self.conv1(x))))
        x = self.last(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return torch.sigmoid(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)

optimizer_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

def train(epoch):
    for i, (real_B, real_A) in enumerate(train_dataloader):
        ##分别导入语义图片read_A(分割结果)与真实图片real_B(ground truth)
        real_A, real_B = real_A.to(device), real_B.to(device)

        ##先训练判别器
        optimizer_D.zero_grad()
        fake_B = G(real_A)#通过语义图片生成一张图像fake_B
        real_labels = torch.ones(real_A.size(0), 1).to(device)
        fake_labels = torch.zeros(real_A.size(0), 1).to(device)
        real_loss = criterion_GAN(D(real_A, real_B), real_labels)
        fake_loss = criterion_GAN(D(real_A, fake_B.detach()), fake_labels)
        d_loss = (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        ##再训练生成器
        optimizer_G.zero_grad()
        g_loss_GAN = criterion_GAN(D(real_A, fake_B), real_labels)
        g_loss_L1 = torch.mean(torch.abs(fake_B-real_B))
        g_loss = g_loss_GAN + 7.0*g_loss_L1
        g_loss.backward()
        optimizer_G.step()
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_dataloader)}], "
                  f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
        if epoch % 5 == 0 and i == 0:
            save_images(real_A, real_B, fake_B, 'train_results', epoch)

num_epochs = 800
for epoch in range(num_epochs):
    train(epoch)
