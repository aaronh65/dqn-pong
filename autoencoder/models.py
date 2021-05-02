import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# AUTOENCODER MODELS


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1), 
            torch.nn.MaxPool2d(2), # 88 -> 44
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.MaxPool2d(2), # 44 -> 22
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.MaxPool2d(2), # 22 -> 11
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 64, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )


    def forward(self, x):
        x = self.module(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.upsample = nn.Sequential(
            DecoderBlock(64, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
        )
        self.project = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.project(x)
        x = F.sigmoid(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.module(x)

class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResnetDecoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.upsample = nn.Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
        )
        self.project = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.project(x)
        x = F.sigmoid(x)
        return x










