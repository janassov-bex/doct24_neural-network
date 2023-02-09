import torch
import torch.nn as nn


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1)
        self.conv1.to(device)
        self.bn2d1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2d1.to(device)
        self.conv2 = torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1)
        self.conv2.to(device)
        self.bn2d2 = torch.nn.BatchNorm2d(out_channels)
        self.bn2d2.to(device)

    def forward(self, inp):
        out = torch.nn.functional.relu(self.conv1(inp))
        out = self.bn2d1(out)
        out = torch.nn.functional.relu(self.conv2(out))
        out = self.bn2d2(out)
        return out


class FinalBlock(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, kernel_size, device):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(kernel_size=(4, 4), in_channels=in_channels, out_channels=mid_channel, padding=1)
        self.conv1.to(device)
        self.bn2d1 = torch.nn.BatchNorm2d(mid_channel)
        self.bn2d1.to(device)
        self.conv2 = torch.nn.Conv2d(kernel_size=(4, 4), in_channels=mid_channel, out_channels=mid_channel, padding=1)
        self.conv2.to(device)
        self.bn2d2 = torch.nn.BatchNorm2d(mid_channel)
        self.bn2d2.to(device)
        self.conv3 = torch.nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channel, out_channels=out_channels, padding=1)
        self.conv3.to(device)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(80, 64)
        self.linear1.to(device)
        self.linear2 = nn.Linear(64, 32)
        self.linear2.to(device)
        self.linear3 = nn.Linear(32, 8)
        self.linear3.to(device)

    def forward(self, inp):
        out = torch.nn.functional.relu(self.conv1(inp))
        out = self.bn2d1(out)
        out = torch.nn.functional.relu(self.conv2(out))
        out = self.bn2d2(out)
        out = torch.sigmoid(self.conv3(out))
        out = self.flatten(out)
        out = torch.sigmoid(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = torch.nn.functional.softmax(self.linear3(out))
        return out


class UNet_PredictLungMany(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(UNet_PredictLungMany  , self).__init__()
        self.conv1 = DownSampling(in_channels, 64, 3, device)
        self.conv1.to(device)
        self.conv2 = DownSampling(64, 48, 3, device)
        self.conv2.to(device)
        self.conv3 = DownSampling(48, 40, 3, device)
        self.conv3.to(device)
        self.conv4 = DownSampling(40, 36, 3, device)
        self.conv4.to(device)
        self.conv5 = DownSampling(36, 28, 3, device)
        self.conv5.to(device)
        self.final = FinalBlock(28, 24, 20, out_channels, device)
        self.final.to(device)
        self.conv_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_maxpool.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_maxpool(x)
        x = self.conv2(x)
        x = self.conv_maxpool(x)
        x = self.conv3(x)
        x = self.conv_maxpool(x)
        x = self.conv4(x)
        x = self.conv_maxpool(x)
        x = self.conv5(x)
        x = self.conv_maxpool(x)
        x = self.final(x)
        return x

