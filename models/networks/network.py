import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(AutoEncoder, self).__init__()

        channels = 16

        self.encoder = nn.Sequential(
            Conv2DBatchNormRelu(in_channels, channels, 3, 1, 0),
            Conv2DBatchNormRelu(channels, channels*2, 3, 1, 0),
            Conv2DBatchNormRelu(channels*2, channels*4, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            UpConv2DBatchNormRelu(channels*4, channels*2, 3, 1, 0),
            UpConv2DBatchNormRelu(channels*2, channels, 3, 1, 0),
            UpConv2DBatchNormRelu(channels, in_channels, 3, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


class UpConv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpConv2DBatchNormRelu, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x
