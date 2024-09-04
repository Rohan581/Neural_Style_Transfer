import torch.nn as nn
import torch

device = torch.device("cpu")

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding).to(device)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride).to(device)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = out.to(device)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1).to(device)
        self.in1 = nn.InstanceNorm2d(channels, affine=True).to(device)
        self.relu = nn.ReLU().to(device)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1).to(device)
        self.in2 = nn.InstanceNorm2d(channels, affine=True).to(device)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        out = out.to(device)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest').to(device)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding).to(device)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride).to(device)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = out.to(device)
        return out
