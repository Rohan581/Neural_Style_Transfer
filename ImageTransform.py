import torch
import torch.nn as nn
from Conv import ConvLayer, ResidualBlock, UpsampleConvLayer

device = torch.device("cpu")

class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU().to(device)
        self.tanh = nn.Tanh().to(device)

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1).to(device)
        self.in1_e = nn.InstanceNorm2d(32, affine=True).to(device)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2).to(device)
        self.in2_e = nn.InstanceNorm2d(64, affine=True).to(device)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2).to(device)
        self.in3_e = nn.InstanceNorm2d(128, affine=True).to(device)

        # residual layers
        self.res1 = ResidualBlock(128).to(device)
        self.res2 = ResidualBlock(128).to(device)
        self.res3 = ResidualBlock(128).to(device)
        self.res4 = ResidualBlock(128).to(device)
        self.res5 = ResidualBlock(128).to(device)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2).to(device)
        self.in3_d = nn.InstanceNorm2d(64, affine=True).to(device)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2).to(device)
        self.in2_d = nn.InstanceNorm2d(32, affine=True).to(device)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1).to(device)
        self.in1_d = nn.InstanceNorm2d(3, affine=True).to(device)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = self.tanh(self.in1_d(self.deconv1(y)))
        # y = self.deconv1(y)
        y = y.to(device)

        return y
