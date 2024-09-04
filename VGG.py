import torch
import torch.nn as nn
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cpu")

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        device = torch.device("mps")
        features = models.vgg16(weights='VGG16_Weights.DEFAULT').features.type(torch.float32).to(device)
        self.to_relu_1_2 = nn.Sequential().type(torch.float32).to(device)
        self.to_relu_2_2 = nn.Sequential().type(torch.float32).to(device)
        self.to_relu_3_3 = nn.Sequential().type(torch.float32).to(device)
        self.to_relu_4_3 = nn.Sequential().type(torch.float32).to(device)

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.type(torch.float32)
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h.type(torch.float32)
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h.type(torch.float32)
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h.type(torch.float32)
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h.type(torch.float32)
        out = (h_relu_1_2.type(torch.float32), h_relu_2_2.type(torch.float32), h_relu_3_3.type(torch.float32), h_relu_4_3.type(torch.float32))
        return out
