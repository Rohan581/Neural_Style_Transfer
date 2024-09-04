import torch
from PIL import Image
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import Dataset
import os
device = torch.device("cpu")

class ImageOps():
    def __init__(self, filename: str, data):
        self.filename = filename
        self.data = data

    def load_image(self):
        img = Image.open(self.filename)
        return img

    def save_image(self):
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img = self.data.clone().numpy()
        img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        img = Image.fromarray(img)
        img.save(self.filename)


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image.type(torch.float32).to(device)


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


def normalize_tensor_transform():
    return v2.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]).to(device)
