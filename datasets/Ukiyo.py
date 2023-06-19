import torch
from PIL import Image
import os
from torch.utils.data import Dataset



 

class MyDataset(Dataset):

    def __init__(self, path_0, path_1, transform):
        super().__init__()

        self.path_0 = path_0
        self.path_1 = path_1

        self.img_0 = os.listdir(path_0)
        self.img_1 = os.listdir(path_1)

        self.max_length = max(len(self.img_0), len(self.img_1))
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.img_0[index % len(self.img_0)]).convert("RGB")
        y = Image.open(self.img_1[index % len(self.img_1)]).convert("RGB")

        x = self.transform(x)
        y = self.transform(y)

        return x, y

    def __len__(self):
        return self.max_length