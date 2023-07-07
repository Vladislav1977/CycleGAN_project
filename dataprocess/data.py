import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as tt



 

class MyDataset(Dataset):

    def __init__(self, path_0, path_1):
        super().__init__()

        self.path_0 = path_0
        self.path_1 = path_1

        self.img_0 = os.listdir(path_0)
        self.img_1 = os.listdir(path_1)

        self.max_length = max(len(self.img_0), len(self.img_1))
        self.transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5),
                     (0.5, 0.5, 0.5))])

    def __getitem__(self, index):

        x = os.path.join(self.path_0, self.img_0[index % len(self.img_0)])
        y = os.path.join(self.path_1, self.img_1[index % len(self.img_1)])

        x = Image.open(x).convert("RGB")
        y = Image.open(y).convert("RGB")


        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return self.max_length