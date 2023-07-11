import torch
import matplotlib.pyplot as plt

import random

from models.Generator import *


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def grad_penalty(disc, fake, real, device, lambda_gp=10, constant=1):
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=device)
    x_interpolate = alpha * real + (1 - alpha) * fake
    x_interpolate.requires_grad_(True)
    disc_interpolate = disc(x_interpolate)
    grad = torch.autograd.grad(
        inputs=x_interpolate,
        outputs=disc_interpolate,
        grad_outputs=torch.ones_like(disc_interpolate),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(real.shape[0], -1)
    grad = ((torch.norm(grad + 1e-16, 2, dim=1) - constant) ** 2) / (constant ** 2)
    grad_penalty = grad.mean() * lambda_gp
    return grad_penalty

class ImageBuffer:

    def __init__(self, bufsize):
        self.bufsize = bufsize
        self.n_images = 0
        self.buff = []

    def extract(self, images):

        return_images = []

        for image in images:
            if self.n_images < self.bufsize:
                return_images.append(image)
                self.buff.append(image)
                self.n_images += 1
            else:
                if random.uniform(0, 1) >= 0.5:
                    return_images.append(image)
                else:
                    idx = random.randint(0, self.bufsize-1)
                    img_return = self.buff[idx].clone()
                    return_images.append(img_return)
                    self.buff[idx] = image
        return torch.stack(return_images, dim=0)


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar", mode="Kaggle"):
    print("Saving checkpoint")
    checkpoint = {
        "model_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if mode == "Kaggle":
        torch.save(checkpoint, filename)
    elif mode =="Collab":
        path = f'/content/gdrive/MyDrive/models/GAN/{filename}'
        torch.save(checkpoint, path)
"""
def init_collab(mode):
    if mode =="Collab":
        from google.colab import drive
        drive.mount('/content/gdrive')
    else:
        pass """

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):

    print("Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def plot_reconstruct(x, y):

    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    names = ["real", "fake", "reconstructed"]
    for i in range(3):
        #    i = np.random.randint(13143)
        axes[0][i].imshow(torch.permute(x[i] * 0.5 + 0.5, (1, 2, 0)).detach().to("cpu"))
        axes[0][i].set_title(names[i])
        axes[1][i].imshow(torch.permute(y[i] * 0.5 + 0.5, (1, 2, 0)).detach().to("cpu"))
    plt.show()


def scheduler(optim, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - 100) / float(100 + 1)
        return lr_l
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda_rule)