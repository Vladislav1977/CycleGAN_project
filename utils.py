import torch
import torch.nn as nn

import random

from  Generator import *


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def grad_penalty(disc, fake, real, lambda_gp):
    alpha = torch.rand(real.shape[0], 1, 1, 1)
    x_interpolate = alpha * real + (1 - alpha) * fake
    disc_interpolate = disc(x_interpolate)
    grad = torch.autograd.grad(
        inputs=x_interpolate,
        outputs=disc_interpolate,
        grad_outputs=torch.ones_like(disc_interpolate),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(real.shape[0], -1)
    grad = (torch.norm(grad, 2, dim=1) - 1) ** 2
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


buff = ImageBuffer(50)
imgs = torch.randn(2, 3, 256, 256)
print(buff.extract(imgs).shape)
print(buff.n_images)



"""def init_weights(m):
    print("m:", m)
    print("class:", m.__class__)
    print("class.name:", m.__class__.__name__)
    print("")
    if type(m) == nn.Conv2d:
        m.weight.data.fill_(1.0)
#        print(m.weight)

#net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
netG = ResNetGen(3, 64, 3)

netG.apply(init_weights)
"""