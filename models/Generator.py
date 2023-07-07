import torch
import torch.nn as nn



class Convblock(nn.Module):

    def __init__(self, input, out, kernel_size,
                 stride, padding, down=True, **kwargs):
        super().__init__()
        block = [nn.Conv2d(input, out, kernel_size, stride, padding, **kwargs)
                if down
                else nn.ConvTranspose2d(input, out, kernel_size, stride, padding, **kwargs),
                nn.InstanceNorm2d(out),
                nn.ReLU(True)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)

class ResBlock(nn.Module):

    def __init__(self, dim, padding=1, **kwargs):
        super().__init__()
        block = [nn.Conv2d(dim, dim, 3, 1, padding=padding, **kwargs),
                 nn.InstanceNorm2d(dim),
                 nn.ReLU(True)]
        block += [nn.Conv2d(dim, dim, 3, 1, padding=padding, **kwargs),
                  nn.InstanceNorm2d(dim)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        x = x + self.conv_block(x)
        return x

class ResNetGen(nn.Module):

    def __init__(self, inc=3, nfc=64, out=3):
        super().__init__()
        self.conv_down = nn.Sequential(
            Convblock(input=inc, out=nfc, kernel_size=7,
                      stride=1, padding=3, padding_mode='reflect'),

            Convblock(input=nfc, out=nfc * 2, kernel_size=3,
                      stride=2, padding=1),

            Convblock(input=nfc * 2, out=nfc * 4, kernel_size=3,
                      stride=2, padding=1)
        )
        residual = [ResBlock(nfc * 4, padding=1, padding_mode="reflect") for i in range(9)]
        self.residual = nn.Sequential(*residual)

        self.up = nn.Sequential(
            Convblock(input=nfc * 4, out=nfc * 2, kernel_size=3,
                      stride=2, padding=1,
                      output_padding=1, down=False),

            Convblock(input=nfc * 2, out=nfc, kernel_size=3,
                      stride=2, padding=1,
                      output_padding=1, down=False)
        )
        self.final = nn.Conv2d(nfc, out, kernel_size=7, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.conv_down(x)
        x = self.residual(x)
        x = self.up(x)
        x = torch.tanh(self.final(x))
        return x





"""a = [nn.Conv2d(2, 2, 3, 1, padding_mode="reflect"),
                 nn.InstanceNorm2d(2),
                 nn.ReLU(True)]

a1 = nn.Sequential(*a)
print(nn.Sequential(*a1, *a1))
"""