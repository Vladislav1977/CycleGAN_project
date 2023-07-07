import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, input, out, kernel_size,
                 stride, padding, **kwargs):
        super().__init__()
        block = [nn.Conv2d(input, out, kernel_size, stride, padding, **kwargs),
                nn.InstanceNorm2d(out),
                nn.LeakyReLU(0.2, True)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)

class PatchDisc(nn.Module):

    def __init__(self, inc=3, nfc=64):
        super().__init__()

        self.first_block = nn.Sequential(
            nn.Conv2d(inc, nfc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.blocks = nn.Sequential(
            Block(nfc, nfc*2, 4, stride=2, padding=1),
            Block(nfc * 2, nfc * 4, kernel_size=4, stride=2, padding=1),
            Block(nfc * 4, nfc * 8, kernel_size=4, stride=2, padding=1)
        )
        self.final_block = nn.Conv2d(nfc * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.first_block(x)
        x = self.blocks(x)
        x = self.final_block(x)
        return x

