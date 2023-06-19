import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import itertools


import config
from utils import *

from tqdm import tqdm
from torchvision.utils import save_image
from models.Discriminator import PatchDisc
from models.Generator import ResNetGen
from datasets.Ukiyo import MyDataset


Buffer_A = ImageBuffer(50)
Buffer_F = ImageBuffer(50)

def fit(DiscA, DiscF, GenF_A, GenA_F,
        opt_D, opt_G, mse, l1, loader,
        Buffer_A, Buffer_F, lambda_a,
        lambda_b, lambda_idt=0, sched_D=None,
        sched_G=None, penalty=None, device=torch.device('cuda')):


    for i, (A_real, F_real) in enumerate(tqdm(loader)):
        A_real = A_real.to(device)
        F_real = F_real.to(device)

        #Train D_A
        A_fake = GenF_A(F_real)
        A_fake_buff = Buffer_A.extract(A_fake)
        D_A_real = DiscA(A_real)
        D_A_fake = DiscA(A_fake_buff)
        MSE_A_real = mse(D_A_real, torch.ones_like(D_A_real))
        MSE_A_fake = mse(D_A_fake, torch.zeros_like(D_A_fake))

        D_A_loss = (MSE_A_real + MSE_A_fake) / 2

        #Train D_F

        F_fake = GenA_F(A_real)
        F_fake_buff = Buffer_F.extract(F_fake)
        D_F_real = DiscF(F_real)
        D_F_fake = DiscF(F_fake_buff)
        MSE_F_real = mse(D_F_real, torch.ones_like(D_F_real))
        MSE_F_fake = mse(D_F_fake, torch.zeros_like(D_F_fake))
        D_F_loss = (MSE_F_real + MSE_F_fake) / 2

        if penalty is not None:
            D_F_loss += grad_penalty(DiscF, F_fake, F_real)
            D_A_loss += grad_penalty(DiscA, A_fake, A_real)

        D_loss = D_A_loss + D_F_loss
        
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()
        if sched_D is not None:
            sched_D.step()

        #Train Generator


        D_F_fake = DiscF(F_fake)
        D_A_fake = DiscA(A_fake)

        loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))
        loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))

        A_rec = GenF_A(F_fake)
        F_rec = GenA_F(A_fake)

        A_cycle_loss = l1(A_rec, A_real) * lambda_a
        F_cycle_loss = l1(F_rec, F_real) * lambda_b

        loss = loss_G_F + loss_G_A + A_cycle_loss + F_cycle_loss

        if lambda_idt > 0:

            idt_A = GenF_A(A_real)
            idt_F = GenA_F(F_real)

            idt_loss_A = l1(idt_A, A_real) * lambda_a * lambda_idt
            idt_loss_F = l1(idt_F, F_real) * lambda_b * lambda_idt

            loss += idt_loss_A + idt_loss_F

        opt_G.zero_grad()
        loss.backward()
        opt_G.step()
        if sched_G is not None:
            sched_G.step()

        if i % 500 == 0:
            x = torch.cat([A_real, F_fake, A_rec], dim=0)
            y = torch.cat([F_real, A_fake, F_rec], dim=0)
        plot_reconstruct(x, y)


def train(epoches):

    DiscA = PatchDisc()
    DiscA.apply(weights_init)

    DiscF = PatchDisc()
    DiscF.apply(weights_init)

    GenA_F = ResNetGen()
    GenA_F.apply(weights_init)

    GenF_A = ResNetGen()
    GenF_A.apply(weights_init)

    opt_D = torch.optim.Adam(
        itertools.chain(DiscA.parameters(), DiscF.parameters()),
        lr=config.lr, betas=(0.5, 0.999))

    opt_G = torch.optim.Adam(
        itertools.chain(GenA_F.parameters(), GenF_A.parameters()),
        lr=config.lr, betas=(0.5, 0.999))


    df = MyDataset(config.PATH_0,
                   config.PATH_1,
                   config.transform)

    train_dl = DataLoader(
        df, batch_size=config.BATCH, shuffle=True)

    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    lambda_a = config.LAMBDA_A
    lambda_b = config.LAMBDA_B
    lambda_idt = config.LAMBDA_IDT

    if config.LOAD:
        load_checkpoint(config.PATH_D_A, DiscA, opt_D, config.lr)
        load_checkpoint(config.PATH_D_F, DiscF, opt_D, config.lr)

        load_checkpoint(config.PATH_G_A_F, GenA_F, opt_G, config.lr)
        load_checkpoint(config.PATH_G_F_A, GenF_A, opt_G, config.lr)

    Buffer_A = ImageBuffer(50)
    Buffer_F = ImageBuffer(50)

    for epoch in range(epoches):

        fit(DiscA, DiscF, GenF_A, GenA_F,
            opt_D, opt_G, mse, l1,
            train_dl, Buffer_A, Buffer_F,
            lambda_a, lambda_b, lambda_idt,
            sched_D=None, sched_G=None, penalty=True)

        if config.SAVE:
            save_checkpoint(DiscF, opt_D, filename="DiscF.pth.tar")
            save_checkpoint(DiscA, opt_D, filename="DiscA.pth.tar")
            save_checkpoint(GenA_F, opt_G, filename="GenA_F.pth.tar")
            save_checkpoint(GenF_A, opt_G, filename="GenF_A.pth.tar")

