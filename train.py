from torch.utils.data import DataLoader

import itertools


from configs.config_train import Config
from utils import *

from tqdm import tqdm
from models.Discriminator import PatchDisc
from models.Generator import ResNetGen
from dataprocess.data import MyDataset




def fit(DiscA, DiscF, GenF_A, GenA_F,
        opt_D, opt_G, mse, l1, loader,
        Buffer_A, Buffer_F, device, lambda_a=10,
        lambda_b=10, lambda_idt=0, sched_D=None,
        sched_G=None, penalty=False):


    for i, (A_real, F_real) in enumerate(tqdm(loader)):
        A_real = A_real.to(device)
        F_real = F_real.to(device)

        #Train D_A
        A_fake = GenF_A(F_real)
        A_fake_buff = Buffer_A.extract(A_fake.detach())
        D_A_real = DiscA(A_real)
        D_A_fake = DiscA(A_fake_buff)
        MSE_A_real = mse(D_A_real, torch.ones_like(D_A_real))
        MSE_A_fake = mse(D_A_fake, torch.zeros_like(D_A_fake))

        D_A_loss = (MSE_A_real + MSE_A_fake) / 2

        #Train D_F

        F_fake = GenA_F(A_real)
        F_fake_buff = Buffer_F.extract(F_fake.detach())
        D_F_real = DiscF(F_real)
        D_F_fake = DiscF(F_fake_buff)
        MSE_F_real = mse(D_F_real, torch.ones_like(D_F_real))
        MSE_F_fake = mse(D_F_fake, torch.zeros_like(D_F_fake))
        D_F_loss = (MSE_F_real + MSE_F_fake) / 2

        if penalty:
            D_F_loss += grad_penalty(DiscF, F_fake_buff, F_real)
            D_A_loss += grad_penalty(DiscA, A_fake_buff, A_real)

        D_loss = D_A_loss + D_F_loss
        
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()


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


        if i % 1500 == 0:
            x = torch.cat([A_real, F_fake, A_rec], dim=0)
            y = torch.cat([F_real, A_fake, F_rec], dim=0)
            plot_reconstruct(x, y)


    if sched_G is not None:
        print("Current Lr:", opt_D.param_groups[0]["lr"])
        sched_G.step()
        print("Lr after scheduling:", opt_D.param_groups[0]["lr"])

    if sched_D is not None:
        sched_D.step()



def train(opt, sched_G=None, sched_D=None):

    DiscA = PatchDisc().to(opt.device)
    DiscA.apply(weights_init)

    DiscF = PatchDisc().to(opt.device)
    DiscF.apply(weights_init)

    GenA_F = ResNetGen().to(opt.device)
    GenA_F.apply(weights_init)

    GenF_A = ResNetGen().to(opt.device)
    GenF_A.apply(weights_init)

    opt_D = torch.optim.Adam(
        itertools.chain(DiscA.parameters(), DiscF.parameters()),
        lr=opt.lr, betas=(0.5, 0.999))

    opt_G = torch.optim.Adam(
        itertools.chain(GenA_F.parameters(), GenF_A.parameters()),
        lr=opt.lr, betas=(0.5, 0.999))

    if sched_D is not None:
        sched_D = scheduler(opt_D, opt)

    if sched_G is not None:
        sched_G = scheduler(opt_G, opt)




    df = MyDataset(opt.PATH_A,
                   opt.PATH_B)

    train_dl = DataLoader(
        df, batch_size=opt.BATCH, shuffle=True)

    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    lambda_a = opt.LAMBDA_A
    lambda_b = opt.LAMBDA_B
    lambda_idt = opt.LAMBDA_IDT

    if opt.LOAD:
        load_checkpoint(opt.PATH_D_A, DiscA, opt_D, opt.lr, device=opt.device)
        load_checkpoint(opt.PATH_D_F, DiscF, opt_D, opt.lr, device=opt.device)

        load_checkpoint(opt.PATH_G_A_F, GenA_F, opt_G, opt.lr, device=opt.device)
        load_checkpoint(opt.PATH_G_F_A, GenF_A, opt_G, opt.lr, device=opt.device)

    Buffer_A = ImageBuffer(50)
    Buffer_F = ImageBuffer(50)

    for epoch in range(opt.epoches):

        fit(DiscA, DiscF, GenF_A, GenA_F,
            opt_D, opt_G, mse, l1,
            train_dl, Buffer_A, Buffer_F,
            lambda_a, lambda_b, lambda_idt,
            sched_D=sched_D, sched_G=sched_G, penalty=opt.penalty)

        if opt.SAVE:
            save_checkpoint(DiscF, opt_D, filename="DiscF.pth.tar")
            save_checkpoint(DiscA, opt_D, filename="DiscA.pth.tar")
            save_checkpoint(GenA_F, opt_G, filename="GenA_F.pth.tar")
            save_checkpoint(GenF_A, opt_G, filename="GenF_A.pth.tar")

if __name__ == "__main__":
    opt = Config().parse()
    train(opt, sched_G=True, sched_D=True)