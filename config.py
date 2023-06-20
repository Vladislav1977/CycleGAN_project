import torch


PATH_0 = r"/kaggle/input/ukiyoe2photo/ukiyoe2photo/trainA"
PATH_1 = r"/kaggle/input/ukiyoe2photo/ukiyoe2photo/trainB"
BATCH = 1
LOAD = False
lr = 0.0002
LAMBDA_A = 10
LAMBDA_B = 10
LAMBDA_IDT = 0.5
LOAD = False
PATH_D_A = ""
PATH_D_F = ""
PATH_G_A_F = ""
PATH_G_F_A = ""
SAVE = False
device = torch.device('cuda')