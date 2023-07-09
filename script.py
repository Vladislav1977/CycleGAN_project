from configs.config_inference import Config_test as Config
import torch
from models.Generator import ResNetGen
import torchvision.transforms as tt
from PIL import Image
from torchvision.utils import save_image
import argparse




opt = Config().parse()

#Model weight load

path_input = opt.src
path_output = opt.dest

model = ResNetGen().to(opt.device)

checkpoint = torch.load(opt.PATH_G_A_F_m if opt.male
                        else opt.PATH_G_A_F, map_location=opt.device)
model.load_state_dict(checkpoint["model_dict"])

transform = tt.Compose([
    tt.Resize(256),
    tt.CenterCrop(256),
    tt.ToTensor(),
    tt.Normalize((0.5, 0.5, 0.5),
                 (0.5, 0.5, 0.5))])

img = Image.open(path_input).convert("RGB")
img = transform(img)

res_img = model(img)
save_image(res_img * 0.5 + 0.5, path_output)





