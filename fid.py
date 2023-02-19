import pytorch_fid_wrapper as pfw
import argparse
import os

from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torchvision.transforms as transforms

from dcgan import Generator
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='./model/bce/model_final.pth', help='Checkpoint to load path from')
parser.add_argument("--batch_size", default=4)
parser.add_argument("--channels", default=3)
parser.add_argument("--img_size", default=64)
args = parser.parse_args()

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

dataset = datasets.STL10('./dataset/test/', split='test', download=False, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

state_dict = torch.load(args.load_path)
params = state_dict['params']
netG = Generator(params).cuda()
netG.load_state_dict(state_dict['generator'], strict=False)

for a, (src, dst) in enumerate(tqdm(dataloader)):
    noise = torch.randn(src.shape[0], params['nz'], 1, 1, device=device)
    src, dst = src.to(device), dst.to(device)
    fake = netG(noise)

    if a == 0:
        fake_images = fake.detach().cpu()
        real_images = src.detach().cpu()
    else:
        fake_images = torch.cat([fake_images, fake.detach().cpu()])
        real_images = torch.cat([real_images, src.detach().cpu()])
print("fake_images : ", fake_images.shape)
print("real_images : ", real_images.shape)
print("FID Score : ", pfw.fid(fake_images, real_images))
