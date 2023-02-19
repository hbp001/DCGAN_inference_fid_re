import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


from utils import get_celeba
from dcgan import weights_init, Generator, Discriminator
from tqdm import tqdm
from focal_loss import FocalLoss

# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='focal_loss')
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2, help="interval between image sampling")
opt = parser.parse_args()

os.makedirs("./images/"+ opt.exp_name + "/", exist_ok=True)
os.makedirs("./model/"+ opt.exp_name + "/", exist_ok=True)
# Parameters to define the model.
params = {
    "bsize" : opt.batch_size,
    'imsize' : opt.img_size,
    'nc' : opt.nc,
    'nz' : opt.latent_dim,
    'ngf' : opt.ngf,
    'ndf' : opt.ndf, 
    'nepochs' :opt.n_epochs,
    'lr' : opt.lr,
    'beta1' : opt.beta1,
    'save_epoch' : opt.sample_interval}

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Get the data.
transform = transforms.Compose([
            transforms.Resize(params['imsize']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
dataset = datasets.STL10('./dataset/train/', split='train', download=False, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=params['bsize'], shuffle=True, num_workers=8)

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 64], padding=2, normalize=True).cpu(), (1, 2, 0)))

plt.show()

# Create the generator.
netG = Generator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model.
print(netG)

# Create the discriminator.
netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)

# Binary Cross Entropy loss function.
# criterion = nn.BCELoss()
criterion = FocalLoss(gamma=2.0, alpha=0.25)

fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizer for the discriminator.
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# Optimizer for the generator.
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

iters = 0

print("Starting Training Loop...")
print("-"*25)

for epoch in range(params['nepochs']):
    print("[Epoch %d/%d]" % (epoch, params['nepochs']))
    databar = tqdm(dataloader)
    for i, data in enumerate(databar):

        real_data = data[0].to(device)
        b_size = real_data.size(0)
        
        netD.zero_grad()
        label = torch.full((b_size, ), real_label, device=device)
        output = netD(real_data).view(-1)

        errD_real = criterion(output.to(torch.float32), label.to(torch.float32))
        errD_real.backward()
        D_x = output.mean().item()
        
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        
        fake_data = netG(noise)
        label.fill_(fake_label)

        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output.to(torch.float32), label.to(torch.float32))
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_data).view(-1)

        errG = criterion(output.to(torch.float32), label.to(torch.float32))
        errG.backward()

        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i%50 == 0:
            print(torch.cuda.is_available())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting.
        G_losses.append(errG.item())
        D_losses.append(errD.item())
       
        # save_image
        if epoch % 20 == 0 and i % 50 == 0:
            save_image(fake_data, "images/" + opt.exp_name + "/epoch_" + str(epoch) + ".png")

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

        databar.set_description('D_loss: %.3f   G_loss: %.3f' % (errD, errG))

    # Save the model.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'model/' + opt.exp_name + '/model_epoch_{}.pth'.format(epoch))

# Save the final trained model.
torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'model/' + opt.exp_name + '/model_final.pth')

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('STL.gif', dpi=80, writer='imagemagick')
