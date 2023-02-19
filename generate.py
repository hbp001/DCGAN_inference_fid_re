import argparse

import torch
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os

from dcgan import Generator
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=500, help='Number of generated outputs')
parser.add_argument("--output_size", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument('--exp_name', default='bce', help='Checkpoint to load path from')
args = parser.parse_args()

# Load the checkpoint file.
os.makedirs("images_test/"+args.exp_name, exist_ok=True)
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print("num_output : ", args.num_output)
print("output_size : ", args.output_size)
for num in range(args.num_output):
    # Get latent vector Z from unit normal distribution.
    print("num : ", num)
    noise = torch.randn(args.output_size, params['nz'], 1, 1, device=device)

    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
        # Get generated image from the noise vector using
        # the trained generator.
        generated_img = netG(noise).detach().cpu()
        save_image(generated_img, "images_test/"+args.exp_name+"/%d.png" % num)

# Display the generated image.
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

plt.show()
