import os
import yaml
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np
import cv2
from time import time
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from datetime import datetime
from torch.optim.lr_scheduler import LinearLR 
import gc
import json
from utils import sampling_distribution_rgb_histogram, get_distribution_rgb_histogram, sample_rgb_histogram
from TwoDGS.loss import ssim, combined_loss, d_ssim_loss
from focal_frequency_loss import FocalFrequencyLoss as FFL
import wandb

def generate_dead_leaves_images(rmin, rmax, alpha, color_histogram, num_gaussians,kernel_size=101, device='cpu', image_size=(500,500,3)):

    batch_size= num_gaussians #batch here refers to the number of gaussians to be plotted
    hist_r, hist_g, hist_b = color_histogram

    rand = torch.rand((num_gaussians,), device=device)
    tmp = (rmax ** (1-alpha)) + ((rmin ** (1-alpha)) - (rmax ** (1-alpha))) * rand
    radius = tmp ** (-1 / (alpha - 1))

    # print("radius requires_grad:", radius.requires_grad)

    #GAUSSIAN LOGIC

    coords = torch.rand((num_gaussians, 2), device=device) * 2 - 1
    colours = sample_rgb_histogram(hist_b=hist_b, hist_r=hist_r, hist_g=hist_g, num_gaussians=num_gaussians, device=device)
    scale = torch.stack([radius, radius], dim=1)
    # print("scale requires_grad:",scale.requires_grad)
    rotation = torch.zeros((num_gaussians,), device=device)

    #covariance matrix components
    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)

    R = torch.stack([
        torch.stack([cos_rot, -sin_rot], dim=-1),
        torch.stack([sin_rot, cos_rot], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(scale)
    covariance = R @ S @ S @ R.transpose(-1, -2)
    # print("covariance requires_grad:",covariance.requires_grad)

    inv_covariance = torch.inverse(covariance)

    # Create the kernel
    x = torch.linspace(-5, 5, kernel_size, device=device)
    y = torch.linspace(-5, 5, kernel_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    z = torch.einsum('bxyi,bij,bxyj->bxy', xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance))).view(batch_size, 1, 1)

    # Normalize the kernel
    kernel_max = kernel.amax(dim=(-2, -1), keepdim=True)
    kernel_normalized = kernel / kernel_max

    # Reshape the kernel for RGB channels
    kernel_rgb = kernel_normalized.unsqueeze(1).expand(-1, 3, -1, -1)

    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    kernel_rgb_padded = F.pad(kernel_rgb, padding, "constant", 0)

    # Translate the kernel
    b, c, h, w = kernel_rgb_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    # Apply colors and sum the layers
    rgb_values_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)

    return final_image


def train():

    #configs
    source_images_path = './source_images/beach'
    # source_image_path = './source_images/beach.jpg' #only to test gradient propagation !!!
    output_directory = './temp/beach/4'
    rmin = 1.0
    rmax = 1000.0
    alpha = 3.0
    width = 500
    length = 500
    num_images = 0
    num_epochs = 100
    batch_size = 1
    num_samples = 5
    num_gaussians = 200

    kernel_size=101

    learning_rate = 0.1

    wandb.init(
        project="diffdl", 
        name="beach4", 
        config={
            "lr": learning_rate,
            "num_gaussians": num_gaussians,
            "batch_size": 1,
            "epochs": num_epochs,
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #process source image(s)
    source_images = []
    source_images_raw = []
    for source_img_name in os.listdir(source_images_path):
        source_img = cv2.imread(os.path.join(source_images_path, source_img_name))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        source_img = cv2.resize(source_img, (500,500))
        source_images_raw.append(source_img)
        source_img = np.array(source_img) /255.0
        source_images.append(source_img)

    target_tensor = torch.tensor(np.stack(source_images), dtype=torch.float32, device=device)

    #!!!
    target_tensor = target_tensor[0]

    #build a color histogram from all source images
    hist = get_distribution_rgb_histogram(source_images_raw,visualize=False)
    # exit()

    #global parameters
    RMIN = nn.Parameter(torch.tensor(rmin, device=device,requires_grad=True),requires_grad=True)
    RMAX = nn.Parameter(torch.tensor(rmax, device=device,requires_grad=True),requires_grad=True)
    ALPHA = nn.Parameter(torch.tensor(alpha, device=device,requires_grad=True),requires_grad=True)

    optimizer = Adam([RMIN, RMAX, ALPHA], lr=learning_rate)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=num_epochs)
    loss_history = []

    ffl = FFL(loss_weight=1.0, alpha=1.0) 

    for epoch in range(num_epochs):

        #skipping the pruning process

        gc.collect()
        torch.cuda.empty_cache()
        
        # W_output = W
        rmin = RMIN
        rmax = RMAX
        alpha = ALPHA

        g_tensor_batch = generate_dead_leaves_images(rmin, rmax, alpha, color_histogram=hist, num_gaussians=num_gaussians, device=device)
        # print(target_tensor.shape,g_tensor_batch.shape)

        # plt.imshow(g_tensor_batch.detach().cpu().numpy()) 
        # plt.axis('off')
        # plt.show()
        
        # loss = combined_loss(g_tensor_batch, target_tensor, lambda_param=0.2)  
        generated_batch = g_tensor_batch.permute(2, 0, 1).unsqueeze(0)
        target_batch = target_tensor.permute(2, 0, 1).unsqueeze(0)
        loss = ffl(generated_batch, target_batch)

        print(f"Epoch: {epoch}, Loss: {loss}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        wandb.log({
            "epoch": epoch,
            "ffl_loss": loss.item(),
            "lr": learning_rate,
            "rmin": RMIN.item(),
            "rmax": RMAX.item(),
            "alpha": ALPHA.item()
        })
        
        generated_array = g_tensor_batch.cpu().detach().numpy()
        img = Image.fromarray((generated_array * 255).astype(np.uint8))
        filename = f"{epoch}.png"
        file_path = os.path.join(output_directory, filename)
        img.save(file_path)


    print(RMIN, RMAX, ALPHA)


if __name__ == '__main__':

    train()
