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
from torch.utils.checkpoint import checkpoint

def generate_gaussian_batch_checkpointed(coords, colours, scale_all, rotation_all, xy_base, padding, device, start_idx, end_idx):
    #checkpointed function for single batch of gaussians
    
    current_batch = end_idx - start_idx
    
    scale = scale_all[start_idx:end_idx]
    rotation = rotation_all[start_idx:end_idx] 
    coords_batch = coords[start_idx:end_idx]
    colours_batch = colours[start_idx:end_idx]

    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)

    R = torch.stack([
        torch.stack([cos_rot, -sin_rot], dim=-1),
        torch.stack([sin_rot, cos_rot], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(scale)
    covariance = R @ S @ S @ R.transpose(-1, -2)
    inv_covariance = torch.inverse(covariance)

    xy = xy_base.unsqueeze(0).expand(current_batch, -1, -1, -1)
    z = torch.einsum('bxyi,bij,bxyj->bxy', xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance))).view(current_batch, 1, 1)

    # Normalize the kernel
    kernel_max = kernel.amax(dim=(-2, -1), keepdim=True)
    kernel_normalized = kernel / kernel_max

    # Reshape the kernel for RGB channels
    kernel_rgb = kernel_normalized.unsqueeze(1).expand(-1, 3, -1, -1)
    kernel_rgb_padded = F.pad(kernel_rgb, padding, "constant", 0)

    # Translate the kernel
    b, c, h, w = kernel_rgb_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords_batch

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    # Apply colors and sum the layers
    rgb_values_reshaped = colours_batch.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    partial_image = final_image_layers.sum(dim=0)
    
    return partial_image

def generate_dead_leaves_images(rmin, rmax, alpha, color_histogram, num_gaussians,batch_size=2000, kernel_size=101, device='cpu', image_size=(500,500,3)):
    
    hist_r, hist_g, hist_b = color_histogram

    rand = torch.rand((num_gaussians,), device=device)
    tmp = (rmax ** (1-alpha)) + ((rmin ** (1-alpha)) - (rmax ** (1-alpha))) * rand
    radius_all = tmp ** (-1 / (alpha - 1))

    #GAUSSIAN PARAMETERS

    coords_all = torch.rand((num_gaussians, 2), device=device) * 2 - 1
    colours_all = sample_rgb_histogram(hist_b=hist_b, hist_r=hist_r, hist_g=hist_g, 
                                     num_gaussians=num_gaussians, device=device)
    scale_all = torch.stack([radius_all, radius_all], dim=1)
    rotation_all = torch.zeros((num_gaussians,), device=device)

    #kernel creation
    x = torch.linspace(-5, 5, kernel_size, device=device)
    y = torch.linspace(-5, 5, kernel_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xy_base = torch.stack([xx, yy], dim=-1)

    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    
    #batch processing of gaussians with checkpointing
    partial_images = []
    for start in range(0, num_gaussians, batch_size):
        end = min(start + batch_size, num_gaussians)
        
        partial_image = checkpoint(
            generate_gaussian_batch_checkpointed,
            coords_all, colours_all, scale_all, rotation_all,
            xy_base, padding, device, start, end,
            use_reentrant=False  # Use the newer, more efficient checkpointing
        )
        
        partial_images.append(partial_image.permute(1, 2, 0))
    
    final_image = torch.stack(partial_images, dim=0).sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    
    return final_image

def train():

    #configs
    category = 'beach'
    source_images_path = f'./source_images/{category}'
    output_directory = f'./runs/{category}'
    rmin = 1.0
    rmax = 1000.0
    alpha = 3.0
    width = 500
    length = 500
    num_images = 0
    num_samples = 5
    wandb_log = True
    loss_type = 'ffl'
    save_freq= 10

    kernel_size=101
    batch_size=1000

    num_epochs = 500
    learning_rate = 0.1
    num_gaussians =10000

    run_no = get_run_no(output_directory)
    run_dir = os.path.join(output_directory, str(run_no))
    os.makedirs(run_dir, exist_ok=True)


    if wandb_log:
        wandb.init(
            project="diffdl", 
            name=f"{category}{run_no}", 
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

    for epoch in range(1,num_epochs+1):

        gc.collect()
        torch.cuda.empty_cache()
        
        # W_output = W
        rmin = RMIN
        rmax = RMAX
        alpha = ALPHA

        g_tensor_batch = generate_dead_leaves_images(rmin, rmax, alpha, color_histogram=hist, batch_size=batch_size, num_gaussians=num_gaussians, device=device)
        # print(target_tensor.shape,g_tensor_batch.shape)

        # plt.imshow(g_tensor_batch.detach().cpu().numpy()) 
        # plt.axis('off')
        # plt.show()
        
        generated_batch = g_tensor_batch.permute(2, 0, 1).unsqueeze(0)
        target_batch = target_tensor.permute(2, 0, 1).unsqueeze(0)
        loss = ffl(generated_batch, target_batch)

        print(f"Epoch: {epoch}, Loss: {loss}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if wandb_log:
            wandb.log({
                "epoch": epoch,
                f"{loss_type}_loss": loss.item(),
                "lr": learning_rate,
                "rmin": RMIN.item(),
                "rmax": RMAX.item(),
                "alpha": ALPHA.item()
            })

        if epoch % save_freq == 0 or epoch == 1:
            generated_array = g_tensor_batch.cpu().detach().numpy()
            img = Image.fromarray((generated_array * 255).astype(np.uint8))
            filename = f"{epoch}.png"
            file_path = os.path.join(run_dir, filename)
            img.save(file_path)


    print(RMIN, RMAX, ALPHA)


def get_run_no(output_dir):
    runs = os.listdir(output_dir)
    runs.sort()
    run_no = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    return run_no

if __name__ == '__main__':

    train()
