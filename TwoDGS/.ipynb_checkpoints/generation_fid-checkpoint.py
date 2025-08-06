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
import scipy
from scipy import linalg
from inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from fid.fastfid import fastfid 

def generate_2D_gaussian_splatting(kernel_size, scale, rotation, coords, colours, image_size=(256, 256, 3), device="cpu"):
    batch_size = colours.shape[0]

    # Ensure scale and rotation have the correct shape
    scale = scale.view(batch_size, 2)
    rotation = rotation.view(batch_size)

    # Compute the components of the covariance matrix
    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)

    R = torch.stack([
        torch.stack([cos_rot, -sin_rot], dim=-1),
        torch.stack([sin_rot, cos_rot], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(scale)

    # Compute covariance matrix: RSS^TR^T
    # covariance = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    # since in 2D , SS^T = SS and since transpose potentially could cause memory access issue
    covariance = R @ S @ S @ R.transpose(-1, -2)

    # Compute inverse covariance
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

    # Add padding to match image size
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


def give_required_data(input_coords, image_size, image_array, device):

  # normalising pixel coordinates [-1,1]
  coords = torch.tensor(input_coords / [image_size[0],image_size[1]], device=device).float()
  center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
  coords = (center_coords_normalized - coords) * 2.0

  # Fetching the colour of the pixels in each coordinates
  colour_values = [image_array[coord[1], coord[0]] for coord in input_coords]
  colour_values_np = np.array(colour_values)
  colour_values_tensor =  torch.tensor(colour_values_np, device=device).float()

  return colour_values_tensor, coords

def train(directory):
    with open('C:/Users/mahee/Desktop/dead leaves project/DiffDL/configs/2dgs_config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    KERNEL_SIZE = config["KERNEL_SIZE"]
    image_size = tuple(config["image_size"])
    primary_samples = config["primary_samples"]
    backup_samples = config["backup_samples"]
    num_epochs = config["num_epochs"]
    densification_interval = config["densification_interval"]
    learning_rate = config["learning_rate"]
    images_folder = config['images_folder']
    display_interval = config["display_interval"]
    grad_threshold = config["gradient_threshold"]
    gauss_threshold = config["gaussian_threshold"]
    display_loss = config["display_loss"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = primary_samples + backup_samples
    image_files = sorted(os.listdir(images_folder))
    num_images = len(image_files)

    all_source_tensors = []
    all_W = []
    all_masks = []
    all_optimizers = []
    all_schedulers = []

    for image_file in image_files:
        img = Image.open(os.path.join(images_folder, image_file)).resize((image_size[0],image_size[0])).convert('RGB')
        img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32, device=device)
        width, height, _ = img_tensor.shape
        all_source_tensors.append(img_tensor)

        coords = np.random.randint(0, [width, height], size=(num_samples, 2))
        colour_values, pixel_coords = give_required_data(coords, image_size, np.array(img), device)
        colour_values = torch.logit(colour_values)
        pixel_coords = torch.atanh(pixel_coords)

        scale_values = torch.logit(torch.rand(num_samples, 2, device=device))
        rotation_values = torch.atanh(2 * torch.rand(num_samples, 1, device=device) - 1)
        alpha_values = torch.logit(torch.rand(num_samples, 1, device=device))

        W_values = torch.cat([scale_values, rotation_values, alpha_values, colour_values, pixel_coords], dim=1)
        persistent_mask = torch.cat([torch.ones(primary_samples, dtype=bool), torch.zeros(backup_samples, dtype=bool)], dim=0)

        W = nn.Parameter(W_values)
        optimizer = Adam([W], lr=learning_rate)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=num_epochs)

        all_W.append(W)
        all_masks.append(persistent_mask)
        all_optimizers.append(optimizer)
        all_schedulers.append(scheduler)

    loss_history = []

    for epoch in range(num_epochs):
        g_tensor_list = []
        current_scales = []

        for idx in range(num_images):
            W = all_W[idx]
            persistent_mask = all_masks[idx]
            optimizer = all_optimizers[idx]
            scheduler = all_schedulers[idx]
            target_tensor = all_source_tensors[idx]

            # Densification / Pruning
            if epoch % (densification_interval + 1) == 0 and epoch > 0:
                indices_to_remove = (torch.sigmoid(W[:, 3]) < 0.01).nonzero(as_tuple=True)[0]
                persistent_mask[indices_to_remove] = False
                W.data[~persistent_mask] = 0.0

            gc.collect()
            torch.cuda.empty_cache()

            output = W[persistent_mask]
            scale = torch.sigmoid(output[:, 0:2])
            rotation = np.pi / 2 * torch.tanh(output[:, 2])
            alpha = torch.sigmoid(output[:, 3])
            colours = torch.sigmoid(output[:, 4:7])
            pixel_coords = torch.tanh(output[:, 7:9])
            colours_with_alpha = colours * alpha.view(-1, 1)

            g_tensor = generate_2D_gaussian_splatting(
                KERNEL_SIZE, scale, rotation, pixel_coords, colours_with_alpha, image_size, device=device
            )  # [H, W, 3]
            g_tensor_list.append(g_tensor.permute(2, 0, 1))  # to [C, H, W]

            if epoch + 1 == num_epochs:
                current_scales.append(scale)

        # === Compute loss over full batch ===
        g_tensor_batch = torch.stack(g_tensor_list)           # [N, C, H, W]
        target_batch = torch.stack([t.permute(2, 0, 1) for t in all_source_tensors])  # [N, C, H, W]
        loss = fastfid(g_tensor_batch, target_batch, gradient_checkpointing=True)

        # === Backprop per-image ===
        loss.backward()

        for idx in range(num_images):
            W = all_W[idx]
            mask = all_masks[idx]
            optimizer = all_optimizers[idx]
            scheduler = all_schedulers[idx]

            if mask is not None:
                W.grad.data[~mask] = 0.0
            optimizer.step()
            scheduler.step()

        loss_history.append(loss.item())

        if epoch % display_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # === Save per-image JSONs and preview ===
        if epoch + 1 == num_epochs:
            os.makedirs(directory, exist_ok=True)
            for idx, scale in enumerate(current_scales):
                W_final = all_W[idx][all_masks[idx]]
                alpha = torch.sigmoid(W_final[:, 3])

                avg_scale = scale.mean(dim=0).detach().cpu().numpy()
                avg_alpha = alpha.mean().item()

                image_name = os.path.splitext(image_files[idx])[0]
                output_json = {
                    "x_scale": float(avg_scale[0]),
                    "y_scale": float(avg_scale[1]),
                    "alpha": float(avg_alpha),
                }
                json_path = os.path.join(directory, f"{image_name}_scales.json")
                with open(json_path, "w") as f:
                    json.dump(output_json, f, indent=2)

                out_img = g_tensor_list[idx].permute(1, 2, 0).cpu().numpy()
                out_img = (out_img * 255).astype(np.uint8)
                img_pil = Image.fromarray(out_img)
                img_pil.save(os.path.join(directory, f"{image_name}_final.jpg"))

def compute_statistics(images, model, batch_size=32, dims=2048, device='cpu'):

    model.eval()
    images = images.to(device)
    n_images = images.shape[0]

    pred_arr = np.empty((n_images, dims))

    with torch.no_grad():
        for i in range(0, n_images, batch_size):
            batch = images[i:i+batch_size].to(device)

            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            pred = model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[i:i+batch.shape[0]] = pred

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_fid(source_images,generated_images, device, grayscale=True):

    source_images = preprocess_for_fid(source_images)
    generated_images = preprocess_for_fid(generated_images)

    dims=2048

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    mu1, sigma1 = compute_statistics(source_images, model,device=device)
    mu2, sigma2 = compute_statistics(generated_images, model, device=device)

    fid = calculate_fid(mu1, sigma1, mu2, sigma2)

    return fid

def preprocess_for_fid(tensor):
    if tensor.ndim == 3:  # single image: [H, W, C] or [C, H, W]
        if tensor.shape[0] <= 3:  # [C, H, W] likely
            tensor = tensor.unsqueeze(0)  # [1, C, H, W]
        else:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    elif tensor.ndim == 4 and tensor.shape[1] != 3:
        # possibly [N, H, W, C], convert to [N, C, H, W]
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor

if __name__=='__main__':

    train('./runs/19')
