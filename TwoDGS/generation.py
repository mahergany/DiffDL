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
from TwoDGS.loss import ssim, combined_loss, d_ssim_loss

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

    # Read the config.yml file
    with open('C:/Users/mahee/Desktop/dead leaves project/DiffDL/configs/2dgs_config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract values from the loaded config
    KERNEL_SIZE = config["KERNEL_SIZE"]
    image_size = tuple(config["image_size"])
    primary_samples = config["primary_samples"]
    backup_samples = config["backup_samples"]
    num_epochs = config["num_epochs"]
    densification_interval = config["densification_interval"]
    learning_rate = config["learning_rate"]
    image_file_name = config["image_file_name"]
    display_interval = config["display_interval"]
    grad_threshold = config["gradient_threshold"]
    gauss_threshold = config["gaussian_threshold"]
    display_loss = config["display_loss"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = primary_samples + backup_samples

    PADDING = KERNEL_SIZE // 2
    image_path = image_file_name
    original_image = Image.open(image_path)
    original_image = original_image.resize((image_size[0],image_size[0]))
    original_image = original_image.convert('RGB')
    original_array = np.array(original_image)
    original_array = original_array / 255.0
    width, height, _ = original_array.shape

    image_array = original_array
    target_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)
    coords = np.random.randint(0, [width, height], size=(num_samples, 2))
    random_pixel_means = torch.tensor(coords, device=device)
    pixels = [image_array[coord[0], coord[1]] for coord in coords]
    pixels_np = np.array(pixels)
    random_pixels =  torch.tensor(pixels_np, device=device)

    colour_values, pixel_coords = give_required_data(coords, image_size, image_array, device)

    colour_values = torch.logit(colour_values)
    pixel_coords = torch.atanh(pixel_coords)

    scale_values = torch.logit(torch.rand(num_samples, 2, device=device))
    rotation_values = torch.atanh(2 * torch.rand(num_samples, 1, device=device) - 1)
    alpha_values = torch.logit(torch.rand(num_samples, 1, device=device))
    W_values = torch.cat([scale_values, rotation_values, alpha_values, colour_values, pixel_coords], dim=1)

    starting_size = primary_samples
    left_over_size = backup_samples
    persistent_mask = torch.cat([torch.ones(starting_size, dtype=bool),torch.zeros(left_over_size, dtype=bool)], dim=0)
    current_marker = starting_size

    W = nn.Parameter(W_values)
    optimizer = Adam([W], lr=learning_rate)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=num_epochs)
    loss_history = []

    scales = []

    for epoch in range(num_epochs):

        #find indices to remove and update the persistent mask
        if epoch % (densification_interval + 1) == 0 and epoch > 0:
            indices_to_remove = (torch.sigmoid(W[:, 3]) < 0.01).nonzero(as_tuple=True)[0]

            if len(indices_to_remove) > 0:
                print(f"number of pruned points: {len(indices_to_remove)}")

            persistent_mask[indices_to_remove] = False

            # Zero-out parameters and their gradients at every epoch using the persistent mask
            W.data[~persistent_mask] = 0.0


        gc.collect()
        torch.cuda.empty_cache()

        output = W[persistent_mask]

        batch_size = output.shape[0]

        scale = torch.sigmoid(output[:, 0:2])

        if epoch + 1 == num_epochs:
            scales.append(scale)

        rotation = np.pi / 2 * torch.tanh(output[:, 2])
        alpha = torch.sigmoid(output[:, 3])
        colours = torch.sigmoid(output[:, 4:7])
        pixel_coords = torch.tanh(output[:, 7:9])

        colours_with_alpha  = colours * alpha.view(batch_size, 1)
        g_tensor_batch = generate_2D_gaussian_splatting(KERNEL_SIZE, scale, rotation, pixel_coords, colours_with_alpha, image_size, device=device)
        loss = combined_loss(g_tensor_batch, target_tensor, lambda_param=0.2)

        optimizer.zero_grad()

        loss.backward()

        # Apply zeroing out of gradients at every epoch
        if persistent_mask is not None:
            W.grad.data[~persistent_mask] = 0.0

        if epoch % densification_interval == 0 and epoch > 0:

            # Calculate the norm of gradients
            gradient_norms = torch.norm(W.grad[persistent_mask][:, 7:9], dim=1, p=2)
            gaussian_norms = torch.norm(torch.sigmoid(W.data[persistent_mask][:, 0:2]), dim=1, p=2)

            sorted_grads, sorted_grads_indices = torch.sort(gradient_norms, descending=True)
            sorted_gauss, sorted_gauss_indices = torch.sort(gaussian_norms, descending=True)

            large_gradient_mask = (sorted_grads > grad_threshold)
            large_gradient_indices = sorted_grads_indices[large_gradient_mask]

            large_gauss_mask = (sorted_gauss > gauss_threshold)
            large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

            common_indices_mask = torch.isin(large_gradient_indices, large_gauss_indices)
            common_indices = large_gradient_indices[common_indices_mask]
            distinct_indices = large_gradient_indices[~common_indices_mask]

            # Split points with large coordinate gradient and large gaussian values and descale their gaussian
            if len(common_indices) > 0:
                print(f"Number of splitted points: {len(common_indices)}")
                start_index = current_marker + 1
                end_index = current_marker + 1 + len(common_indices)
                persistent_mask[start_index: end_index] = True
                W.data[start_index:end_index, :] = W.data[common_indices, :]
                scale_reduction_factor = 1.6
                W.data[start_index:end_index, 0:2] /= scale_reduction_factor
                W.data[common_indices, 0:2] /= scale_reduction_factor
                current_marker = current_marker + len(common_indices)

            # Clone it points with large coordinate gradient and small gaussian values
            if len(distinct_indices) > 0:

                print(f"Number of cloned points: {len(distinct_indices)}")
                start_index = current_marker + 1
                end_index = current_marker + 1 + len(distinct_indices)
                persistent_mask[start_index: end_index] = True
                W.data[start_index:end_index, :] = W.data[distinct_indices, :]

                # Calculate the movement direction based on the positional gradient
                positional_gradients = W.grad[distinct_indices, 7:9]
                gradient_magnitudes = torch.norm(positional_gradients, dim=1, keepdim=True)
                normalized_gradients = positional_gradients / (gradient_magnitudes + 1e-8)  # Avoid division by zero

                # Define a step size for the movement
                step_size = 0.01

                # Move the cloned Gaussians
                W.data[start_index:end_index, 7:9] += step_size * normalized_gradients

                current_marker = current_marker + len(distinct_indices)

        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if epoch % display_interval == 0:
            num_subplots = 3 if display_loss else 2
            fig_size_width = 18 if display_loss else 12

            # fig, ax = plt.subplots(1, num_subplots, figsize=(fig_size_width, 6))  # Adjust subplot to 1x3

            generated_array = g_tensor_batch.cpu().detach().numpy()

            # ax[0].imshow(g_tensor_batch.cpu().detach().numpy())
            # ax[0].set_title('2D Gaussian Splatting')
            # ax[0].axis('off')

            # ax[1].imshow(target_tensor.cpu().detach().numpy())
            # ax[1].set_title('Ground Truth')
            # ax[1].axis('off')

            # if display_loss:
            #     ax[2].plot(range(epoch + 1), loss_history[:epoch + 1])
            #     ax[2].set_title('Loss vs. Epochs')
            #     ax[2].set_xlabel('Epoch')
            #     ax[2].set_ylabel('Loss')
            #     ax[2].set_xlim(0, num_epochs)  # Set x-axis limits

            # Display the image
            #plt.show(block=False)
            # plt.subplots_adjust(wspace=0.1)  # Adjust this value to your preference
            # plt.pause(0.1)  # Brief pause

            img = Image.fromarray((generated_array * 255).astype(np.uint8))

            # Create filename
            filename = f"{epoch}.jpg"

            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Save the image
            img.save(file_path)

            # fig.savefig(file_path, bbox_inches='tight')

            # plt.clf()  # Clear the current figure
            # plt.close()  # Close the current figure

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, on {len(output)} points")

    final_scale_tensor = scales[0].detach().cpu().numpy()
    final_alpha_tensor = alpha.detach().cpu().numpy()
    gaussian_scales = [ {"id": int(i), "scale_x": float(s[0]), "scale_y": float(s[1]), "alpha": float(a)} for i, (s, a) in enumerate(zip(final_scale_tensor, final_alpha_tensor)) ]
    output_path = os.path.join(directory, "final_gaussian_scales.json")
    with open(output_path, "w") as f:
        json.dump(gaussian_scales, f, indent=4)

    return file_path


# if __name__=='__main__':

#     train()
