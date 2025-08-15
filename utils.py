import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import os
from datetime import datetime
import jax
import jax.numpy as jnp
from jax import lax
import torch


def save_debug_kernel(kernel_tensor, idx):
    # kernel_tensor: [1, H, W] or [3, H, W], values in [0,1]
    kernel_np = kernel_tensor.detach().cpu().squeeze().numpy()  # Remove batch dim if present
    
    # If RGB (3 channels), permute to HWC format
    if len(kernel_np.shape) == 3 and kernel_np.shape[0] == 3:
        kernel_np = kernel_np.transpose(1, 2, 0)
    
    # Set figure size to match the kernel dimensions (in inches)
    dpi = 100
    height, width = kernel_np.shape[0], kernel_np.shape[1]
    figsize = width / float(dpi), height / float(dpi)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Remove margins
    ax.axis('off')
    
    if len(kernel_np.shape) == 2:  # Grayscale
        ax.imshow(kernel_np, cmap='gray')
    else:  # RGB
        ax.imshow(kernel_np)
    
    timestamp = int(time() * 1000)  # ms resolution
    filename = f"kernel_{idx}_{timestamp}.png"
    filepath = os.path.join(DEBUG_KERNEL_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def sampling_grayscale_histogram(source_image, grayscale=True, visualize=False):
    
    #slow operation
    # source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
    
    if source_image is None:
        raise FileNotFoundError('Invalid path')
    
    #check number of bins !!
    histogram = cv2.calcHist([source_image], [0], None, [256], [0, 256])
    flat_histogram = histogram.flatten()
    
    normalized_hist = flat_histogram / flat_histogram.sum()
    grayscale_value = np.random.choice(256, p=normalized_hist)
    
    if visualize:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(source_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(256), flat_histogram)
        plt.title(f'Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    return grayscale_value

def sampling_distribution_rgb_histogram(source_images, visualize=False):

    hist_r = np.zeros(256)
    hist_g = np.zeros(256)
    hist_b = np.zeros(256)
    
    for img in source_images:
        if img is None:
            raise ValueError("One of the source images is invalid")
            
        b, g, r = cv2.split(img)
        
        hist_r += cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
        hist_g += cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        hist_b += cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    
    norm_r = hist_r / hist_r.sum()
    norm_g = hist_g / hist_g.sum()
    norm_b = hist_b / hist_b.sum()
    
    r_val = np.random.choice(256, p=norm_r)
    g_val = np.random.choice(256, p=norm_g)
    b_val = np.random.choice(256, p=norm_b)
    
    if visualize:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.bar(range(256), hist_r, color='red')
        plt.title('Red Channel Histogram')
        
        plt.subplot(1, 3, 2)
        plt.bar(range(256), hist_g, color='green')
        plt.title('Green Channel Histogram')
        
        plt.subplot(1, 3, 3)
        plt.bar(range(256), hist_b, color='blue')
        plt.title('Blue Channel Histogram')
        
        plt.tight_layout()
        
        plt.figure()
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        color[:, :, 0] = b_val
        color[:, :, 1] = g_val
        color[:, :, 2] = r_val
        plt.imshow(color)
        plt.title(f'Sampled Color (R={r_val}, G={g_val}, B={b_val})')
        plt.axis('off')
        plt.show()
    
    return (r_val, g_val, b_val)


def sampling_rgb_histogram(source_image, visualize=False):

    b, g, r = cv2.split(img)
    
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    
    norm_r = hist_r / hist_r.sum()
    norm_g = hist_g / hist_g.sum()
    norm_b = hist_b / hist_b.sum()
    
    r_val = np.random.choice(256, p=norm_r)
    g_val = np.random.choice(256, p=norm_g)
    b_val = np.random.choice(256, p=norm_b)
    
    if visualize:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.bar(range(256), hist_r, color='red')
        plt.title('Red Channel Histogram')
        
        plt.subplot(1, 3, 2)
        plt.bar(range(256), hist_g, color='green')
        plt.title('Green Channel Histogram')
        
        plt.subplot(1, 3, 3)
        plt.bar(range(256), hist_b, color='blue')
        plt.title('Blue Channel Histogram')
        
        plt.tight_layout()
        
        plt.figure()
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        color[:, :, 0] = b_val
        color[:, :, 1] = g_val
        color[:, :, 2] = r_val
        plt.imshow(color)
        plt.title(f'Sampled Color (R={r_val}, G={g_val}, B={b_val})')
        plt.axis('off')
        plt.show()
    
    return (r_val, g_val, b_val)

def get_distribution_rgb_histogram(source_images, visualize=False):
    hist_r = np.zeros(256)
    hist_g = np.zeros(256)
    hist_b = np.zeros(256)
    
    for img in source_images:
        if img is None:
            raise ValueError("One of the source images is invalid")
            
        b, g, r = cv2.split(img)
        
        hist_r += cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
        hist_g += cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        hist_b += cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    
    if visualize:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.bar(range(256), hist_r, color='red')
        plt.title('Red Channel Histogram')
        
        plt.subplot(1, 3, 2)
        plt.bar(range(256), hist_g, color='green')
        plt.title('Green Channel Histogram')
        
        plt.subplot(1, 3, 3)
        plt.bar(range(256), hist_b, color='blue')
        plt.title('Blue Channel Histogram')
        
        plt.tight_layout()
        plt.show()
    
    return (hist_r, hist_g, hist_b)

def sample_rgb_histogram(hist_r, hist_g, hist_b, num_gaussians=1, device='cpu'):
    hist_r = hist_r / np.sum(hist_r)
    hist_g = hist_g / np.sum(hist_g)
    hist_b = hist_b / np.sum(hist_b)

    r_vals = np.random.choice(256, size=num_gaussians, p=hist_r)
    g_vals = np.random.choice(256, size=num_gaussians, p=hist_g)
    b_vals = np.random.choice(256, size=num_gaussians, p=hist_b)

    colours_np = np.stack([r_vals, g_vals, b_vals], axis=1) / 255.0
    colours = torch.tensor(colours_np, dtype=torch.float32, device=device)
    return colours

def get_rgb_histogram(source_image, visualize=False):
    r,g,b = cv2.split(source_image)
    
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()

    return (hist_r, hist_g, hist_b)

def sampling_uniform_distribution():
    return np.random.uniform(0,255)

def batch_rgb_to_grayscale(rgb_dir_path):
    grayscale_dir_path = os.path.join(rgb_dir_path, 'grayscale')
    os.makedirs(grayscale_dir_path, exist_ok=True)
    
    for file in os.listdir(rgb_dir_path):
        file_path = os.path.join(rgb_dir_path, file)
        
        if not os.path.isfile(file_path):
            continue

        rgb_img = cv2.imread(file_path)

        if rgb_img is None:
            print('Could not read ', file_path)
            continue

        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(grayscale_dir_path, file), gray_img)

        return grayscale_dir_path

def add_log(path, message, should_print=True):
    with open(path, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{timestamp} - {message}")
        if should_print:
            print(f"{timestamp} - {message}")

@jax.jit
def get_radius(rmin, rmax, alpha, k1):

    lax.stop_gradient(k1)

    tmp = (rmax ** (1-alpha)) + ((rmin ** (1-alpha)) - (rmax ** (1-alpha))) * jax.random.uniform(k1)
    # radius = int(tmp ** (-1 / (alpha - 1)))
    radius = jnp.array((tmp ** (-1/(alpha - 1))), int)

    return radius


if __name__ == '__main__':

    path = './source_images/beach'
    source_images = []
    
    for img_name in os.listdir(path):
        if img_name.startswith('.'):
            continue
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        source_images.append(img)
    
    print(sampling_distribution_rgb_histogram(source_images, visualize=True))