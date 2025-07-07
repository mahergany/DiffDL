import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import os
from datetime import datetime
import jax
import jax.numpy as jnp

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

def sampling_rgb_histogram():
    pass

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



if __name__ == '__main__':

    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    value = sampling_grayscale_histogram('C:/Users/mahee/Desktop/dead leaves project/DiffDL/source_images/forest/00000015.jpg', visualize=True)
    # print(value)

    # batch_rgb_to_grayscale(r'C:/Users/mahee/Desktop/dead leaves project/DiffDL/source_images/forest')