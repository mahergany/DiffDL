import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml

def sampling_color_histogram(source_image, visualize=False):
    
    #slow operation
    # source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE
    
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

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    value = sampling_color_histogram(config['color']['color_path'], visualize=True)
    print(value)