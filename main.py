import os
import yaml
from dead_leaves import DeadLeavesGenerator
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np
from time import time

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

    rmin = config['params']['rmin']
    rmax = config['params']['rmax']
    alpha = config['params']['alpha']
    grayscale = config['color']['grayscale']
    color_path = config['color']['color_path']
    uniform_sampling = config['color']['uniform_sampling']

    no_of_images = config['settings']['no_of_images']
    width = config['settings']['width']
    length = config['settings']['length']
    category = config['settings']['category']
    source_directory = config['settings']['source_directory']
    output_directory = config['settings']['output_directory']

#setting up generation folder
output_path = output_directory + f'/{category}/'
os.makedirs(output_directory, exist_ok=True)
files = os.listdir(output_path)
files.sort()
index = (int(files[-1].split('_')[-1].split('.')[0]) + 1) if files else 0


#init a DL generation object that will generate images with set parameters
object = DeadLeavesGenerator(rmin, rmax, alpha, width, length, grayscale, color_path, uniform_sampling)
images = []

# for i in range(0, no_of_images):
for i in range(0, no_of_images):
    t0 = time()
    #generation of a dead leaves image based on the parameters
    image = object.generate()
    object.postprocess(image)

    skio.imsave(f'{output_directory}/generated_{category}_{index}.png', np.uint8(np.clip(255*rgb2gray(image.resulting_image),0,255)))
    print('Time taken:', time()-t0)

    images.append(image)
    index+=1

    print(image.disks)

