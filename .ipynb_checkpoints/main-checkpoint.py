#for generation of dead leaves images

import os
import yaml
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np
from time import time

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

    rmin = config['params']['rmin']
    rmax = config['params']['rmax']
    alpha = config['params']['alpha']
    grayscale = config['color']['grayscale']
    uniform_sampling = config['color']['uniform_sampling']

    no_of_images = config['settings']['no_of_images']
    width = config['settings']['width']
    length = config['settings']['length']
    category = config['settings']['category']
    source_directory = config['settings']['source_directory']
    output_directory = config['settings']['output_directory']
    postprocess = config['settings']['postprocess']
    enableJax = config['settings']['enableJax']

#setting up generation folder
output_path = output_directory
os.makedirs(output_directory, exist_ok=True)
files = os.listdir(output_path)

indices = []
for f in files:
    try:
        indices.append(int(f.split('_')[-1].split('.')[0]))
    except (ValueError, IndexError):
        continue
index = max(indices) + 1 if indices else 0

if enableJax:
    from dead_leaves_jax import DeadLeavesGenerator
else:
    from dead_leaves import DeadLeavesGenerator

#init a DL generation object that will generate images with set parameters
object = DeadLeavesGenerator(source_dir_path=source_directory, rmin=rmin, rmax=rmax, alpha=alpha, width=width, length=length, grayscale=grayscale, uniform_sampling=uniform_sampling, enableJax=enableJax)
images = []

# for i in range(0, no_of_images):
for i in range(0, no_of_images):
    t0 = time()
    #generation of a dead leaves image based on the parameters
    image = object.generate()
    print('Time taken:', time()-t0)
    if postprocess:
        object.postprocess(image)

    skio.imsave(f'{output_directory}/generated_{category}_{index}.png', np.uint8(np.clip(255*rgb2gray(image.resulting_image),0,255)))

    images.append(image)
    index+=1

    image.visualizeDiskVisibility()

