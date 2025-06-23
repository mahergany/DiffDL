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
    no_of_images = config['settings']['no_of_images']
    width = config['settings']['width']
    length = config['settings']['length']

#init a DL generation object that will generate images with set parameters
object = DeadLeavesGenerator(rmin, rmax, alpha, width, length, grayscale, color_path)
images = []

# for i in range(0, no_of_images):
for i in range(0, 1):
    t0 = time()
    #generation of a dead leaves image based on the parameters
    object.generate()
    object.postprocess()

    skio.imsave('generated_tree_6.png', np.uint8(np.clip(255*rgb2gray(object.resulting_image),0,255)))
    print('Time taken:', time()-t0)

    images.append(object.resulting_image)
    pass

