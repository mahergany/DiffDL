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
# from TwoDGS.generation import train
from TwoDGS.generation_ffl import train

from dead_leaves import DeadLeavesGenerator

'''
generate 2dgs image
save x and y scale values
map the distribution 1 to 1 
'''


#directory init
os.makedirs('./TwoDGS/runs', exist_ok=True)
runs = os.listdir('./TwoDGS/runs')

if runs:
    runs = sorted([int(r) for r in runs])
    run_dir = os.path.join('./TwoDGS/runs',str(int(runs[-1]) + 1))
else:
    run_dir =  os.path.join('./TwoDGS/runs','1')

os.makedirs(run_dir)

source_image = train(run_dir)

#got json with x and y scales of the final 2dgs generated image

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

    alpha = config['params']['alpha']
    grayscale = config['color']['grayscale']
    uniform_sampling = config['color']['uniform_sampling']

    no_of_images = config['settings']['no_of_images']
    width = config['settings']['width']
    length = config['settings']['length']
    category = config['settings']['category']
    output_directory = run_dir
    postprocess = config['settings']['postprocess']
    enableJax = config['settings']['enableJax']

# !!!!!!!!!!
rmin = None
rmax = None
# !!!!!!!!!!


#generated DL will be based on one 2dgs image
# object = DeadLeavesGenerator(source_dir_path=False, source_image=source_image, rmin=rmin, rmax=rmax, alpha=alpha, width=width, length=length, grayscale=grayscale, uniform_sampling=uniform_sampling, enableJax=enableJax)

# image = object.generate()
# object.postprocess(image)

# skio.imsave(f'{output_directory}/generated_dead_leaves.png', np.uint8(np.clip(255*rgb2gray(image.resulting_image),0,255)))
