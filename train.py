#for training dead leaves images to look a certain way

import os
import yaml
from dead_leaves import DeadLeavesGenerator, DeadLeavesImage, Disk
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np
from time import time

with open('configs/train_config.yaml') as file:
    config = yaml.safe_load(file)

    source_dir = config['settings']['source_directory']
    output_dir = config['settings']['output_directory']

    iterations = config['training']['iterations']

real_images = os.listdir(source_dir)

#training output directory setup
runs = os.listdir('runs')
runs.sort()
run_no = int(runs[-1].split('_')[-1].split('.')[0] + 1) if runs else 0
output_dir += f'/{run_no}'
os.makedirs(output_dir, exist_ok=True)

for i in range(iterations):
    os.makedirs('')
    pass