import os
import yaml
from dead_leaves import DeadLeavesGenerator, DeadLeavesImage, Disk
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np, cv2
from time import time
from utils import add_log, batch_rgb_to_grayscale
from loss.fid import get_fid
import jax

# @jax.jit(static_argnums=(3,4,5,))
def generate_dead_leaves_image(rmin, rmax, alpha, width, length, key):

    k1,k2,k3 = jax.random.split(key, 3)
    
    for i in range(0, 150000):
        r, x,y = generate_disk(rmin, rmax, alpha, width, length, k1, k2, k3)

    #lax.scan for the application of disks

@jax.jit(static_argnums=(3,4,5,6,7,))
def generate_disk(rmin, rmax, alpha, width, length, k1, k2, k3):
    
    radius = get_radius(rmin, rmax, alpha, k1)

    pos_x, pos_y = get_position(width, length, k2,k3)

    return radius, pos_x, pos_y

    
@jax.jit
def apply_disk_to_image():
    pass

@jax.jit
def get_radius(rmin, rmax, alpha, k1):

    tmp = (rmax ** (1-alpha)) + ((rmin ** (1-alpha)) - (rmax ** (1-alpha))) * jax.random.uniform(k1)
    radius = jnp.array((tmp ** (-1/(alpha - 1))), int)

    return radius

@jax.jit(static_argnums=(0,1,2,3,))
def get_position(width, length, k2, k3):
    pos = [jax.random.randint(key=k2, shape=(), minval=0, maxval=width), jax.random.randint(key=k3, shape=(), minval=0, maxval=length)]

    return pos[0], pos[1]

if __name__=='__main__':
    
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

    