import os
import yaml
# from dead_leaves import DeadLeavesGenerator, DeadLeavesImage, Disk
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np, cv2
from time import time
# from utils import add_log, batch_rgb_to_grayscale, sampling_grayscale_histogram
from loss.fid import get_fid
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax

def generate_dead_leaves_image(rmin, rmax, alpha, width, length, source_images_path, num_disks, key):

    print("Generating keys")
    keys = jax.random.split(key, num_disks)

    print("Generating colors")
    colors = get_colors(num_disks, source_images_path)
    
    print("Precomputing disks")
    disk_masks = precompute_disks(rmax)
    
    print("Generating disk_params")
    # disk_params = jax.vmap(lambda c, k: generate_disk(rmin, rmax, alpha, width, length, dict_instance, c, k))(colors,keys)
    disk_params = jax.vmap(lambda c, k: generate_disk(rmin, rmax, alpha, width, length, disk_masks, c, k))(colors,keys)

    # shape_1d = dict_instance[disk_params[0]]
    # disk_params.append(shape_1d)
    # disk_params = generate_disk_vmapped(colors, keys)

    image = jnp.ones((width, length, 3), dtype=jnp.uint8)
    base_mask = jnp.ones((width, length), dtype=int)
    carry = (image, base_mask)

    print("Applying disks")
    (final_image, final_mask), _ = jax.lax.scan(apply_disk_to_image, carry, disk_params)

    return final_image

@partial(jit, static_argnums=(3,4,))
def generate_disk(rmin, rmax, alpha, width, length, disk_masks, color, keys):
    keys = lax.stop_gradient(keys)
    color = lax.stop_gradient(color)
    disk_masks = lax.stop_gradient(disk_masks)

    k1, k2, k3 = jax.random.split(keys, 3)

    radius = get_radius(rmin, rmax, alpha, k1)
    # L = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    # X, Y = jnp.meshgrid(L, L)
    # shape_1d = (X ** 2 + Y ** 2) <= radius ** 2
    shape_1d = disk_masks[radius]

    pos_x, pos_y = get_position(width, length, k2,k3)

    return radius, pos_x, pos_y, color, shape_1d


@jit
def apply_disk_to_image(carry, disk_param):
    image, base_mask = carry

    # r, x,y, color = disk_param
    r, x,y, color, shape_1d = disk_param

    width, length, _ = image.shape

    width_shape, length_shape = shape_1d.shape

    x_min = jnp.maximum(0, x - width_shape//2)
    x_max = jnp.minimum(width, x + width_shape//2 + 1)
    y_min = jnp.maximum(0, y - length_shape//2)
    y_max = jnp.minimum(width, y + length_shape//2 + 1)

    slice_width = x_max - x_min
    slice_height = y_max - y_min

    shape_x_start = jnp.maximum(0, width_shape // 2 - x)
    shape_y_start = jnp.maximum(0, length_shape // 2 - y)

    shape_x_end = shape_x_start + slice_width
    shape_y_end = shape_y_start + slice_height

    shape_mask_1d = lax.dynamic_slice(base_mask, (x_min.astype(int), y_min.astype(int)), (slice_width, slice_height))

    shape_1d_slice = shape_1d[shape_x_start:shape_x_end, shape_y_start:shape_y_end]
    shape_mask_1d = shape_mask_1d * shape_1d_slice
    new_base_mask_patch = base_mask[x_min:x_max, y_min:y_max] * jnp.logical_not(shape_mask_1d)
    base_mask = lax.dynamic_update_slice(base_mask, new_base_mask_patch, (x_min, y_min))
    shape_mask_rgb = jnp.float32(jnp.repeat(shape_mask_1d[:, :, jnp.newaxis], 3, axis=2))
    shape_render = color * shape_mask_rgb

    image_patch = lax.dynamic_slice(image, (x_min, y_min, 0), (slice_width, slice_height, 3))
    new_image_patch = image_patch * (1 - shape_mask_rgb) + shape_render
    image = lax.dynamic_update_slice(image, new_image_patch, (x_min, y_min, 0))

    return (image, base_mask), None

@jax.jit
def get_radius(rmin, rmax, alpha, k1):

    lax.stop_gradient(k1)

    tmp = (rmax ** (1-alpha)) + ((rmin ** (1-alpha)) - (rmax ** (1-alpha))) * jax.random.uniform(k1)
    # radius = int(tmp ** (-1 / (alpha - 1)))
    radius = jnp.array((tmp ** (-1/(alpha - 1))), int)

    return radius

# @jax.jit(static_argnums=(0,1,2,3,))
@partial(jit, static_argnums=(0,1,))
def get_position(width, length, k2, k3):
    lax.stop_gradient(k2)
    lax.stop_gradient(k3)
    pos = [jax.random.randint(key=k2, shape=(), minval=0, maxval=width), jax.random.randint(key=k3, shape=(), minval=0, maxval=length)]
    return pos[0], pos[1]

def get_colors(num_disks, source_images_path):
    i = np.random.randint(0, len(source_images_path))
    color_image = cv2.imread(source_images_path[i], cv2.IMREAD_GRAYSCALE)

    histogram = cv2.calcHist([color_image], [0], None, [256], [0, 256])
    flat_histogram = histogram.flatten()
    normalized_hist = flat_histogram / flat_histogram.sum()
    
    #sampling num_disks times
    colors = np.random.choice(256, size=num_disks, p=normalized_hist)

    return jnp.array(colors, dtype=jnp.float32)

def precompute_disks(rmax):
    disks = []
    max_size = 2 * rmax + 1
    for r in range(rmax + 1):
        L = np.arange(-r, r + 1)
        X, Y = np.meshgrid(L, L)
        mask = ((X ** 2 + Y ** 2) <= r ** 2).astype(np.bool_)
        # Pad mask to max_size with False to have uniform shape
        padded_mask = np.zeros((max_size, max_size), dtype=np.bool_)
        offset = rmax - r
        padded_mask[offset:offset+mask.shape[0], offset:offset+mask.shape[1]] = mask
        disks.append(padded_mask)
    return jnp.array(disks)  # Shape: (rmax+1, max_size, max_size)

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


    # dict_instance = jnp.load('npy/dict_jnp2.npy', allow_pickle=True)
    # dict_instance = jnp.array(np.load('npy/dict.npy', allow_pickle=True))
    # print(type(dict_instance))
    # print(dict_instance.keys()) 
    # exit()

    # print("dict_instance shape:", dict_instance.shape)
    # exit()
    source_images_path = [os.path.join(source_directory, file) for file in os.listdir(source_directory) if file.lower().endswith(('.png','.jpg','.jpeg'))]
    key = jax.random.key(int(time() * 1000))

    image = generate_dead_leaves_image(rmin, rmax, alpha,width, length, source_images_path, 150000, key)
    skio.imsave(f'this_better_be_fast.png', np.uint8(np.clip(255*rgb2gray(image),0,255)))
