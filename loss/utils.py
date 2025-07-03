#Reference taken from: https://github.com/matthias-wright/jax-fid

import os
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import cv2
from tqdm import tqdm
import tempfile
import requests
import matplotlib.pyplot as plt

def compute_data_distribution(images_path, apply_fn, fn_params, isGenerated=True, batch_size=1, grayscale=True):
    ''' returns mean and sdev of the assumed normal distribution '''
    images = []

    #go through all image files and resize
    for file in tqdm(os.listdir(images_path), f"Loading {'generated' if isGenerated else 'source'} images"):

        #check for a precomputed file
        if os.path.join(images_path, file).endswith('.npz'):
            stats=np.load(os.path.join(images_path, file))
            mu,sigma = stats['mu'],stats['sigma']
            return mu, sigma

        if grayscale:
            img = cv2.imread(os.path.join(images_path, file))

        if img is None:
            continue
        
        #TODO: check if this might slow things down
        #convert to rgb for global processing
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #normalizing
        img = np.array(img) / 255

        images.append(img)

    activations = []

    #loop through batches and pass through model
    num_batches = int(len(images) // batch_size)
    for i in tqdm(range(num_batches)):
        x = images[i*batch_size: i*batch_size + batch_size]
        x = np.asarray(x)
        x = 2*x+1
        pred = apply_fn(fn_params, jax.lax.stop_gradient(x)) #stopping gradient flow as it is not required here
        # print(pred.shape)
        activations.append(pred.squeeze(axis=1).squeeze(axis=1)) #removing spatial dimensions
        # print(activations[-1].shape)

    activations = jnp.concatenate(activations, axis=0)

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savez(os.path.join(images_path, 'stats'), mu=mu, sigma=sigma)

    return mu, sigma

def download(url, ckpt_dir=None):
    name = url[url.rfind('/') + 1 : url.rfind('?')]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'jax_fid')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file

def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]