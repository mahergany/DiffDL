import os
import yaml
from generation import generate_dead_leaves_image
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np, cv2
from time import time
from utils import add_log, batch_rgb_to_grayscale
from loss.fid import get_fid
import jax


def train():
    with open('configs/train_config.yaml') as file:
        config = yaml.safe_load(file)

        source_dir = config['settings']['source_directory']
        output_dir = config['settings']['output_directory']
        no_of_images = config['settings']['no_of_images']
        category = config['settings']['category']
        width = config['settings']['width']
        length = config['settings']['length']

        #params
        rmin = config['params']['rmin']
        rmax = config['params']['rmin']
        alpha = config['params']['alpha']

        num_disks = config['params']['num_disks']

        grayscale=config['color']['grayscale']

        iterations = config['training']['iterations']

    #check source directory
    source_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not source_images or len(source_images) == 0:
        raise ValueError("No images in source directory")
    
    #in the event that images are not grayscale despite the setting
    if grayscale:
        if len(cv2.imread(os.path.join(source_dir, np.random.choice(source_images))).shape) == 3:
            source_dir = batch_rgb_to_grayscale(source_dir)
            
    #training output directory setup
    runs = os.listdir('runs')
    runs.sort()
    run_no = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    output_dir += f'/run_{run_no}'
    os.makedirs(output_dir, exist_ok=True)

    #logging file
    log_path = os.path.join(output_dir, "training_log.txt")
    with open(log_path, 'w'):
        pass

    for iter in range(1, iterations+1):

        add_log(log_path, f"Iteration {iter}")

        iteration_dir = os.path.join(output_dir, str(iter))
        generation_dir = os.path.join(iteration_dir, 'generated images')

        os.makedirs(iteration_dir,exist_ok=True)
        os.makedirs(generation_dir, exist_ok=True)

        generated_images = []
        
        for i in range(no_of_images):
            add_log(log_path, f"Generating Dead Leaves image {i+1}/{no_of_images}")

            t0 = time()

            key = jax.random.key(int(time.time() * 1000))


            image = generate_dead_leaves_image(rmin, rmax, alpha, width, length, source_images,num_disks, key)

            generated_images.append(image)

            skio.imsave(f'{generation_dir}/generated_{category}_{i}.jpg', np.uint8(np.clip(255*rgb2gray(image),0,255)))

            add_log(log_path, f'Time taken:, {time()-t0}')

        add_log(log_path, "Computing FID Score")

        fid_score = get_fid(source_images_path=source_dir, generated_images_path=generation_dir)

        add_log(log_path, "FID Score: ", fid_score)

        #TODO: update parameters atp through jax automatic differentiation
        grad_fid = jax.grad(get_fid)


if __name__ == '__main__':
    # pass
    train()