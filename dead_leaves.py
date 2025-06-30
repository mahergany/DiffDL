import os 
import numpy as np
import jax
import jax.numpy as jnp
import random
import time
import math
from utils import sampling_grayscale_histogram, sampling_rgb_histogram, sampling_uniform_distribution
import cv2
import matplotlib.pyplot as plt
import gc
from jax.scipy import ndimage
from jax.image import resize
import json

dict_instance = np.load('npy/dict.npy', allow_pickle=True)

class DeadLeavesGenerator:
    def __init__(self, rmin, rmax, alpha, width, length, grayscale, color_path, uniform_sampling):
        self.rmin = rmin
        self.rmax = rmax
        self.alpha = alpha
        self.width = width
        self.length = width # for now keeping it width by width

        self.grayscale = grayscale
        self.color_path = color_path
        self.uniform_sampling = uniform_sampling
        self.color_image = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE) #to speed up process

        self.generated_images = []

        self.key = jax.random.key(int(time.time() * 1000))

        self.subkey_buffer = []
        self.subkey_buffer_size = 50000
        
        self.key_index = 0
        self.batch_subkey_generation()


    def batch_subkey_generation(self):
        print("Generating subkeys")
        self.subkey_buffer = []
        for i in range(self.subkey_buffer_size):
            self.key, pos_x_key, pos_y_key, radius_key = jax.random.split(self.key, 4)
            self.subkey_buffer.append([self.key, pos_x_key, pos_y_key, radius_key])
        print(f"Done generating {self.subkey_buffer_size} subkeys")

    def get_next_subkeys(self):
        self.key_index+=1
        if self.key_index >= self.subkey_buffer_size:
            self.batch_subkey_generation()
            self.key_index=0
        return self.subkey_buffer[self.key_index]

    def get_shape_mask(self, subkey):
        # self.key, subkey = jax.random.split(self.key)
        #calculating radius
        tmp = (self.rmax ** (1-self.alpha)) + ((self.rmin ** (1-self.alpha)) - (self.rmax ** (1-self.alpha))) * jax.random.uniform(subkey)
        radius = int(tmp ** (-1/(self.alpha - 1)))

        #plotting shape
        # L = jnp.arange(-radius,radius + 1,dtype = jnp.int32)
        # X, Y = jnp.meshgrid(L, L) #coordinate space definition based on radius
        # shape_1d = jnp.array((X ** 2 + Y ** 2) <= radius ** 2,dtype = bool) #circle definition

        shape_1d = dict_instance[()][str(radius)]

        return (shape_1d, radius)
    
    def update_base_mask(self, image, shape_1d, pos_x_key, pos_y_key):
        # self.key, pos_x_key, pos_y_key = jax.random.split(self.key,3)
        width_shape, length_shape = shape_1d.shape
        # pos = [jax.random.randint(key=self.key, shape=(), minval=0, maxval=self.width), jax.random.randint(key=self.key, shape=(), minval=0, maxval=self.width)] #get random coordinates for the center of the disk
        pos = [jax.random.randint(key=pos_x_key, shape=(), minval=0, maxval=self.width), jax.random.randint(key=pos_y_key, shape=(), minval=0, maxval=self.width)] #get random coordinates for the center of the disk
        
        #mapping into top left and bottom right coordinates
        x_min = max(0, pos[0] - width_shape//2)
        x_max = min(self.width, pos[0] + width_shape//2 + 1)
        y_min = max(0, pos[1] - length_shape//2)
        y_max = min(self.width, pos[1] + length_shape//2 + 1)
        
        shape_x_start = max(0,width_shape//2-pos[0])
        shape_x_end = min(width_shape,self.width+width_shape//2-pos[0])
        shape_y_start = max(0,length_shape//2-pos[1])
        shape_y_end = min(length_shape,self.width+length_shape//2-pos[1])

        # print(x_min, x_max, shape_x_start, shape_x_end)
        
        target_width = x_max - x_min
        target_height = y_max - y_min
        shape_crop_width = shape_x_end - shape_x_start
        shape_crop_height = shape_y_end - shape_y_start
        
        #updating the max parameters to ensure uniformity
        shape_x_end = shape_x_start + min(target_width, shape_crop_width)
        shape_y_end = shape_y_start + min(target_height, shape_crop_height)
        x_max = x_min + min(target_width, shape_crop_width)
        y_max = y_min + min(target_height, shape_crop_height)
        
        shape_mask_1d = image.base_mask[x_min:x_max, y_min:y_max].copy()
        shape_1d = shape_1d[shape_x_start:shape_x_end, shape_y_start:shape_y_end]

        shape_mask_1d *= shape_1d
        # image.base_mask[x_min:x_max, y_min:y_max] *= jnp.logical_not(shape_mask_1d)
        # image.base_mask = image.base_mask.at[x_min:x_max, y_min:y_max].multiply(jnp.logical_not(shape_mask_1d))
        image.base_mask = image.base_mask.at[x_min:x_max, y_min:y_max].set(image.base_mask[x_min:x_max, y_min:y_max] * jnp.logical_not(shape_mask_1d))
        return(x_min,x_max,y_min,y_max,shape_mask_1d)

    def render_shape(self, shape_mask_1d):
        width_shape,length_shape = shape_mask_1d.shape[0],shape_mask_1d.shape[1]

        shape_mask= jnp.float32(jnp.repeat(shape_mask_1d[:, :, jnp.newaxis], 3, axis=2))
        shape_render = shape_mask.copy()

        if self.grayscale:
            if self.uniform_sampling:
                color = sampling_uniform_distribution()
            else:
                color = sampling_grayscale_histogram(self.color_image)
        else:
            sampling_rgb_histogram()
            exit()

        shape_render = color*shape_render

        return(shape_mask,shape_render)
        
    def generate(self):

        image = DeadLeavesImage(self.width, self.length)

        noOfDisks = 0

        #repeat until the entire mask is covered (slow operation)
        while jnp.any(image.base_mask == 1):
        
        #(faster operation)
        # for i in range(0,1000):

            noOfDisks +=1
            print(f"\rDisk no. {noOfDisks}", end="", flush=True)

            subkeys = self.get_next_subkeys()

            #creating a disk mask
            shape_1d, radius = self.get_shape_mask(subkey=subkeys[3])

            #getting the position and updating the base mask
            x_min,x_max,y_min,y_max,shape_mask_1d = self.update_base_mask(image=image,shape_1d=shape_1d, pos_x_key=subkeys[1], pos_y_key=subkeys[2])

            shape_mask,shape_render =self.render_shape(shape_mask_1d)

            image.addDisk(radius=radius, topLeft=[x_min, y_min], bottomRight=[x_max, y_max], shape_mask=shape_mask)

            image.resulting_image = image.resulting_image.at[x_min:x_max,y_min:y_max,:].multiply(jnp.uint8(1-shape_mask))
            image.resulting_image = image.resulting_image.at[x_min:x_max,y_min:y_max,:].add(jnp.uint8(shape_render))

            #cleanup
            if noOfDisks % 1000 == 0:
                gc.collect()
                jax.clear_caches()

        print()
        
        print("dead_leaves stack created with", noOfDisks, "disks")

        self.generated_images.append(image.resulting_image)
            
        return image
    
    def postprocess(self,image,blur=True,ds=True):
        #TODO: implement this through self defined jax functions to remove np dependency
        #currently switching to np for cv2 implementation

        resulting_image = np.array(image.resulting_image)
        if blur or ds:
            if blur:
                blur_value = np.random.uniform(1,3)
                resulting_image = cv2.GaussianBlur(resulting_image,(11,11),sigmaX =  blur_value, borderType = cv2.BORDER_DEFAULT)
            if ds:
                resulting_image = cv2.resize(resulting_image,(0,0), fx = 1/2.,fy = 1/2. , interpolation = cv2.INTER_AREA)
            resulting_image = np.uint8(resulting_image)
        
        image.resulting_image = jnp.array(resulting_image)

class DeadLeavesImage:
    def __init__(self, width, length):
        self.width = width
        self.length = length

        self.base_mask = jnp.ones((self.width, self.length), dtype=int)
        self.resulting_image = jnp.ones((width,width,3), dtype = jnp.uint8)

        self.disks = []
        self.diskCount = 0
        self.disk_visibility = jnp.zeros((self.width, self.length), dtype=int)

    def addDisk(self, radius, topLeft, bottomRight, shape_mask):
        self.diskCount+=1
        disk = Disk(id=self.diskCount, radius=radius, topLeft=topLeft, bottomRight=bottomRight, shape_mask=shape_mask)
        # print(disk.radius, disk.topLeft, disk.bottomRight)
        self.disks.append(disk)

        #update visibility matrix

        if shape_mask.ndim == 3:
            shape_mask = shape_mask.any(axis=-1)

        disk_id_expanded = jnp.full(shape_mask.shape[:2], disk.id) #the shape needs to be broadcast while using jnp.where so for consistency

        current_visibility = self.disk_visibility[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]
        updated_visibility = jnp.where(shape_mask, disk_id_expanded, current_visibility)
        self.disk_visibility = self.disk_visibility.at[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]].set(updated_visibility)

    def visualizeDiskVisibility(self):
        plt.imshow(self.disk_visibility)
        plt.show()
        

class Disk:
    def __init__(self, id, topLeft, bottomRight, radius, shape_mask):
        self.id = id #starts from 1

        #coordinates
        self.topLeft = topLeft
        self.bottomRight = bottomRight

        # self.x_min = topLeft[0]
        # self.x_max = bottomRight[0]
        # self.y_min = topLeft[1]
        # self.y_max = bottomRight[1]

        self.radius = radius
        # self.mask = shape_mask

if __name__ == '__main__':
    generator = DeadLeavesGenerator(rmin=1, rmax=1000, alpha=3, width=500, length=500, grayscale=True, color_path='C:/Users/mahee/Desktop/dead leaves project/DiffDL/source_images/tree.jpg', uniform_sampling=False)
    # for i in range(10):
    #     print(generator.get_shape_mask())
    generator.batch_subkey_generation()
    # with open('subkeys.txt', 'w') as file:
    #     for subkeys in generator.subkey_buffer:
    #         file.write((' '.join(map(str, subkeys))) + '\n')
