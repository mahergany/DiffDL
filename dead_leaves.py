import os 
import numpy as np
import random
import time
import math
from utils import sampling_color_histogram
import cv2

class DeadLeavesGenerator:
    def __init__(self, rmin, rmax, alpha, width, length, grayscale, color_path):
        self.rmin = rmin
        self.rmax = rmax
        self.alpha = alpha
        self.width = width
        self.length = width # for now keeping it width by width 
        self.radius = 0

        self.gray_scale = grayscale
        self.color_path = color_path
        self.color_image = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE) #to speed up process

        self.base_mask = np.ones((self.width, self.length), dtype=int)
        self.resulting_image = np.ones((width,width,3), dtype = np.uint8)

        self.generated_images = []

    def get_shape_mask(self):
        tmp = (self.rmax ** (1-self.alpha)) + ((self.rmin ** (1-self.alpha)) - (self.rmax ** (1-self.alpha))) * np.random.random()
        radius = tmp ** (-1/(self.alpha - 1))

        L = np.arange(-radius,radius + 1,dtype = np.int32)
        X, Y = np.meshgrid(L, L)
        shape_1d = np.array((X ** 2 + Y ** 2) <= radius ** 2,dtype = bool)

        return (shape_1d, radius)
    
    def update_base_mask(self, shape_1d):
        width_shape, length_shape = shape_1d.shape
        pos = [np.random.randint(0, self.width), np.random.randint(0, self.width)]
        
        x_min = max(0, pos[0] - width_shape//2)
        x_max = min(self.width, pos[0] + width_shape//2 + 1)
        y_min = max(0, pos[1] - length_shape//2)
        y_max = min(self.width, pos[1] + length_shape//2 + 1)
        
        shape_x_start = max(0, width_shape//2 - pos[0])
        shape_x_end = min(width_shape, width_shape//2 + (self.width - pos[0]))
        shape_y_start = max(0, length_shape//2 - pos[1])
        shape_y_end = min(length_shape, length_shape//2 + (self.width - pos[1]))
        
        target_width = x_max - x_min
        target_height = y_max - y_min
        shape_crop_width = shape_x_end - shape_x_start
        shape_crop_height = shape_y_end - shape_y_start
        
        shape_x_end = shape_x_start + min(target_width, shape_crop_width)
        shape_y_end = shape_y_start + min(target_height, shape_crop_height)
        x_max = x_min + min(target_width, shape_crop_width)
        y_max = y_min + min(target_height, shape_crop_height)
        
        shape_mask_1d = self.base_mask[x_min:x_max, y_min:y_max].copy()
        shape_cropped = shape_1d[shape_x_start:shape_x_end, shape_y_start:shape_y_end]
    
        if shape_mask_1d.shape != shape_cropped.shape:
            print(f"mismatch: {shape_mask_1d.shape} {shape_cropped.shape}")
            min_shape = (min(shape_mask_1d.shape[0], shape_cropped.shape[0]),
                        min(shape_mask_1d.shape[1], shape_cropped.shape[1]))
            shape_mask_1d = shape_mask_1d[:min_shape[0],:min_shape[1]]
            shape_cropped = shape_cropped[:min_shape[0],:min_shape[1]]
        
        shape_mask_1d &= shape_cropped
        self.base_mask[x_min:x_max, y_min:y_max] &= ~shape_mask_1d
        
        return x_min, x_max, y_min, y_max, shape_mask_1d

    def render_shape(self, shape_mask_1d, radius):
        width_shape,length_shape = shape_mask_1d.shape[0],shape_mask_1d.shape[1]

        shape_mask= np.float32(np.repeat(shape_mask_1d[:, :, np.newaxis], 3, axis=2))
        shape_render = shape_mask.copy()

        color = sampling_color_histogram(self.color_image)
        shape_render = color*shape_render

        return(shape_mask,shape_render)
        
    def generate(self):

        # image = DeadLeavesImage(self.width, self.length)

        disks = 0

        #repeat until the entire mask is covered
        while np.any(self.base_mask == 1):

            print(disks)

            disks +=1
            #creating the shape mask
            shape_1d, radius = self.get_shape_mask()

            #getting the position and updating the base mask
            x_min,x_max,y_min,y_max,shape_mask_1d = self.update_base_mask(shape_1d)

            shape_mask,shape_render =self.render_shape(shape_mask_1d, radius)

            self.resulting_image[x_min:x_max,y_min:y_max,:]*=np.uint8(1-shape_mask)
            self.resulting_image[x_min:x_max,y_min:y_max,:]+=np.uint8(shape_render)

        print("dead_leaves stack created with", disks, "disks")
            
        return self.resulting_image
    
    def postprocess(self,blur=True,ds=True):
        if blur or ds:
            if blur:
                blur_value = np.random.uniform(1,3)
                self.resulting_image = cv2.GaussianBlur(self.resulting_image,(11,11),sigmaX =  blur_value, borderType = cv2.BORDER_DEFAULT)
            if ds:
                self.resulting_image = cv2.resize(self.resulting_image,(0,0), fx = 1/2.,fy = 1/2. , interpolation = cv2.INTER_AREA)
            self.resulting_image = np.uint8(self.resulting_image)

# class DeadLeavesImage:
#     def __init__(self, width, length):
#         self.width = width
#         self.length = length

#         self.base_mask = np.ones((self.width, self.length), dtype=int)
#         self.resulting_image = np.ones((width,width,3), dtype = np.uint8)

#         self.disks = []


# class Disk:
#     def __init__(self, id, cx, cy, radius):
#         self.id = id
#         #center coordinates
#         self.cx = cx
#         self.cy = cy

#         self.radius = radius

#         x, y = np.meshgrid(np.arange(cx - radius, cx + radius + 1), 
#                            np.arange(cy - radius, cy + radius + 1))
        
#         disk = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2 #contains points within the disk on the meshgrid

#         #pixel wise visibility dictionary
#         self.visibility_mask = {(xi,yi): False
#                                     for xi, yi in zip(x[disk], y[disk])
#                                 }
        