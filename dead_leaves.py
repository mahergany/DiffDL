import os 
import numpy as np
import random
import time
import math
from utils import sampling_grayscale_histogram, sampling_rgb_histogram, sampling_uniform_distribution
import cv2

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

    def get_shape_mask(self):
        #calculating radius
        tmp = (self.rmax ** (1-self.alpha)) + ((self.rmin ** (1-self.alpha)) - (self.rmax ** (1-self.alpha))) * np.random.random()
        radius = tmp ** (-1/(self.alpha - 1))

        #plotting shape
        L = np.arange(-radius,radius + 1,dtype = np.int32)
        X, Y = np.meshgrid(L, L) #coordinate space definition based on radius
        shape_1d = np.array((X ** 2 + Y ** 2) <= radius ** 2,dtype = bool) #circle definition

        return (shape_1d, radius)
    
    def update_base_mask(self, image, shape_1d):
        width_shape, length_shape = shape_1d.shape
        pos = [np.random.randint(0, self.width), np.random.randint(0, self.width)] #get random coordinates for the center of the disk
        
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
        image.base_mask[x_min:x_max, y_min:y_max] *= np.logical_not(shape_mask_1d)
        return(x_min,x_max,y_min,y_max,shape_mask_1d)

    def render_shape(self, shape_mask_1d):
        width_shape,length_shape = shape_mask_1d.shape[0],shape_mask_1d.shape[1]

        shape_mask= np.float32(np.repeat(shape_mask_1d[:, :, np.newaxis], 3, axis=2))
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

        #repeat until the entire mask is covered
        while np.any(image.base_mask == 1):

            noOfDisks +=1
            print(noOfDisks)

            #creating a disk mask
            shape_1d, radius = self.get_shape_mask()

            #getting the position and updating the base mask
            x_min,x_max,y_min,y_max,shape_mask_1d = self.update_base_mask(image=image,shape_1d=shape_1d)

            shape_mask,shape_render =self.render_shape(shape_mask_1d)

            image.addDisk(radius=radius, topLeft=[x_min, y_min], bottomRight=[x_max, y_max], shape_mask=shape_mask)
            image.updateDiskVisibility()

            image.resulting_image[x_min:x_max,y_min:y_max,:]*=np.uint8(1-shape_mask)
            image.resulting_image[x_min:x_max,y_min:y_max,:]+=np.uint8(shape_render)
        
        print("dead_leaves stack created with", noOfDisks, "disks")

        self.generated_images.append(image.resulting_image)
            
        return image
    
    def postprocess(self,image=None,blur=True,ds=True):
        if not image:
            image = self.generated_images[-1]
        if blur or ds:
            if blur:
                blur_value = np.random.uniform(1,3)
                image.resulting_image = cv2.GaussianBlur(image.resulting_image,(11,11),sigmaX =  blur_value, borderType = cv2.BORDER_DEFAULT)
            if ds:
                image.resulting_image = cv2.resize(image.resulting_image,(0,0), fx = 1/2.,fy = 1/2. , interpolation = cv2.INTER_AREA)
            image.resulting_image = np.uint8(image.resulting_image)


class DeadLeavesImage:
    def __init__(self, width, length):
        self.width = width
        self.length = length

        self.base_mask = np.ones((self.width, self.length), dtype=int)
        self.resulting_image = np.ones((width,width,3), dtype = np.uint8)

        self.disks = []
        self.diskCount = 0
        self.disk_visibility = None

    def addDisk(self, radius, topLeft, bottomRight, shape_mask):
        self.diskCount+=1
        disk = Disk(id=self.diskCount, radius=radius, topLeft=topLeft, bottomRight=bottomRight, shape_mask=shape_mask)
        self.disks.append(disk)

    def updateDiskVisibility(self):
        pass
        
    def get_shape_mask(self):
        tmp = (self.rmax ** (1-self.alpha)) + ((self.rmin ** (1-self.alpha)) - (self.rmax ** (1-self.alpha))) * np.random.random()
        radius = tmp ** (-1/(self.alpha - 1))

        L = np.arange(-radius,radius + 1,dtype = np.int32)
        X, Y = np.meshgrid(L, L)
        shape_1d = np.array((X ** 2 + Y ** 2) <= radius ** 2,dtype = bool)

        return (shape_1d, radius)


class Disk:
    def __init__(self, id, topLeft, bottomRight, radius, shape_mask):
        self.id = id

        #coordinates
        self.topLeft = topLeft
        self.bottomRight = bottomRight

        self.x_min = topLeft[0]
        self.x_max = bottomRight[0]
        self.y_min = topLeft[1]
        self.y_max = bottomRight[1]

        self.radius = radius
        self.mask = shape_mask

if __name__ == '__main__':
    generator = DeadLeavesGenerator(rmin=1, rmax=1000, alpha=3, width=500, length=500, grayscale=True, color_path='C:/Users/mahee/Desktop/dead leaves project/DiffDL/source_images/tree.jpg')
    generator.get_shape_mask()