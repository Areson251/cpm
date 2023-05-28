from ctypes.wintypes import SHORT
from math import sqrt
import cv2
import numpy as np
import time, datetime
from  math import sin, cos, radians, ceil   
import random
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from algorithm import *
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import argparse
import imutils


class ImageData:
    def __init__(self):
        self.IMAGE_1_PATH = None
        self.IMAGE_2_PATH = None
        self.MAP_SLICE = None
        self.image1 = None
        self.image2 = None
        self.coords = ()
        self.cropped_image = None

    def start_preprocessing(self, path1, path2, slice=None):
        self.IMAGE_1_PATH = path1
        self.IMAGE_2_PATH = path2
        self.MAP_SLICE = slice

        self.image1 = self.read_img(self.IMAGE_1_PATH)
        self.image2 = self.read_img(self.IMAGE_2_PATH)

    def read_img(self, path):
        print(path)
        img = cv2.imread(path,0)
        return img


    def rotate_img(self, img, degree):
        rotated = imutils.rotate_bound(img, degree)
        # cv2.imshow(f"Rotated by {degree} Degrees", rotated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        xmin = ceil(self.MAP_SLICE * abs(sin(radians(degree)) * cos(radians(degree))))

        # i, yfirst = 0, 0
        # for x in rotated:  # find ymax for current xmin
        #     if x[xmin]:
        #         yfirst = i
        #         break
        #     i+=1

        # print(kol)
        # image = cv2.rectangle(rotated, (xmin, yfirst), (xmin+self.MAP_SLICE, yfirst+self.MAP_SLICE), 255, 2)
        # cropped_image = rotated[yfirst:yfirst+self.MAP_SLICE, xmin:xmin+self.MAP_SLICE]

        # print(f"FUNCTION: rotate_img {xmin} {yfirst}")
        # plt.imshow(rotated,cmap = 'gray')
        # plt.plot(xmin, yfirst, "rx")
        # plt.show()

        # plt.imshow(image,cmap = 'gray')
        # plt.show()

        return rotated, xmin
    

    def piece_of_map(self, img, coords, shape):
        source_img = img.copy()
        cropped_image = source_img[coords[1]:coords[1]+shape, coords[0]:coords[0]+shape]

        # fig, ax = plt.subplots(1,2,figsize=(18,9))
        # ax[0].imshow(image,cmap = 'gray')
        # ax[1].imshow(cropped_image,cmap = 'gray')
        # plt.show()

        return cropped_image.copy()

    
    def random_piece_of_map(self, img, xmin):
        source_img = img.copy()
        max_width = source_img.shape[1]
        max_hight = source_img.shape[0]
        left_w = random.randint(xmin, max_width - xmin - self.MAP_SLICE) 

        i, ymin, ymax = 0, 0, max_hight
        for x in source_img:  # find ymin and ymax for current xmin
            if x[left_w]:
                if not ymin:
                    ymin = i
                ymax = i
            i+=1
        i=0

        # print(f"FUNCTION: random_piece_of_map {xmin} {ymin} {ymax}")
        # plt.imshow(source_img,cmap = 'gray')
        # plt.plot(xmin, ymin, "rx")
        # plt.plot(xmin, ymax, "rx")
        # plt.show()


        left_h = random.randint(ymin, ymax - self.MAP_SLICE) 
        self.coords = (left_w, left_h)
        bottom_right = (self.coords[0] + self.MAP_SLICE, self.coords[1] + self.MAP_SLICE)
        # print(top_left, bottom_right)
        image = cv2.rectangle(source_img , self.coords, bottom_right, 255, 2)
        self.cropped_image = source_img[left_h:left_h+self.MAP_SLICE, left_w:left_w+self.MAP_SLICE]

        # print(self.coords, bottom_right)
        # plt.imshow(image,cmap = 'gray')
        # plt.show()

        # plt.imshow(self.cropped_image,cmap = 'gray')
        # plt.show()

        return self.cropped_image, self.coords


    def show_current_result(self, vis=None, img1_shape=None, img2_shape=None, original_coords=None, photo_coords=None, 
                            map=None, length_hist=None, degree_hist=None):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))
        plt.gray()

        ax[0, 0].imshow(vis)
        ax[0, 0].axis('off')
        ax[0, 0].set_title("Map vs. Photo on board\n"
                        "(all keypoints and matches)")

        ax[0, 1].bar([str(elem) for elem in length_hist['length']], length_hist['count'])
        # ax[0, 1].axis('off')
        ax[0, 1].set_title("Vectors length histogram")

        ax[1, 1].bar([str(elem) for elem in degree_hist['degree']], degree_hist['count'])
        # ax[1, 1].axis('off')
        ax[1, 1].set_title("Vectors degrees histogram")

        original_coords_br = (original_coords[0]+img1_shape, original_coords[1]+img1_shape)
        map = cv2.rectangle(map, original_coords, original_coords_br, 255, 2)

        photo_coords_br = (photo_coords[0]+img2_shape, photo_coords[1]+img2_shape)
        map = cv2.rectangle(map, photo_coords, photo_coords_br, 255, 2)

        ax[1, 0].imshow(map,cmap = 'gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title("Current photos")

        plt.tight_layout()
        plt.show()


    def show_total_result(self, map=None, img2_shape=None, photo_coords=None, l_ME_list=None, l_SD_list=None, l_CV_list=None, 
                          d_ME_list=None, d_SD_list=None, d_CV_list=None, algo_name="No name"):
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 8))
        fig.suptitle(algo_name, fontsize=16)
        plt.gray()
        
        photo_coords_br = (photo_coords[0]+img2_shape, photo_coords[1]+img2_shape)
        map = cv2.rectangle(map, photo_coords, photo_coords_br, 255, 2)

        ax[2, 1].imshow(map,cmap = 'gray')
        ax[2, 1].set_title("Map and photo")
        ax[2, 0].axis('off')
        ax[2, 1].axis('off')
        ax[2, 2].axis('off')

        ax[0, 0].set_title("Length mathematical expectation")
        ax[0, 0].imshow(l_ME_list, cmap = "gray", vmin=0, vmax=255)
        ax[0, 0].axis('off')

        ax[0, 1].set_title("Length standard deviation")
        ax[0, 1].imshow(l_SD_list, cmap = "gray", vmin=0, vmax=255)
        ax[0, 1].axis('off')

        ax[0, 2].set_title("Length coefficient of variation")
        ax[0, 2].imshow(l_CV_list, cmap = "gray", vmin=0, vmax=255)
        ax[0, 2].axis('off')

        ax[1, 0].set_title("Degree mathematical expectation")
        ax[1, 0].imshow(d_ME_list, cmap = "gray", vmin=0, vmax=255)
        ax[1, 0].axis('off')

        ax[1, 1].set_title("Degree standard deviation")
        ax[1, 1].imshow(d_SD_list, cmap = "gray", vmin=0, vmax=255)
        ax[1, 1].axis('off')

        ax[1, 2].set_title("Degree coefficient of variation")
        ax[1, 2].imshow(d_CV_list, cmap = "gray", vmin=0, vmax=255)
        ax[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = ImageData()
    IMAGE_1_PATH = 'photos/maps/yandex.jpg'
    IMAGE_2_PATH = 'photos/maps/google.jpg'
    MAP_SLICE = 301
    data.start_preprocessing(IMAGE_1_PATH, IMAGE_2_PATH, MAP_SLICE)
    res = data.rotate_img(data.image1, 60)