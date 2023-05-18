from ctypes.wintypes import SHORT
from math import sqrt
import cv2
import numpy as np
import time, datetime
from  math import sin, cos, radians
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
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img

    def rotate_img(self, img, degree):
        rotated = imutils.rotate_bound(img, degree)
        # cv2.imshow(f"Rotated by {degree} Degrees", rotated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        xmin = math.ceil(self.MAP_SLICE * abs(sin(radians(degree)) * cos(radians(degree))))

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
    
    def piece_of_map(self, img, xmin):
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

        # print(f"FUNCTION: piece_of_map {xmin} {ymin} {ymax}")
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



if __name__ == "__main__":
    data = ImageData()
    IMAGE_1_PATH = 'photos/maps/yandex.jpg'
    IMAGE_2_PATH = 'photos/maps/google.jpg'
    MAP_SLICE = 301
    data.start_preprocessing(IMAGE_1_PATH, IMAGE_2_PATH, MAP_SLICE)
    res = data.rotate_img(data.image1, 60)