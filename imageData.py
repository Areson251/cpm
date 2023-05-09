from ctypes.wintypes import SHORT
from math import sqrt
import cv2
import numpy as np
import time, datetime
import math
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

    def start_preprocessing(self, path1, path2, slice):
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
        return rotated
    
    def piece_of_map(self, img, border):
        map = img.copy()
        max_width = map.shape[1]
        max_hight = map.shape[0]
        left_w = random.randint(border, max_width - border - self.MAP_SLICE) 
        left_h = random.randint(border, max_hight - border - self.MAP_SLICE) 
        self.coords = (left_w, left_h)
        bottom_right = (self.coords[0] + self.MAP_SLICE, self.coords[1] + self.MAP_SLICE)
        # print(top_left, bottom_right)
        image = cv2.rectangle(map , self.coords, bottom_right, 255, 2)
        self.cropped_image = map[left_h:left_h+self.MAP_SLICE, left_w:left_w+self.MAP_SLICE]

        # show_result(cropped_image)
        # cv2.imshow("cropped", cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 

        return self.cropped_image, self.coords



if __name__ == "__main__":
    data = ImageData()