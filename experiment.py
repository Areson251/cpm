# from turtle import width
from ctypes.wintypes import SHORT
import cv2
import numpy as np
import time, datetime
import math
import random
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from comparison import *


import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


EXPERIMENT_COUNT = 1
IMAGE_1_PATH = 'photos/maps/yandex.jpg'
IMAGE_2_PATH = 'photos/maps/google.jpg'
PIXELS_STEP = 51
MAP_SLICE = 501
SHAPE = 10


def piece_of_map(img):
    map = img.copy()
    max_width = map.shape[1]
    max_hight = map.shape[0]
    left_w = random.randint(0, max_width - MAP_SLICE) 
    left_h = random.randint(0, max_hight - MAP_SLICE) 
    top_left = (left_w, left_h)
    bottom_right = (top_left[0] + MAP_SLICE, top_left[1] + MAP_SLICE)
    # print(top_left, bottom_right)
    image = cv2.rectangle(map , top_left, bottom_right, 255, 2)
    cropped_image = map[left_h:left_h+MAP_SLICE, left_w:left_w+MAP_SLICE]

    # show_result(cropped_image)
    # cv2.imshow("cropped", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    return cropped_image, image, (top_left, bottom_right)


def find_extrema(data):
    neighborhood_size = 100
    threshold = 0.01

    # data = scipy.misc.imread(fname)

    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    extrema = {}
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
        # extrema{data[x_center, y_center]} = (x_center, y_center)
        # print(type(x_center),type(y_center))
    
    # plt.imshow(data)
    # plt.plot(x, y, "rx")
    # plt.show()
    return [x,y]


def experiment():
    ''' meth = 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'    '''
    method = 'cv2.TM_CCOEFF_NORMED'
    method_num = eval(method)
    error_count=0
    for i in range(EXPERIMENT_COUNT):
        crop_img, img, coords = piece_of_map(image2)
        img1_coppy = image1.copy()
        result, t_l, b_r, minv, maxv = use_cv_match_template(img1_coppy, crop_img, method_num)
        print(i, coords, t_l, b_r)
        if coords[0] != t_l:
            print("ERROR")
            error_count+=1
        extrema = find_extrema(result)

        # cv2.imwrite(f'photos/results/result.png', result)
        # plt.imshow(result,cmap = 'gray')
        # plt.show()
        # show_result(result, img, crop_img, method)

    print(f"from {EXPERIMENT_COUNT} EXPERIMENTS found {error_count} ERRORS")
    print(f"{round((EXPERIMENT_COUNT-error_count)/EXPERIMENT_COUNT*100, 2)}% true")


if __name__ == "__main__":
    init_time = time.time()
    image1 = cv2.imread(IMAGE_1_PATH,0)
    image2 = cv2.imread(IMAGE_2_PATH,0)
    experiment()