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

EXPERIMENT_COUNT = 30
IMAGE_1_PATH = 'photos/maps/yandex.jpg'
IMAGE_2_PATH = 'photos/pictures/g_cropped.jpg'
PIXELS_STEP = 51
MAP_SLICE = 501
SHAPE = 10


def count_difference_with_step(image1, image2, step=101):
    w, h = count_shapes(image1, image2)
    width = image1.shape[1] - image2.shape[1]
    height = image1.shape[0] - image2.shape[0]

    image_pixels = np.array([]) 
    i_num, kol_steps = 0, 0

    #TODO: write normal cycle
    while i_num <= height:
        pixels_row, j_num = np.array([]), 0
        while j_num <= width:
            sum = 0
            for i in range(image2.shape[0]):
                for j in range(image2.shape[1]):
                    # print(i, j, i_num, j_num)
                    sum += 255 - abs(image1.item((i_num + i, j_num + j)) - image2.item((i, j))) 
            # pixel = sum / (image2.shape[0] * image2.shape[1])
            image_pixels = np.append(image_pixels, sum / (image2.shape[0] * image2.shape[1]))
            # image_pixels.item(i_num, j_num) = pixel
            j_num+=step  
        # image_pixels = np.append(image_pixels, pixels_row, axis=0)  
        i_num += step
        kol_steps += 1
        print(f"STEP NUMBER: {kol_steps}")
    
    min = np.amin(image_pixels)
    max = np.amax(image_pixels)
    A = 255 * (image_pixels - min)//(max-min)
    B = np.reshape(A.astype(int), (math.floor(i_num/step), -1))
    return B


def use_cv_match_template(img, template, method):
    res = cv2.matchTemplate(img, template, method)
    # print(res)
    w = template.shape[1]
    h = template.shape[0]

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # cv2.rectangle(res, top_left, (top_left[0] + template.shape[0], top_left[1] + template.shape[1]), (0,0,0), 2, 8, 0 )
    norm = np.zeros((800,800))
    res = cv2.normalize(res,  norm, 0, 255, cv2.NORM_MINMAX)
    # res = np.around(res)

    # cv2.imshow('result_window', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(f'photos/results/result_{190}.png', res)

    return res, top_left, bottom_right, min_val, max_val


def create_convolution(image1, image2, step):
    width, height = count_shapes(image1, image2)

    image_pixels = []

    #TODO: write normal cycle
    for i_num in range(height):
        pixels_row = []
        for j_num in range(width):
            sum = 0
            for i in range(image2.shape[0]):
                for j in range(image2.shape[1]):
                    sum += 255 - abs(image1.item((i_num*image2.shape[0] + i, j_num*image2.shape[1] + j)) - image2.item((i, j))) 
            pixel = sum / (image2.shape[0] * image2.shape[1])
            pixels_row.append(round(pixel))      
        image_pixels.append(pixels_row)  
    

def count_shapes(image1, image2):
    width = math.floor(image1.shape[1] / image2.shape[1])
    height = math.floor(image1.shape[0] / image2.shape[0])

    return width, height


def create_image(pixels):
    avarage = int(np.sum(pixels) / (pixels.shape[0] * pixels.shape[1]))
    pixels[pixels < avarage] = 0
    
    return pixels


def show_result(result=None, img=None, crop_img=None, method=None, top_left=None, bottom_right=None):
    plt.subplot(221),plt.imshow(result,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(crop_img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)
    plt.show()


if __name__ == "__main__":
    init_time = time.time()

    # Load an color image in grayscale
    image1 = cv2.imread(IMAGE_1_PATH,0)
    image2 = cv2.imread(IMAGE_2_PATH,0)

    # pixels = count_difference(image1, image2)
    # pixels = create_convolution(image1, image2, PIXELS_STEP)

    ''' meth = 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'    '''
    method = eval('cv2.TM_CCOEFF')
    result, t_l, b_r = use_cv_match_template(image1, image2)
    show_result(result, method, t_l, b_r)

    # pixels = count_difference_with_step(image1, image2, PIXELS_STEP)

    # # pixels = create_image(pixels)
    
    # print(f"SECONDS SPENT: {time.time() - init_time}")
    # # show image
    # cv2.imshow('result', pixels.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(f'photos/results/result_{SHAPE}.png', pixels)
