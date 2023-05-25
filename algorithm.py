# from turtle import width
from ctypes.wintypes import SHORT
import cv2
import numpy as np
import time, datetime
import random
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from algorithm import *
import pandas as pd 
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
from IPython.display import clear_output
from math import sin, cos, degrees
from asift import Timer, image_resize, init_feature, filter_matches, affine_detect
from multiprocessing.pool import ThreadPool
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
from imageData import *

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
    # res = np.around(res)

    # cv2.imshow('result_window', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(f'photos/results/result_{190}.png', res)

    return res, top_left, bottom_right, min_val, max_val


def start_SIFT(img1=None, img2=None, original_coords=None, photo_coords=None, map=None):
    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,
                                cross_check=True)

    cor = extract_coords(img1, img2, keypoints1, keypoints2, matches12)

    length_hist = np.array([], dtype=[('length', 'U10'), ('count', 'i4')])
    degree_hist = np.array([], dtype=[('degree', 'U10'), ('count', 'i4')])
    
    for dots in cor: 
        l = sqrt((dots[0][0]-dots[1][0])*(dots[0][0]-dots[1][0])+(dots[0][1]-dots[1][1])*(dots[0][1]-dots[1][1]))
        a = 180 - degrees(np.arccos((dots[0][0]-dots[1][0])/(l)))
        l = str(round(l, 6))
        a = str(round(a, 2))
        lens = length_hist['length']
        l_counts = length_hist['count']
        deg = degree_hist['degree']
        a_counts = degree_hist['count']

        l_idx = np.where(lens == l)[0]
        if not l_idx.size == 0:
            length_hist[l_idx[0]] = (l, l_counts[l_idx[0]]+1)
        else:
            length_elem = np.array([(l, 1)], dtype=[('length', 'U10'), ('count', 'i4')])
            length_hist = np.append(length_hist, length_elem)

        a_idx = np.where(deg == a)[0]
        if not a_idx.size == 0:
            degree_hist[a_idx[0]] = (a, a_counts[a_idx[0]]+1)
        else:
            degree_elem = np.array([(a, 1)], dtype=[('degree', 'U10'), ('count', 'i4')])
            degree_hist = np.append(degree_hist, degree_elem)

    print(f"FOUND {length_hist['length'].size} DIFFERENT VECTORS (by length)")
    print(f"FOUND {degree_hist['degree'].size} DIFFERENT VECTORS (by degree)")

    length_hist = np.sort(length_hist)
    degree_hist = np.sort(degree_hist)

    # ------------------VISUALISATION------------------

    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))
    # plt.gray()
    # cor = plot_matches(ax[0, 0], img1, img2, keypoints1, keypoints2, matches12, only_matches=True)

    # ax[0, 0].plot(dots[0][0], dots[0][1], "rx")
    # ax[0, 0].plot(dots[1][0], dots[1][1], "bx")

    # ax[0, 0].axis('off')
    # ax[0, 0].set_title("Map vs. Photo on board\n"
    #                 "(all keypoints and matches)")

    # ax[0, 1].bar(length_hist['length'], length_hist['count'])
    # # ax[0, 1].axis('off')
    # ax[0, 1].set_title("Vectors length histogram")

    # ax[1, 1].bar(degree_hist['degree'], degree_hist['count'])
    # # ax[1, 1].axis('off')
    # ax[1, 1].set_title("Vectors degrees histogram")
    
    # # plot_matches(ax[1, 1], img1, img3, keypoints1, keypoints3, matches13[::15],
    # #             only_matches=True)
    # # ax[1, 1].axis('off')
    # # ax[1, 1].set_title("Original Image vs. Transformed Image\n"
    # #                 "(subset of matches for visibility)")

    # original_coords_br = (original_coords[0]+img1.shape[0], original_coords[1]+img1.shape[0])
    # map = cv2.rectangle(map, original_coords, original_coords_br, 255, 2)

    # photo_coords_br = (photo_coords[0]+img2.shape[0], photo_coords[1]+img2.shape[0])
    # map = cv2.rectangle(map, photo_coords, photo_coords_br, 255, 2)

    # ax[1, 0].imshow(map,cmap = 'gray')
    # ax[1, 0].axis('off')
    # ax[1, 0].set_title("Current photos")

    # plt.tight_layout()
    # plt.show()
    # i=0

    return length_hist, degree_hist


def extract_coords(image1, image2, keypoints1, keypoints2, matches, alignment='horizontal'):
    # image1 = img_as_float(image1)
    # image2 = img_as_float(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == 'horizontal':
        offset[0] = 0
    elif alignment == 'vertical':
        offset[1] = 0
    else:
        mesg = (f"plot_matches accepts either 'horizontal' or 'vertical' for "
                f"alignment, but '{alignment}' was given. See "
                f"https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
                f"for details.")
        raise ValueError(mesg)

    coords = []
    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]
        coords.append(((keypoints1[idx1, 1], keypoints1[idx1, 0]),
                    (keypoints2[idx2, 1] + offset[1], keypoints2[idx2, 0] + offset[0])))
    return coords


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
