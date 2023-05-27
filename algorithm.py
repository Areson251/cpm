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
from math import sin, cos, degrees, floor
from asift import Timer, image_resize, init_feature, filter_matches, affine_detect
from multiprocessing.pool import ThreadPool
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
from decimal import Decimal
from imageData import *

EXPERIMENT_COUNT = 30
IMAGE_1_PATH = 'photos/maps/yandex.jpg'
IMAGE_2_PATH = 'photos/pictures/g_cropped.jpg'
PIXELS_STEP = 51
MAP_SLICE = 501
SHAPE = 10


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
    print("START SIFT ALGORITHM")
    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,
                                cross_check=True)

    cor, vis = extract_coords(img1, img2, keypoints1, keypoints2, matches12)

    length_hist = np.array([], dtype=[('length', 'f4'), ('count', 'i4')])
    degree_hist = np.array([], dtype=[('degree', 'f4'), ('count', 'i4')])
    
    for dots in cor: 
        x1, y1 = dots[0][0], dots[0][1]
        x2, y2 = dots[1][0], dots[1][1]

        length_hist, degree_hist = add_elem_to_hist(length_hist, degree_hist, x1, y1, x2, y2)

    length_hist = np.sort(length_hist)
    degree_hist = np.sort(degree_hist)

    print(f"FOUND {length_hist['length'].size} DIFFERENT VECTORS (by length)")
    print(f"FOUND {degree_hist['degree'].size} DIFFERENT VECTORS (by degree)")

    l_ME, l_SD, l_CV = count_metrics(data_hist=length_hist, param="length")
    d_ME, d_SD, d_CV = count_metrics(data_hist=degree_hist, param="degree")

    return length_hist, degree_hist, vis, (l_ME, l_SD, l_CV), (d_ME, d_SD, d_CV)


def start_A_SIFT(ori_img1_, ori_img2_, MAX_SIZE=1024):
    print("START ASIFT ALGORITHM")
    clahe = cv2.createCLAHE(clipLimit=16, tileGridSize=(16,16))
    # ori_img1_ = clahe.apply(ori_img1_)
    # ori_img2_ = clahe.apply(ori_img2_)

    detector_name = "sift-flann"
    detector, matcher = init_feature(detector_name)

    ratio_1 = 1
    ratio_2 = 1

    if ori_img1_.shape[0] > MAX_SIZE or ori_img1_.shape[1] > MAX_SIZE:
        ratio_1 = MAX_SIZE / ori_img1_.shape[1]
        print("Large input detected, image 1 will be resized")
        img1_ = image_resize(ori_img1_, ratio_1)
    else:
        img1_ = ori_img1_

    if ori_img2_.shape[0] > MAX_SIZE or ori_img2_.shape[1] > MAX_SIZE:
        ratio_2 = MAX_SIZE / ori_img2_.shape[1]
        print("Large input detected, image 2 will be resized")
        img2_ = image_resize(ori_img2_, ratio_2)
    else:
        img2_ = ori_img2_

    print(f"Using {detector_name.upper()} detector...")

    # Profile time consumption of keypoints extraction
    with Timer(f"Extracting {detector_name.upper()} keypoints..."):
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        kp1, desc1 = affine_detect(detector, img1_, pool=pool)
        kp2, desc2 = affine_detect(detector, img2_, pool=pool)

    print(f"img1 - {len(kp1)} features, img2 - {len(kp2)} features")

    # Profile time consumption of keypoints matching
    with Timer('Matching...'):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)

    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, ratio=0.7)

    if len(p1) >= 4:
        # TODO: The effect of resizing on homography matrix needs to be investigated.
        # TODO: Investigate function consistency when image aren't resized.
        for index in range(len(p1)):
            pt = p1[index]
            p1[index] = pt / ratio_1

        for index in range(len(p2)):
            pt = p2[index]
            p2[index] = pt / ratio_2

        for index in range(len(kp_pairs)):
            element = kp_pairs[index]
            kp1, kp2 = element

            new_kp1 = cv2.KeyPoint(kp1.pt[0] / ratio_1, kp1.pt[1] / ratio_1, kp1.size)
            new_kp2 = cv2.KeyPoint(kp2.pt[0] / ratio_2, kp2.pt[1] / ratio_2, kp2.size)

            kp_pairs[index] = (new_kp1, new_kp2)

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 100.0)
        print(f"{np.sum(status)} / {len(status)}  inliers/matched")
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
    else:
        H, status = None, None
        print(f"{len(p1)} matches found, not enough for homography estimation")
    
    h1, w1 = img1_.shape[:2]
    h2, w2 = img2_.shape[:2]

    # Create visualized result image
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1_
    vis[:h2, w1:w1 + w2] = img2_
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255), thickness=10)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking

    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) * ratio_2 + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)

    length_hist = np.array([], dtype=[('length', 'f4'), ('count', 'i4')])
    degree_hist = np.array([], dtype=[('degree', 'f4'), ('count', 'i4')])

    for (x1, y1), (x2, y2) in [(p1[i], p2[i]) for i in range(len(p1))]:
        #if inlier:
        cv2.line(vis, (x1, y1), (x2, y2), green, thickness=2)
        length_hist, degree_hist = add_elem_to_hist(length_hist, degree_hist, x1, y1, x2, y2)

    length_hist = np.sort(length_hist)
    degree_hist = np.sort(degree_hist)

    print(f"FOUND {length_hist['length'].size} DIFFERENT VECTORS (by length)")
    print(f"FOUND {degree_hist['degree'].size} DIFFERENT VECTORS (by degree)")
            
    l_ME, l_SD, l_CV = count_metrics(data_hist=length_hist, param="length")
    d_ME, d_SD, d_CV = count_metrics(data_hist=degree_hist, param="degree")

    return length_hist, degree_hist, vis, (l_ME, l_SD, l_CV), (d_ME, d_SD, d_CV)


def count_metrics(data_hist=None, param=""):            
    param_ = data_hist[param]
    counts = data_hist['count']
    n = sum(counts)
    s = counts.size

    if n: 
        ME = floor(count_ME(param_, counts, n, s))
        param_2 = [(x)**2 for x in param_]
        ME2 = ceil(count_ME(param_2, counts, n, s))
        m = floor((ME)**2)
        D = round(ME2-m, 2)
        SD = sqrt(D)
        CV = ME/SD
    else: 
        ME = -1
        SD = -1
        CV = -1
    
    return ME, SD, CV


def count_ME(param_=[], counts=[], n=None, s=None):
    c = np.array([x/n for x in counts])
    return np.sum(param_ * c)


def add_elem_to_hist(length_hist, degree_hist, x1, y1, x2, y2): 
    l = sqrt((x1-x2)**2+(y1-y2)**2)
    a = 180 - degrees(np.arccos((x1-x2)/(l)))
    l = round(l, 4)
    a = round(a, 2)
    lens = length_hist['length']
    l_counts = length_hist['count']
    deg = degree_hist['degree']
    a_counts = degree_hist['count']

    l_idx = np.where(lens == l)[0]
    if not l_idx.size == 0:
        length_hist[l_idx[0]] = (l, l_counts[l_idx[0]]+1)
    else:
        length_elem = np.array([(l, 1)], dtype=[('length', 'f4'), ('count', 'i4')])
        length_hist = np.append(length_hist, length_elem)

    a_idx = np.where(deg == a)[0]
    if not a_idx.size == 0:
        degree_hist[a_idx[0]] = (a, a_counts[a_idx[0]]+1)
    else:
        degree_elem = np.array([(a, 1)], dtype=[('degree', 'f4'), ('count', 'i4')])
        degree_hist = np.append(degree_hist, degree_elem)
    
    return length_hist, degree_hist


def extract_coords(image1, image2, keypoints1, keypoints2, matches, alignment='horizontal', keypoints_color='k', matches_color=None, only_matches=True,):
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
        vis = np.concatenate([image1, image2], axis=1)
    elif alignment == 'vertical':
        offset[1] = 0
        vis = np.concatenate([image1, image2], axis=0)
    else:
        mesg = (f"plot_matches accepts either 'horizontal' or 'vertical' for "
                f"alignment, but '{alignment}' was given. See "
                f"https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
                f"for details.")
        raise ValueError(mesg)

    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    coords=[]
    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]

        
        coords.append(((keypoints1[idx1, 1], keypoints1[idx1, 0]),
                    (keypoints2[idx2, 1] + offset[1], keypoints2[idx2, 0] + offset[0])))
        
        vis = cv2.line(vis, (keypoints1[idx1, 1], keypoints1[idx1, 0]),
                    (keypoints2[idx2, 1] + offset[1], keypoints2[idx2, 0] + offset[0]), (255, 0, 0), thickness=2)
        
    # fig, ax = plt.subplots(figsize=(18,8))
    # ax.imshow(vis)
    # plt.show()
    # plt.close()

    return coords, vis


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
