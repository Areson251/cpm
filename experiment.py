from ctypes.wintypes import SHORT
from math import sqrt
import cv2
import numpy as np
import time, datetime
import math
import random
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from sklearn.preprocessing import normalize
from algorithm import *
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import argparse
import imutils
from imageData import ImageData

# from test import start_A_SIFT


class Experiment:
    def __init__(self, img_path1, img_path2, template_count, map_slice, exp_count, extrema_count, degree=0):
        self.IMAGE_1_PATH = img_path1
        self.IMAGE_2_PATH = img_path2
        self.TEMPLATE_COUNT = template_count 
        self.EXPERIMENT_COUNT = exp_count
        self.PIXELS_STEP = 51
        self.MAP_SLICE = map_slice
        self.SHAPE = 10
        self.EXTREMA_COUNT = extrema_count
        self.MAX_DEGREE = degree
        self.DEGREE = 0
        self.method = 'cv2.TM_CCOEFF_NORMED'
        ''' meth = 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'    '''
        self.image1 = None
        self.image2 = None   
        self.xmin = 0 

        self.original_shape = None
        self.photo_shape = None
        self.photo = None
        self.photo_coords = None
        self.step = None
        self.width = None
        self.hight = None
        self.shape_i = None
        self.shape_j = None

        self.data = ImageData()   


    def experiment_KORR(self):
        self.init_time = time.time()
        self.data.start_preprocessing(self.IMAGE_1_PATH, self.IMAGE_2_PATH, self.MAP_SLICE) 
        self.image1 = self.data.image1
        self.image2 = self.data.image2

        self.method_num = eval(self.method)
        x_indexes, y_indexes = [x for x in range(1, self.MAX_DEGREE+1)], []
        error_count=0
        for i in range(self.MAX_DEGREE):
            print(f"ITERATION: {i}")
            img1_coppy = self.image1.copy()
            img2_coppy = self.image2.copy()
            result_list = []
            true_predicted_count = 0
            img2_rotated, self.xmin = self.data.rotate_img(img2_coppy, self.DEGREE)

            for j in range(self.TEMPLATE_COUNT):
                print(f"TEMPLATE {j}")
                img1_coppy_coppy = img1_coppy.copy()
                img2_rotated_coppy = img2_rotated.copy()
                img2_rotated_coppy, coords = self.data.random_piece_of_map(img2_rotated_coppy, self.xmin)
                result, t_l, b_r, minv, maxv = use_cv_match_template(img1_coppy_coppy, img2_rotated_coppy, self.method_num)  # match images
                result_list.append(result)
                extrema = self.find_extrema(result, self.EXTREMA_COUNT, i, j) # find extrema
                idx = self.search_right_extremum(coords, extrema)
                if idx:
                    true_predicted_count +=1 
                # y_indexes.append(idx)
            true_predicted = true_predicted_count / self.TEMPLATE_COUNT * 100
            y_indexes.append(true_predicted)
            self.DEGREE += 1
            

            # cv2.imwrite(f'photos/results/result.png', result)
            # plt.imshow(result,cmap = 'gray')
            # plt.show()
            # show_result(result, img, crop_img, self.method)

        # print(f"TRUE PREDICTED: {true_predicted}")
        print(x_indexes, '\n', y_indexes)
        print(f"SECONDS SPENT: {time.time() - self.init_time}")
        plt.plot(x_indexes, y_indexes)
        plt.show()
        # print(f"from {self.EXPERIMENT_COUNT} EXPERIMENTS found {error_count} ERRORS")
        # print(f"{round((self.EXPERIMENT_COUNT-error_count)/self.EXPERIMENT_COUNT*100, 2)}% true")


    def experiment_vectors(self, image1=None, image2=None, step=None):
        self.data.start_preprocessing(self.IMAGE_1_PATH, self.IMAGE_2_PATH, self.MAP_SLICE) 
        self.image1 = self.data.image1.copy()
        self.image2 = self.data.image2.copy()

        self.original_shape = self.data.MAP_SLICE * 1
        self.photo_shape = self.data.MAP_SLICE
        self.step = int(self.data.MAP_SLICE * 0.1)
        self.photo, self.photo_coords = self.data.random_piece_of_map(self.image2.copy(), 0)
        # photo_coords = (100, 200)
        # photo = self.data.piece_of_map(image2.copy(), photo_coords, photo_shape)
        self.width = self.image1.shape[1] - self.photo_shape
        self.hight = self.image1.shape[0] - self.photo_shape
        self.shape_i = int(self.hight / self.step)
        self.shape_j = int(self.width / self.step) +1

        self.start_experiment("SIFT")
        self.start_experiment("A-SIFT")

        i=0


    def start_experiment(self, exp_method="A-SIFT"):
        print(f"START {exp_method} EXPERIMENT")
        init_time = time.time()

        it, coord_i = 0, 0
        true_vectors_list, l_ME_list, l_SD_list, l_CV_list, d_ME_list, d_SD_list, d_CV_list = np.empty(shape=(0,self.shape_j)), \
                                                                    np.empty(shape=(0,self.shape_j)), np.empty(shape=(0,self.shape_j)), \
                                                                    np.empty(shape=(0,self.shape_j)), np.empty(shape=(0,self.shape_j)), \
                                                                        np.empty(shape=(0,self.shape_j)), np.empty(shape=(0,self.shape_j)) 

        while coord_i < (self.hight):
            jt, coord_j = 0, 0
            true_vectors, l_ME, l_SD, l_CV, d_ME, d_SD, d_CV = np.array([]), np.array([]), np.array([]), np.array([]), \
                                                                                np.array([]), np.array([]), np.array([])
            while coord_j < (self.width):
                print(f"\n---------- ITERATION: {it}, {jt} ----------")
                original_coords = (coord_j, coord_i)
                original = self.data.piece_of_map(self.image1.copy(), original_coords, self.original_shape)

                true_vectors_count, length_hist, degree_hist, vis1, vis2, l_metrics, d_metrics = self.choose_method(exp_method, original, self.photo)

                true_vectors = np.append(true_vectors, true_vectors_count)
                l_ME = np.append(l_ME, l_metrics[0])
                l_SD = np.append(l_SD, l_metrics[1])
                l_CV = np.append(l_CV, l_metrics[2])
                d_ME = np.append(d_ME, d_metrics[0])
                d_SD = np.append(d_SD, d_metrics[1])
                d_CV = np.append(d_CV, d_metrics[2])

                # self.data.show_current_result(vis1, original_shape, photo_shape, original_coords, photo_coords, image1.copy(), 
                #                               length_hist, degree_hist, "SIFT algorithm (count metrics)")
                # self.data.show_current_result(vis2, original_shape, photo_shape, original_coords, photo_coords, image1.copy(), 
                                            #   length_hist, degree_hist, "SIFT algorithm (check vectors)")
                
                coord_j += self.step
                jt+=1

            true_vectors_list = np.append(true_vectors_list, [true_vectors], axis=0)
            l_ME_list = np.append(l_ME_list, [l_ME], axis=0)
            l_SD_list = np.append(l_SD_list, [l_SD], axis=0)
            l_CV_list = np.append(l_CV_list, [l_CV], axis=0)
            d_ME_list = np.append(d_ME_list, [d_ME], axis=0)
            d_SD_list = np.append(d_SD_list, [d_SD], axis=0)
            d_CV_list = np.append(d_CV_list, [d_CV], axis=0)

            coord_i += self.step
            it+=1

        # ------------------VISUALISATION------------------
        true_vectors_list = self.normalize_array(true_vectors_list)
        l_ME_list = self.normalize_array(l_ME_list)
        l_SD_list = self.normalize_array(l_SD_list)
        l_CV_list = self.normalize_array(l_CV_list)
        d_ME_list = self.normalize_array(d_ME_list)
        d_SD_list = self.normalize_array(d_SD_list)
        d_CV_list = self.normalize_array(d_CV_list)

        time_spent = time.time() - init_time

        print(f"\nSECONDS SPENT: {time_spent}")
        
        self.data.show_total_result_metrics(self.image1.copy(), self.photo_shape, self.photo_coords, l_ME_list, l_SD_list, l_CV_list, \
                                    d_ME_list, d_SD_list, d_CV_list, 
                                    f"{exp_method} algorithm using ME, SD, CV\nby {self.step} step\n{round(time_spent, 2)} seconds spent")
        
        self.data.show_total_result_vectors(self.image1.copy(), self.photo_shape, self.photo_coords, true_vectors_list, exp_method, 
                                            f"{exp_method} algorithm using identical vectors\nby {self.step} step\n{round(time_spent, 2)} seconds spent")

    
    def normalize_array(self, arr):
        # norm = np.linalg.norm(arr)
        # arr = arr / norm * 255
        return (255*(arr - np.min(arr))/np.ptp(arr)).astype(int) 


    def find_extrema(self, res, count, i, j):
        neighborhood_size = 100
        threshold = 0.05

        # data = scipy.misc.imread(fname)

        data = res.copy()

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
            x_center = int((dx.start + dx.stop - 1)/2)
            x.append(x_center)
            y_center = int((dy.start + dy.stop - 1)/2)
            y.append(y_center)
            extrema[data[y_center][x_center]] = (x_center, y_center)
            # print(type(x_center),type(y_center))
        
        extrema = sorted(extrema.items(),  reverse=True)[:count]
        extrema = [x[1] for x in extrema]

        # plt.imshow(data,cmap = 'gray')   #   THIS PLOT SHOWS FALSE INFORMATION!!!!!!!!!!!!!!!
        # plt.plot(x, y, "rx")
        # # plt.show()
        # plt.savefig(f'photos/results/exp/{i}_{j}.png')
        return extrema


    def choose_method(self, method="A-SIFT", original=None, photo=None, original_coords=None, photo_coords=None, image1=None):
        if method == "SIFT":
            return start_SIFT(original, photo)
        if method == "A-SIFT":
            return start_A_SIFT(original, photo)


    def search_right_extremum(self, coords, extrema):
        x, y = coords
        i=1
        for x_, y_ in extrema:
            if sqrt(abs(x-x_)**2 + abs(y-y_)**2) <= self.MAP_SLICE*0.3:
                return i
            else: i+=1
        return 0


if __name__ == "__main__":
    IMAGE_1_PATH = 'photos/maps/yandex.jpg'
    IMAGE_2_PATH = 'photos/maps/google.jpg'
    TEMPLATE_COUNT = 10
    MAP_SLICE = 301
    EXPERIMENT_COUNT = 31
    EXTREMA_COUNT = 10
    MAX_DEGREE = 30
    experiment = Experiment(IMAGE_1_PATH, IMAGE_2_PATH, TEMPLATE_COUNT, MAP_SLICE, EXPERIMENT_COUNT, EXTREMA_COUNT, MAX_DEGREE)
    arr = np.array([134, 54, 2])
    experiment.normalize_array(arr)