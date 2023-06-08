from ctypes.wintypes import SHORT
from collections import defaultdict
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
    def __init__(self, img_path1, img_path2, template_count, map_slice, STEP_BETWEEN, DIFFERENCE, extrema_count, degree=0):
        self.IMAGE_1_PATH = img_path1
        self.IMAGE_2_PATH = img_path2
        self.TEMPLATE_COUNT = template_count 
        self.EXPERIMENT_COUNT = None
        self.PIXELS_STEP = 51
        self.MAP_SLICE = map_slice
        self.DIFFERENCE = DIFFERENCE
        self.SHAPE = 10
        self.EXTREMA_COUNT = extrema_count
        self.MAX_DEGREE = degree
        self.DEGREE = 0
        self.method = 'cv2.TM_CCOEFF_NORMED'
        ''' meth = 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'    '''
        self.image1_colored = None
        self.image2_colored = None
        self.image1 = None
        self.image2 = None   
        self.xmin = 0 

        self.original_shape = None
        self.photo_shape = None
        self.photo = None
        self.photo_coords = None
        self.step = STEP_BETWEEN
        self.width = None
        self.hight = None
        self.shape_i = None
        self.shape_j = None

        self.data = ImageData()   


    def experiment_KORR(self):
        self.init_time = time.time()
        self.data.start_preprocessing(self.IMAGE_1_PATH, self.IMAGE_2_PATH, self.MAP_SLICE) 
        self.image1_colored = self.data.image1.copy()
        self.image2_colored = self.data.image2.copy()
        self.image1 = self.data.img_to_gray(self.data.image1.copy())
        self.image2 = self.data.img_to_gray(self.data.image2.copy())

        self.method_num = eval(self.method)
        x_indexes, y_indexes = [x for x in range(0, self.MAX_DEGREE+1)], []
        self.DEGREE = 0
        for i in range(0, self.MAX_DEGREE+1):
            print(f"ITERATION: {i}")
            self.DEGREE = -i
            img1_coppy = self.image1.copy()
            img2_coppy = self.image2.copy()
            result_list = []
            true_predicted_count = 0
            img2_rotated, self.xmin = self.data.rotate_img(img2_coppy, self.DEGREE)

            for j in range(self.TEMPLATE_COUNT):
                print(f"TEMPLATE {j}")
                img1_coppy_coppy = img1_coppy.copy()
                img2_rotated_coppy = img2_rotated.copy()
                img2_rotated_coppy, coords = self.data.random_piece_of_map(img2_rotated_coppy, self.xmin, self.DEGREE, img1_coppy_coppy)
                result, t_l, b_r, minv, maxv = use_cv_match_template(img1_coppy_coppy, img2_rotated_coppy, self.method_num)  # match images
                result_list.append(result)
                extrema = self.find_extrema(result, self.EXTREMA_COUNT) # find extrema
                idx = self.search_right_extremum(coords, extrema)
                if idx:
                    true_predicted_count +=1 
                # y_indexes.append(idx)
            true_predicted = true_predicted_count / self.TEMPLATE_COUNT * 100
            y_indexes.append(true_predicted)
            
            # cv2.imwrite(f'photos/results/result.png', result)
            # plt.imshow(result,cmap = 'gray')
            # plt.show()
            # show_result(result, img, crop_img, self.method)

        # print(f"TRUE PREDICTED: {true_predicted}")
        fig = plt.figure()
        print(x_indexes, '\n', y_indexes)
        print(f"SECONDS SPENT: {time.time() - self.init_time}")
        plt.plot(x_indexes, y_indexes, "b", label="korrelation")
        plt.legend(loc="upper right")
        plt.savefig(f"photos/results/exp/KORR.jpg", dpi=fig.dpi)
        # print(f"from {self.EXPERIMENT_COUNT} EXPERIMENTS found {error_count} ERRORS")
        # print(f"{round((self.EXPERIMENT_COUNT-error_count)/self.EXPERIMENT_COUNT*100, 2)}% true")


    def experiment_SIFT(self, method=None, image1=None, image2=None, step=None):
        self.init_time = time.time()

        self.data.start_preprocessing(self.IMAGE_1_PATH, self.IMAGE_2_PATH, self.MAP_SLICE) 
        self.image1_colored = self.data.image1.copy()
        self.image2_colored = self.data.image2.copy()
        self.image1 = self.data.img_to_gray(self.data.image1.copy())
        self.image2 = self.data.img_to_gray(self.data.image2.copy())

        self.original_shape = self.data.MAP_SLICE * self.DIFFERENCE
        self.photo_shape = self.data.MAP_SLICE

        self.width = self.image1.shape[1] - self.photo_shape
        self.hight = self.image1.shape[0] - self.photo_shape
        self.shape_i = int(self.hight / self.step)
        self.shape_j = int(self.width / self.step) +1

        x_indexes, metrics_indexes, vectors_indexes = [x for x in range(0, self.MAX_DEGREE+1)], [], []
        for i in range(0, self.MAX_DEGREE+1):
            print(f"ITERATION: {i}")
            self.DEGREE = i
            img1_coppy = self.image1.copy()
            img2_coppy = self.image2.copy()
            result_list = []
            metrics_true_predicted_count = 0
            vectors_true_predicted_count = 0
            img2_rotated, self.xmin = self.data.rotate_img(img2_coppy, self.DEGREE)
            for j in range(self.TEMPLATE_COUNT):
                print(f"TEMPLATE {j}")
                img1_coppy_coppy = img1_coppy.copy()
                img2_rotated_coppy = img2_rotated.copy()

                self.photo, self.photo_coords = self.data.random_piece_of_map(img2_rotated_coppy, self.xmin, self.DEGREE, img1_coppy_coppy)
                CV_list, true_vectors_list, extremas_metrics, extremas_vectors = self.start_experiment(method, i, j) # match images


                idx = self.search_right_extremum(self.photo_coords, extremas_metrics, self.step)
                if idx:
                    metrics_true_predicted_count +=1 

                idx = self.search_right_extremum(self.photo_coords, extremas_vectors, self.step)
                if idx:
                    vectors_true_predicted_count +=1 

            true_predicted = metrics_true_predicted_count / self.TEMPLATE_COUNT * 100
            metrics_indexes.append(true_predicted)
            
            true_predicted = vectors_true_predicted_count / self.TEMPLATE_COUNT * 100
            vectors_indexes.append(true_predicted)

        # print(x_indexes, '\n', y_indexes)
        
        fig = plt.figure()        
        print(f"SECONDS SPENT: {time.time() - self.init_time}")
        plt.suptitle(f"Using {method} method", fontsize=16)
        plt.plot(x_indexes, metrics_indexes, 'b', label="CV")
        plt.plot(x_indexes, vectors_indexes, 'r', label="vectors")
        plt.legend(loc="upper right")
        plt.savefig(f"photos/results/exp/{method}.jpg", dpi=fig.dpi)
        # plt.show()


    def start_experiment(self, exp_method=None, degree=0, template=0):
        print(f"START {exp_method}")
        init_time = time.time()

        it, coord_i = 0, 0
        true_vectors_list, CV_list = np.empty(shape=(0,self.shape_j)), np.empty(shape=(0,self.shape_j))

        while coord_i < (self.hight):
            jt, coord_j = 0, 0
            true_vectors, CV_row = np.array([]), np.array([])

            while coord_j < (self.width):
                print(f"\n[DEG: {degree}, TEMP: {template}]: ---------- ITERATION: {it}, {jt} ----------")
                original_coords = (coord_j, coord_i)
                original = self.data.piece_of_map(self.image1.copy(), original_coords, self.original_shape)

                true_vectors_count, length_hist, degree_hist, vis1, vis2, cv_metric = self.choose_method(exp_method, original, self.photo)

                true_vectors = np.append(true_vectors, true_vectors_count)
                CV_row = np.append(CV_row, cv_metric)

                # self.data.show_current_result(vis1, self.original_shape, self.photo_shape, original_coords, self.photo_coords, self.image1.copy(), 
                #                               length_hist, degree_hist, f"{exp_method} algorithm (count metrics)")
                # self.data.show_current_result(vis2, self.original_shape, self.photo_shape, original_coords, self.photo_coords, self.image1.copy(), 
                #                               length_hist, degree_hist, f"{exp_method} algorithm (check vectors)")
                
                coord_j += self.step
                jt+=1

            true_vectors_list = np.append(true_vectors_list, [true_vectors], axis=0)
            CV_list = np.append(CV_list, [CV_row], axis=0)

            coord_i += self.step
            it+=1

        true_vectors_list = self.normalize_array(true_vectors_list)
        CV_list = self.normalize_array(CV_list)

        extremas_metrics = self.find_extrema(CV_list, self.EXTREMA_COUNT, 3, 0)
        extremas_vectors = self.find_extrema(true_vectors_list, self.EXTREMA_COUNT, 3, 0)

        time_spent = time.time() - init_time

        # ------------------VISUALISATION------------------

        # print(f"\nSECONDS SPENT: {time_spent}")
        
        # self.data.show_total_result(self.image1_colored.copy(), self.original_shape, self.photo_shape, self.photo_coords, CV_list, extremas_metrics, 
        #                             self.step, f"{exp_method} algorithm using CV\nby {self.step} step\n{round(time_spent, 2)} seconds spent")
        
        # self.data.show_total_result(self.image1_colored.copy(), self.original_shape, self.photo_shape, self.photo_coords, true_vectors_list, extremas_vectors,
        #                             self.step, f"{exp_method} algorithm using identical vectors\nby {self.step} step\n{round(time_spent, 2)} seconds spent")
        
        return CV_list, true_vectors_list, extremas_metrics, extremas_vectors

    
    def normalize_array(self, arr):
        return (255*(arr - np.min(arr))/np.ptp(arr)).astype(int) 


    def find_extrema(self, res, count, neighborhood_size=100, threshold=0.05):
        res_data = res.copy()

        data_max = ndimage.maximum_filter(res_data, neighborhood_size)
        maxima = (res_data == data_max)
        data_min = ndimage.minimum_filter(res_data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)

        # slices = np.array(ndimage.center_of_mass(res_data, labeled, range(1, num_objects+1)))
        slices = ndimage.find_objects(labeled)

        x, y = [], []
        extrema = []
        for dy,dx in slices:
            x_center = int((dx.start + dx.stop - 1)/2)
            # x_center = int(dx)
            x.append(x_center)
            y_center = int((dy.start + dy.stop - 1)/2)
            # y_center = int(dy)
            y.append(y_center)
            extrema.append((res_data[y_center][x_center], x_center, y_center))
            # print(type(x_center),type(y_center))

        # plt.imshow(res_data,cmap = 'gray')  
        # plt.plot(x, y, "rx")
        # plt.show()
        
        extrema = sorted(extrema,  reverse=True)[:count]
        # extrema = [x[1] for x in extrema]

        # x, y = [], []
        # for cor in extrema:
        #     x.append(cor[1])
        #     y.append(cor[2])

        # plt.imshow(res_data,cmap = 'gray')  
        # plt.plot(x, y, "rx")
        # plt.show()

        return extrema


    def choose_method(self, method="A-SIFT", original=None, photo=None, original_coords=None, photo_coords=None, image1=None):
        if method == "SIFT":
            return start_SIFT(original, photo)
        if method == "A-SIFT":
            return start_A_SIFT(original, photo)


    def search_right_extremum(self, coords, extrema, step=1):
        x, y = coords
        i=1
        for cor in extrema:
            x_ = cor[1]*step
            y_ = cor[2]*step
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