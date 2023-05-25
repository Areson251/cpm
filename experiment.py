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


    def experiment_SIFT(self, image1=None, image2=None, step=None):
        self.data.start_preprocessing(self.IMAGE_1_PATH, self.IMAGE_2_PATH, self.MAP_SLICE) 
        image1 = self.data.image1.copy()
        image2 = self.data.image2.copy()

        photo, photo_coords = self.data.random_piece_of_map(image2, 0)
        original_shape = self.data.MAP_SLICE * 2
        step = self.data.MAP_SLICE
        width = image1.shape[1] - photo.shape[1]
        hight = image1.shape[0] - photo.shape[0]
        it, coord_i = 0, 0
        Ml_list, SDl_list, CVl_list, Md_list, SDd_list, CVd_list = [], [], [], [], [], []
        a_Ml_list, a_SDl_list, a_CVl_list, a_Md_list, a_SDd_list, a_CVd_list = [], [], [], [], [], []

        while coord_i < (hight):
            jt, coord_j = 0, 0
            while coord_j < (width-step):
                print(f"ITERATION: {it}, {jt}")
                original_coords = (coord_j, coord_i)
                original = self.data.piece_of_map(image1.copy(), original_coords, original_shape)

                length_hist, degree_hist, vis1 = start_SIFT(original, photo, original_coords, photo_coords, image1.copy())
                # self.data.show_result(vis1, original_shape, step, original_coords, photo_coords, image1.copy(), length_hist, degree_hist)

                a_length_hist, a_degree_hist, vis2 = start_A_SIFT(original, photo)
                # self.data.show_result(vis2, original_shape, step, original_coords, photo_coords, image1.copy(), a_length_hist, a_degree_hist)

                # for SIFT
                lens = length_hist['length']
                l_counts = length_hist['count']
                l_n = l_counts.size

                degs = degree_hist['degree']
                d_counts = degree_hist['count']
                d_n = d_counts.size

                if l_n: 
                    Ml = sum([float(lens[i])*l_counts[i]/l_n for i in range(l_n)])
                    l_standard_deviation = sqrt(sum([(float(lens[i])-Ml)**2 for i in range(l_n)])/(l_n-1))
                    CV_l = Ml/l_standard_deviation
                else: 
                    Ml = 0
                    l_standard_deviation = 0
                    CV_l = 0
                
                Ml_list.append(Ml)
                SDl_list.append(l_standard_deviation)
                CVl_list.append(CV_l) 

                if d_n: 
                    Md = sum([float(degs[i])*d_counts[i]/d_n for i in range(d_n)])
                    d_standard_deviation = sqrt(sum([(float(degs[i])-Md)**2 for i in range(d_n)])/(d_n-1))
                    CV_d = Md/d_standard_deviation
                else:
                    Md = 0
                    d_standard_deviation = 0
                    CV_d = 0

                Md_list.append(Md)
                SDd_list.append(d_standard_deviation)
                CVd_list.append(CV_d) 

                # for ASIFT
                a_lens = a_length_hist['length']
                a_l_counts = a_length_hist['count']
                a_l_n = a_l_counts.size

                a_degs = a_degree_hist['degree']
                a_d_counts = a_degree_hist['count']
                a_d_n = a_d_counts.size

                if a_l_n: 
                    a_Ml = sum([float(a_lens[i])*a_l_counts[i]/a_l_n for i in range(a_l_n)])
                    a_l_standard_deviation = sqrt(sum([(float(a_lens[i])-a_Ml)**2 for i in range(a_l_n)])/(a_l_n-1))
                    a_CV_l = a_Ml/a_l_standard_deviation
                else: 
                    a_Ml = 0
                    a_l_standard_deviation = 0
                    a_CV_l = 0
                
                a_Ml_list.append(a_Ml)
                a_SDl_list.append(a_l_standard_deviation)
                a_CVl_list.append(a_CV_l) 

                if a_d_n: 
                    a_Md = sum([float(a_degs[i])*a_d_counts[i]/a_d_n for i in range(a_d_n)])
                    a_d_standard_deviation = sqrt(sum([(float(a_degs[i])-a_Md)**2 for i in range(a_d_n)])/(a_d_n-1))
                    a_CV_d = a_Md/a_d_standard_deviation
                else:
                    a_Md = 0
                    a_d_standard_deviation = 0
                    a_CV_d = 0

                a_Md_list.append(a_Md)
                a_SDd_list.append(a_d_standard_deviation)
                a_CVd_list.append(a_CV_d) 
                
                coord_j += step
                jt+=1
            coord_i += step
            it+=1

        # ------------------VISUALISATION------------------
        figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 10))

        axes[0, 0].set_title("Mathematical expectation SIFT")
        axes[0, 0].set_xlabel("length")
        axes[0, 0].set_ylabel("degree")
        axes[0, 0].scatter(Ml_list, Md_list, alpha = 0.2)

        axes[0, 1].set_title("Standard deviation SIFT")
        axes[0, 1].set_xlabel("length")
        axes[0, 1].set_ylabel("degree")
        axes[0, 1].scatter(SDl_list, SDd_list, alpha = 0.2)

        axes[0, 2].set_title("Coefficient of variation SIFT")
        axes[0, 2].set_xlabel("length")
        axes[0, 2].set_ylabel("degree")
        axes[0, 2].scatter(CVl_list, CVd_list, alpha = 0.2)

        axes[1, 0].set_title("Mathematical expectation A-SIFT")
        axes[1, 0].set_xlabel("length")
        axes[1, 0].set_ylabel("degree")
        axes[1, 0].scatter(Ml_list, Md_list, alpha = 0.2)

        axes[1, 1].set_title("Standard deviation A-SIFT")
        axes[1, 1].set_xlabel("length")
        axes[1, 1].set_ylabel("degree")
        axes[1, 1].scatter(SDl_list, SDd_list, alpha = 0.2)

        axes[1, 2].set_title("Coefficient of variation A-SIFT")
        axes[1, 2].set_xlabel("length")
        axes[1, 2].set_ylabel("degree")
        axes[1, 2].scatter(CVl_list, CVd_list, alpha = 0.2)

        # plt.tight_layout()
        plt.show()


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


    def search_right_extremum(self, coords, extrema):
        x, y = coords
        i=1
        for x_, y_ in extrema:
            if sqrt(abs(x-x_)**2 + abs(y-y_)**2) <= self.MAP_SLICE*0.3:
                return i
            else: i+=1
        return 0


if __name__ == "__main__":
    experiment = Experiment()
    experiment.experiment()