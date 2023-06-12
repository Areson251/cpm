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

from experiment import Experiment
from imageData import ImageData

if __name__ == "__main__":
    IMAGE_1_PATH = 'photos/maps/yandex.jpg'
    IMAGE_2_PATH = 'photos/maps/google.jpg'
    MAP_SLICE = 301
    STEP = int(MAP_SLICE * 0.25)
    DIFFERENCE = 1
    EXTREMA_COUNT = 10
    MAX_DEGREE = 30
    TEMPLATE_COUNT = 10
    EXPERIMENT_COUNT = 10

    experiment = Experiment(IMAGE_1_PATH, IMAGE_2_PATH, TEMPLATE_COUNT, MAP_SLICE, STEP, DIFFERENCE, EXTREMA_COUNT, MAX_DEGREE)

    for i in range(2, EXPERIMENT_COUNT):
        experiment.experiment_KORR(exp_number=i)
        experiment.experiment_SIFT("A-SIFT", exp_number=i)
    for i in range(1, EXPERIMENT_COUNT):
        experiment.experiment_SIFT("SIFT", exp_number=i)
