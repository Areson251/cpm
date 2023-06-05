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
    EXTREMA_COUNT = 5
    MAX_DEGREE = 1
    TEMPLATE_COUNT = 2

    experiment = Experiment(IMAGE_1_PATH, IMAGE_2_PATH, TEMPLATE_COUNT, MAP_SLICE, STEP, DIFFERENCE, EXTREMA_COUNT, MAX_DEGREE)
    experiment.experiment_KORR()
    experiment.experiment_SIFT("SIFT")
    experiment.experiment_SIFT("A-SIFT")
