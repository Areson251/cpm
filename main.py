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
    TEMPLATE_COUNT = 10
    MAP_SLICE = 301
    EXPERIMENT_COUNT = 31
    EXTREMA_COUNT = 10
    experiment = Experiment(IMAGE_1_PATH, IMAGE_2_PATH, TEMPLATE_COUNT, MAP_SLICE, EXPERIMENT_COUNT, EXTREMA_COUNT)
    experiment.experiment()