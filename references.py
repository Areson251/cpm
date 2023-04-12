# from turtle import width
from ctypes.wintypes import SHORT
import cv2
import numpy as np
import time, datetime
import math
import random
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from comparison import *

EXPERIMENT_COUNT = 30
IMAGE_1_PATH = 'photos/maps/yandex.jpg'
IMAGE_2_PATH = 'photos/pictures/g_cropped.jpg'
PIXELS_STEP = 51
MAP_SLICE = 501
SHAPE = 10