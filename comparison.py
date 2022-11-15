# from turtle import width
from ctypes.wintypes import SHORT
import cv2
import numpy as np
import time, datetime
import math

IMAGE_1_PATH = 'photos/1_yandex.png'
IMAGE_2_PATH = 'photos/3_.png'
PIXELS_STEP = 101

SHAPE = 10


def count_difference(image1, image2):
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
    
    return np.array(image_pixels)


def count_difference_with_step(image1, image2, step):
    # width, height = count_shapes(image1, image2)
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
                    sum += 255 - abs(image1.item((i_num + i, j_num + j)) - image2.item((i, j))) 
            pixel = sum / (image2.shape[0] * image2.shape[1])
            # pixels_row.append() 
            pixels_row = np.append(pixels_row, round(pixel))
            j_num+=step  
        image_pixels = np.append(image_pixels, pixels_row, axis=0)  
        i_num += step
        kol_steps += 1
        print(f"STEP NUMBER: {kol_steps}")
    
    min = np.amin(image_pixels)
    max = np.amax(image_pixels)
    
    return 255 * (image_pixels - min)//(max-min)


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


if __name__ == "__main__":
    init_time = time.time()

    # Load an color image in grayscale
    image1 = cv2.imread(IMAGE_1_PATH,0)
    image2 = cv2.imread(IMAGE_2_PATH,0)

    # pixels = count_difference(image1, image2)
    # pixels = create_convolution(image1, image2, PIXELS_STEP)
    pixels = count_difference_with_step(image1, image2, PIXELS_STEP)

    # pixels = create_image(pixels)
    
    print(f"SECONDS SPENT: {time.time() - init_time}")
    # show image
    cv2.imshow('result', pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(f'photos/result_{SHAPE}.png', pixels)
