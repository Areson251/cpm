import cv2
import numpy as np

IMAGE_1_PATH = 'photos/1.png'
IMAGE_2_PATH = 'photos/3.png'


def count_difference(image1, image2):
    width = round(image1.shape[1] / image2.shape[1])
    height = round(image1.shape[0] / image2.shape[0])

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


#TODO: rename function to correct 
def create_hist(pixels):
    avarage = int(np.sum(pixels) / (pixels.shape[0] * pixels.shape[1]))
    pixels[pixels < avarage] = 0
    
    return pixels


if __name__ == "__main__":
    # Load an color image in grayscale
    image1 = cv2.imread(IMAGE_1_PATH,0)
    image2 = cv2.imread(IMAGE_2_PATH,0)

    pixels = count_difference(image1, image2)

    pixels = create_hist(pixels)
    
    # show image
    cv2.imshow('result',np.uint8(pixels))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
