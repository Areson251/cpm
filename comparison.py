import cv2

IMAGE_1_PATH = 'photos/1.png'
IMAGE_2_PATH = 'photos/2.png'

if __name__ == "__main__":
    # Load an color image in grayscale
    image1 = cv2.imread(IMAGE_1_PATH,0)
    image2 = cv2.imread(IMAGE_2_PATH,0)
    
    # show image
    cv2.imshow('image',image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
