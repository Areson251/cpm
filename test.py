import numpy as np
import cv2
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

fname = 'photos/maps/yandex.jpg'


# load the image and show it
image = cv2.imread(fname)
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# grab the dimensions of the image and calculate the center of the
# image
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
# rotate our image by 45 degrees around the center of the image
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# rotate our image by -90 degrees around the image
M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -90 Degrees", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows() 