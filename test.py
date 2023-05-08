import numpy as np
import cv2
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

fname = 'photos/results/result_12212.jpg'
neighborhood_size = 5
threshold = 1500

data = cv2.imread(fname)
# data = scipy.misc.imread(fname)

data_max = ndimage.maximum_filter(data, neighborhood_size)
maxima = (data == data_max)
data_min = ndimage.minimum_filter(data, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

labeled, num_objects = ndimage.label(maxima)
slices = ndimage.find_objects(labeled)
x, y = [], []
for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2
    x.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2    
    y.append(y_center)

# plt.show()
# plt.savefig('data.jpg', bbox_inches = 'tight')

# plt.autoscale(False)
plt.plot(x,y, 'ro')
plt.show()
# plt.savefig('/tmp/result.png', bbox_inches = 'tight')