import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('datasets/partA/train_data/images/IMG_1.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)
plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.imshow(img,cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(edges,cmap = 'gray')
plt.savefig('canny.png')