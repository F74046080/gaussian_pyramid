import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv2.imread('hw2_data/task1and2_hybrid_pyramid/bicycle.bmp', 0)
# cv2.imshow('bike', img)

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]])/256

# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# # cv2.imshow('bike', magnitude_spectrum)
# plt.imshow(magnitude_spectrum, cmap='gray')
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
def convolution(image, kernel):
    image_padded = np.zeros((image.shape[0] + 4, image.shape[1] + 4))
    image_padded[2:-2, 2:-2] = image
    out = np.zeros_like(image)
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            out[y, x] = (kernel * image_padded[y:y + 5, x:x + 5]).sum()
    return out


def gaussian_pyramid(img, kernel):
    temp = convolution(img, kernel)
    row, col = img.shape
    m = math.floor(row/2)
    n = math.floor(col/2)
    pyramid = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            pyramid[i, j] = temp[2*i, 2*j]
    return pyramid
