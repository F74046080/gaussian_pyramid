import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# # cv2.imshow('bike', magnitude_spectrum)
# plt.imshow(magnitude_spectrum, cmap='gray')
# plt.show()

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

def expand(img, k, f):
    row, col = img.shape
    tmp = np.zeros((row*2, col*2))

    for i in range(row):
        for j in range(col):
            tmp[2*i+1, 2*j+1] = img[i][j]
            tmp[2*i, 2*j] = 0

    if f == 1:
        tmp = np.row_stack((tmp, np.zeros(col*2)))
    elif f == 2:
        tmp = np.column_stack((tmp, np.zeros(row*2)))
    elif f == 3:
        tmp = np.row_stack((tmp, np.zeros(col*2)))
        tmp = np.column_stack((tmp, np.zeros(row*2+1)))
    return 4*convolution(tmp, k)

def laplace_pyramid(img, kernel):
    pydown = gaussian_pyramid(img, kernel)
    flag = 0
    if pydown.shape[0] % 2 == 0 and pydown.shape[1] % 2 == 0:
        flag = 0
    elif pydown.shape[0] % 2 == 1 and pydown.shape[1] % 2 == 0:
        flag = 1
    elif pydown.shape[0] % 2 == 0 and pydown.shape[1] % 2 == 1:
        flag = 2
    else:
        flag = 3
    pyup = expand(gaussian_pyramid(pydown, kernel), kernel, flag)
    # print(pydown.shape)
    # print(pyup.shape)
    out = cv2.subtract(pydown, pyup)
    return out

def plot_magnitude_spectrum(images):
    for i in range(len(images)):
        f = np.fft.fft2(images[i])
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        plt.subplot(5, 1, i+1, aspect='auto')
        plt.imshow(magnitude_spectrum, cmap='gray')
        # cv2.imwrite('sg'+str(i)+'.jpg', magnitude_spectrum)
    plt.show()

# img = cv2.imread('hw2_data/task1and2_hybrid_pyramid/motorcycle.bmp', 0)
# img = cv2.imread('images/pyramids_Gray.jpg',0)
# cv2.imshow('bike', img)

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]])/256
g = []
l = []
tmp = img
for i in range(5):
    gauss = gaussian_pyramid(tmp, kernel)
    g.append(gauss)
    # tmp = gauss
    lap = laplace_pyramid(tmp, kernel)
    l.append(lap)
    tmp = gauss

plt.figure('pyramid')
for i in range(5):
    # cv2.imshow('g' +str(i), g[i].astype('uint8') * 255)
    # cv2.imshow('l' +str(i), l[i].astype('uint8') * 255)
    
    plt.subplot(5, 2, (i+1)*2, aspect='auto')
    plt.imshow(cv2.cvtColor(np.uint8(g[i]), cv2.COLOR_BGR2RGB))
    # cv2.imwrite('g'+str(i)+'.jpg',g[i])

    plt.subplot(5, 2, (i+1)*2-1, aspect='auto')
    plt.imshow(cv2.cvtColor(np.uint8(l[i]), cv2.COLOR_BGR2RGB))
    # cv2.imwrite('l'+str(i)+'.jpg', l[i])

plt.show()
plt.figure('gaussian_spectrum')
plot_magnitude_spectrum(g)
plt.figure('laplacian_spectrum')
plot_magnitude_spectrum(l)

# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
