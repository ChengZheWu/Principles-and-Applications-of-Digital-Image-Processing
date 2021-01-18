from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import cv2


def mask_operator(img, size=3):
    kernel = np.ones((size, size))
    kernel_sum = np.sum(kernel)
    filter_kernel = kernel / kernel_sum

    m, n = filter_kernel.shape

    padding_y = (m - 1)//2
    padding_x = (n - 1)//2

    # zero padding
    padding_img = cv2.copyMakeBorder(
        img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, value=0)

    y, x = padding_img.shape

    y_out = y - m + 1
    x_out = x - n + 1

    # convolution
    new_img = np.zeros((y_out, x_out))
    for i in range(y_out):
        for j in range(x_out):
            new_img[i][j] = np.sum(padding_img[i:i+m, j:j+n]*filter_kernel)

    new_img = new_img.astype(np.uint8)

    computation_time = (m*n)/(m + n)

    return new_img, computation_time

# def Marr_hildreth():


def Marr_Hildreth(img, sigma):
    size = int(2*(np.ceil(3*sigma))+1)

    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # LoG filter

    kern_size = kernel.shape[0]
    img_LoG = np.zeros_like(img, dtype=float)

    # applying filter
    for i in range(img.shape[0]-(kern_size-1)):
        for j in range(img.shape[1]-(kern_size-1)):
            window = img[i:i+kern_size, j:j+kern_size] * kernel
            img_LoG[i, j] = np.sum(window)

    img_LoG = img_LoG.astype(np.int64, copy=False)

    zero_crossing = np.zeros_like(img_LoG)

    # computing zero crossing
    for i in range(img_LoG.shape[0]-(kern_size-1)):
        for j in range(img_LoG.shape[1]-(kern_size-1)):
            if img_LoG[i][j] == 0:
                if (img_LoG[i][j-1] < 0 and img_LoG[i][j+1] > 0) or (img_LoG[i][j-1] < 0 and img_LoG[i][j+1] < 0) or (img_LoG[i-1][j] < 0 and img_LoG[i+1][j] > 0) or (img_LoG[i-1][j] > 0 and img_LoG[i+1][j] < 0):
                    zero_crossing[i][j] = 255
            if img_LoG[i][j] < 0:
                if (img_LoG[i][j-1] > 0) or (img_LoG[i][j+1] > 0) or (img_LoG[i-1][j] > 0) or (img_LoG[i+1][j] > 0):
                    zero_crossing[i][j] = 255

    return zero_crossing


def median_filter(img, filter_size):
    padding = (filter_size - 1)//2

    # zero padding
    padding_img = cv2.copyMakeBorder(
        img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    y, x = padding_img.shape

    y_out = y - filter_size + 1
    x_out = x - filter_size + 1

    # convolution
    new_img = np.zeros((y_out, x_out))
    for i in range(y_out):
        for j in range(x_out):
            sort_array = sorted(
                padding_img[i:i+filter_size, j:j+filter_size].flatten())
            new_img[i][j] = np.median(sort_array)

    new_img = new_img.astype(np.uint8)

    return new_img


def max_filter(img, filter_size):
    padding = (filter_size - 1)//2

    # zero padding
    padding_img = cv2.copyMakeBorder(
        img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    y, x = padding_img.shape

    y_out = y - filter_size + 1
    x_out = x - filter_size + 1

    # convolution
    new_img = np.zeros((y_out, x_out))
    for i in range(y_out):
        for j in range(x_out):
            sort_array = sorted(
                padding_img[i:i+filter_size, j:j+filter_size].flatten())
            new_img[i][j] = np.max(sort_array)

    new_img = new_img.astype(np.uint8)

    return new_img


def min_filter(img, filter_size):
    padding = (filter_size - 1)//2

    # zero padding
    padding_img = cv2.copyMakeBorder(
        img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    y, x = padding_img.shape

    y_out = y - filter_size + 1
    x_out = x - filter_size + 1

    # convolution
    new_img = np.zeros((y_out, x_out))
    for i in range(y_out):
        for j in range(x_out):
            sort_array = sorted(
                padding_img[i:i+filter_size, j:j+filter_size].flatten())
            new_img[i][j] = np.min(sort_array)

    new_img = new_img.astype(np.uint8)

    return new_img


img = cv2.imread("./Image 3-2.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img1, computation_time = mask_operator(img, 20)

# img2 = Marr_Hildreth(img, 1)
# img3 = cv2.Sobel(img, -1, dx=1, dy=0, ksize=3)

# print(img.shape)
# img3 = median_filter(img, 3)
# img4 = max_filter(img, 3)
# img5 = min_filter(img, 3)

plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.subplot(122)
plt.imshow(img1, cmap="gray")
plt.show()
