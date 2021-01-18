import numpy as np
import cv2
import matplotlib.pyplot as plt


def grayA(img):
    img = img.astype(np.float32)
    img = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
    img = img.astype(np.uint8)[:, :, np.newaxis]
    return img


def grayB(img):
    img = img.astype(np.float32)
    img = img[:, :, 0]*0.299 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114
    img = img.astype(np.uint8)[:, :, np.newaxis]
    return img


def binary(img, threshold):
    img = img.astype(np.float32)
    img[img > threshold] = 255
    img[img <= threshold] = 0
    img = img.astype(np.uint8)
    return img


def rescale(img, factor):
    size = img.shape[0]
    rescale_factor = factor
    rescale_size = int(size*rescale_factor)
    img_rescale = cv2.resize(img, (rescale_size, rescale_size),
                             interpolation=cv2.INTER_LINEAR)  # 雙線性插植
    return img_rescale


def brightness(img, factor):
    img = img.astype(np.float32)
    img += factor
    img[img >= 255] = 255
    img[img <= 0] = 0
    img = img.astype(np.uint8)
    return img


def constrast(img, factor):
    img = img.astype(np.float32)
    img *= factor
    img[img >= 255] = 255
    img[img <= 0] = 0
    img = img.astype(np.uint8)
    return img


def equalized(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = ((cdf_m - cdf_m.min())*255) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
    img_equal = cdf[img]
    return img_equal


img = cv2.imread("./dog.jpeg")  # order of color is BGR
# order of color is BGR
img = cv2.imread("C:/Users/Zhe/Desktop/Image_Processing/hw2/dog.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cover BGR to RGB


img_grayA = grayA(img)
img_grayB = grayB(img)
img_gray_diff = img_grayA - img_grayB

img_binary = binary(img, 100)

img_rescale = rescale(img, 1.5)

img_brightness = brightness(img, 100)

img_constrast = constrast(img, 2)

img_equal = equalized(img)


plt.imshow(img_gray_diff, cmap="gray")
plt.show()
