import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img = cv2.imread("./IP_dog.bmp")

# Original Image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Trapezoidal Transformation
rows, cols = img_tmp.shape
new_img = np.zeros(img_tmp.shape, dtype=img_tmp.dtype)

for i in range(rows):
    for j in range(cols):
        new_x = int(np.round(3*i/4 + j*i/(cols*rows)))
        new_y = int(np.round(j+i/4 - j*i/(2*cols)))
        new_img[new_x][new_y] = img_tmp[i][j]


# Wavy Transformation
rows, cols = img_tmp.shape
new_img = np.zeros(img_tmp.shape, dtype=img_tmp.dtype)

for i in range(rows):
    for j in range(cols):
        new_x = int(np.round(j - 32*np.sin(i/32)))
        new_y = int(np.round(i - 32*np.sin(j/32)))
        if new_x >= 0 and new_x <= rows-1 and new_y >= 0 and new_y <= cols-1:
            new_img[j][i] = img_tmp[new_x][new_y]

# Circular Transformation
rows, cols = img_tmp.shape
new_img = np.zeros(img_tmp.shape, dtype=img_tmp.dtype)

for i in range(cols):
    for j in range(rows):
        d = np.sqrt((rows/2)**2 - (rows/2 - i)**2)
        new_x = np.round((j - cols/2)*cols/(d*2) + cols/2)
        new_y = i
        if new_x >= 0 and new_x <= cols-1 and new_y >= 0 and new_y <= cols-1:
            new_img[i][j] = img_tmp[new_y][int(new_x)]

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(new_img, cmap="gray")
plt.show()
