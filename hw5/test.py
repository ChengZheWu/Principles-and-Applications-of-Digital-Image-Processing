import numpy as np
import matplotlib.pyplot as plt
import cv2

# part1


def RGB2CMY(img):
    new_img = np.zeros((img.shape))
    RGB_scale = 255.0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R = float(img[i, j, 0])
            G = float(img[i, j, 1])
            B = float(img[i, j, 2])

            C = 1 - R / RGB_scale
            M = 1 - G / RGB_scale
            Y = 1 - B / RGB_scale

            new_img[i, j, 0] = C
            new_img[i, j, 1] = M
            new_img[i, j, 2] = Y

    return new_img


def RGB2HSI(img):
    new_img = np.zeros((img.shape))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R = float(img[i, j, 0])
            G = float(img[i, j, 1])
            B = float(img[i, j, 2])

            fraction = ((R-G) + (R-B)) / 2
            denominator = np.sqrt((R-G)**2 + (R-B)*(G-B))
            if denominator == 0:
                H = 0
            else:
                theta = np.arccos(fraction / denominator)
                if B <= G:
                    H = theta
                else:
                    H = 360 - theta

            S = 1 - (3 / (R+G+B+1e-7))*np.min((R, G, B))
            I = (R+G+B) / 3

            new_img[i, j, 0] = H
            new_img[i, j, 1] = S*255
            new_img[i, j, 2] = I
    new_img = new_img.astype(np.uint8)
    return new_img


def RGB2XYZ(img):
    M = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]])
    new_img = np.zeros((img.shape))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R = float(img[i, j, 0])
            G = float(img[i, j, 1])
            B = float(img[i, j, 2])
            rgb = np.array([R, G, B])
            XYZ = np.dot(M, rgb.T)
            XYZ = XYZ / 255.0
            X = XYZ[0] / 0.95047
            Y = XYZ[1] / 1.0
            Z = XYZ[2] / 1.08883
            new_img[i, j, 0] = X
            new_img[i, j, 1] = Y
            new_img[i, j, 2] = Z

    return new_img


def h(q):
    return np.power(q, 1 / 3) if q > 0.008856 else 7.787 * q + 0.137931


def RGB2Lab(img):
    M = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]])
    new_img = np.zeros((img.shape))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R = float(img[i, j, 0])
            G = float(img[i, j, 1])
            B = float(img[i, j, 2])
            rgb = np.array([R, G, B])
            XYZ = np.dot(M, rgb.T)
            XYZ = XYZ / 255.0

            h_XYZ = [h(x) for x in XYZ]
            L = 116 * h_XYZ[1] - 16 if XYZ[1] > 0.008856 else 903.3 * XYZ[1]
            a = 500 * (h_XYZ[0] - h_XYZ[1])
            b = 200 * (h_XYZ[1] - h_XYZ[2])

            new_img[i, j, 0] = L
            new_img[i, j, 1] = a
            new_img[i, j, 2] = b

    return new_img


def RGB2YUV(img):
    m = np.array([[0.29900, -0.16874,  0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])
    new_img = np.zeros((img.shape))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R = float(img[i, j, 0])
            G = float(img[i, j, 1])
            B = float(img[i, j, 2])
            rgb = np.array([R, G, B])
            yuv = np.dot(rgb, m)
            yuv[1:] += 128.0

            Y = yuv[0]
            U = yuv[1]
            V = yuv[2]

            new_img[i, j, 0] = Y
            new_img[i, j, 1] = U
            new_img[i, j, 2] = V

    new_img = new_img.astype(np.uint8)
    return new_img


# img = cv2.imread("./HW05-Part 3-02.bmp")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# new_img = RGB2CMY(img)
# new_img = RGB2HSI(img)
# new_img = RGB2XYZ(img)
# new_img = RGB2Lab(img)
# new_img = RGB2YUV(img)

# plt.imshow(new_img)
# plt.show()


# part2

img = cv2.imread("./HW05-Part 2-01.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.subplot(122)
plt.imshow(img, cmap="jet")
plt.colorbar()
plt.show()

# part 3

img = cv2.imread("./HW05-Part 2-01.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def segmentation(img):
    img_tmp = img.reshape((img.shape[0]*img.shape[1], 3))
    img_tmp = img_tmp.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        img_tmp, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    new_img = label.reshape((img.shape[0], img.shape[1]))
    return new_img


new_img = segmentation(img)


plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.subplot(122)
plt.imshow(new_img, cmap="gray")
plt.show()
