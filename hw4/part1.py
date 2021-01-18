import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./C1HW04_IMG01_2020.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# part 1

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
phase = np.angle(dft_shift)[:, :, 0]

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


plt.subplot(221)
plt.title("original image")
plt.imshow(img, cmap="gray")
plt.subplot(222)
plt.title("image spectrum")
plt.imshow(spectrum, cmap="gray")
plt.subplot(223)
plt.title("image phase angle")
plt.imshow(phase, cmap="gray")
plt.subplot(224)
plt.title("inverse image")
plt.imshow(img_back, cmap="gray")
plt.show()

# compute FFT time


def compute(img, factor):
    size = img.shape[0]*factor
    img = np.resize(img, (size, size))

    start = time.time()
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    end = time.time()

    T = end - start

    return T


T1 = compute(img, 1)
T2 = compute(img, 2)
T3 = compute(img, 4)
T4 = compute(img, 8)

print("size X1: ", T1)
print("size X2: ", T2)
print("size X4: ", T3)
print("size X8: ", T4)
