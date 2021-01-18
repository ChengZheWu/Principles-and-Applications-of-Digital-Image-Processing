from scipy.signal import gaussian
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def motiom_blur(img, a, b, T):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    M = dft.shape[0]
    N = dft.shape[1]
    H = np.zeros((M, N, 2))

    for u in range(M):
        for v in range(N):
            x = np.pi*(u*a + v*b)
            if x == 0:
                H[u][v] = 0
            else:
                number = (T*np.sin(x)*np.exp(-1j*x)) / x
                H[u][v][0] = np.real(number)
                H[u][v][1] = np.imag(number)

    fshift = dft_shift*H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def Inverse_filter(img, k):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    M = dft.shape[0]
    N = dft.shape[1]
    H = np.zeros((M, N, 2))

    for u in range(M):
        for v in range(N):
            H[u][v] = np.exp((-k)*(((u + M/2)**2 + (v - N/2)**2)**(5/6)))

    fshift = dft_shift*H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def add_gaussian_noise(img, sigma):
    gauss = np.random.normal(0, sigma, np.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy


def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


if __name__ == "__main__":
    img = cv2.imread("./C1HW04_IMG02_2020.bmp")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mb = motiom_blur(img, 0.1, 0.1, 1)
    img_gaussain_noise = add_gaussian_noise(img_mb, 20)
    img_inv = Inverse_filter(img_mb, 0.00000001)
    kernel = gaussian_kernel(5)
    img_w = wiener_filter(img_gaussain_noise, kernel, 10)

    plt.subplot(131)
    plt.imshow(img, cmap="gray")
    plt.subplot(132)
    plt.imshow(img_gaussain_noise, cmap="gray")
    plt.subplot(133)
    plt.imshow(img_w, cmap="gray")
    plt.show()
