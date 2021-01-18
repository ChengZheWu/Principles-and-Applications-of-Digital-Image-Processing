import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def Ideal_filter(img, D0, mode="L"):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    P = dft.shape[0]
    Q = dft.shape[1]
    H = np.zeros((P, Q, 2))

    if mode == "L":  # low pass
        for u in range(P):
            for v in range(Q):
                distance = ((u - P/2)**2 + (v - Q/2)**2)**(0.5)
                if distance <= D0:
                    H[u][v] = 1
                else:
                    H[u][v] = 0
    elif mode == "H":  # high pass
        for u in range(P):
            for v in range(Q):
                distance = ((u - P/2)**2 + (v - Q/2)**2)**(0.5)
                if distance <= D0:
                    H[u][v] = 0
                else:
                    H[u][v] = 1
    else:
        print("wrong mode")

    fshift = dft_shift*H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def Butterworth_filter(img, D0, n, mode="L"):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    P = dft.shape[0]
    Q = dft.shape[1]
    H = np.zeros((P, Q, 2))

    if mode == "L":  # low pass
        for u in range(P):
            for v in range(Q):
                distance = ((u - P/2)**2 + (v - Q/2)**2)**(0.5)
                H[u][v] = 1 / (1 + (distance/D0)**(2*n))
    elif mode == "H":  # high pass
        for u in range(P):
            for v in range(Q):
                distance = ((u - P/2)**2 + (v - Q/2)**2)**(0.5)
                if distance != 0:
                    H[u][v] = 1 / (1 + (D0/distance)**(2*n))
    else:
        print("wrong mode")

    fshift = dft_shift*H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def Gaussian_filter(img, D0, mode="L"):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    P = dft.shape[0]
    Q = dft.shape[1]
    H = np.zeros((P, Q, 2))

    if mode == "L":  # low pass
        for u in range(P):
            for v in range(Q):
                distance = ((u - P/2)**2 + (v - Q/2)**2)**(0.5)
                H[u][v] = np.exp(-((distance**2)/(2*(D0**2))))
    elif mode == "H":  # high pass
        for u in range(P):
            for v in range(Q):
                distance = ((u - P/2)**2 + (v - Q/2)**2)**(0.5)
                H[u][v] = 1 - np.exp(-((distance**2)/(2*(D0**2))))
    else:
        print("wrong mode")

    fshift = dft_shift*H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


if __name__ == "__main__":
    img = cv2.imread("./C1HW04_IMG01_2020.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img1 = Ideal_filter(img, 10)
    img2 = Ideal_filter(img, 30)
    img3 = Ideal_filter(img, 60)

    # img1 = Ideal_filter(img, 10, "H")
    # img2 = Ideal_filter(img, 30, "H")
    # img3 = Ideal_filter(img, 60, "H")

    # butterworth
    # img1 = Butterworth_filter(img, 10, 1)
    # img2 = Butterworth_filter(img, 30, 1)
    # img3 = Butterworth_filter(img, 60, 1)

    # img1 = Butterworth_filter(img, 30, 1)
    # img2 = Butterworth_filter(img, 30, 5)
    # img3 = Butterworth_filter(img, 30, 10)

    # img1 = Butterworth_filter(img, 10, 1, "H")
    # img2 = Butterworth_filter(img, 30, 1, "H")
    # img3 = Butterworth_filter(img, 60, 1, "H")

    # img1 = Butterworth_filter(img, 30, 1, "H")
    # img2 = Butterworth_filter(img, 30, 5, "H")
    # img3 = Butterworth_filter(img, 30, 10, "H")

    # img1 = Gaussian_filter(img, 10)
    # img2 = Gaussian_filter(img, 30)
    # img3 = Gaussian_filter(img, 60)

    # img1 = Gaussian_filter(img, 10, "H")
    # img2 = Gaussian_filter(img, 30, "H")
    # img3 = Gaussian_filter(img, 60, "H")

    # plt.imshow(img, cmap="gray")
    # plt.show()

    plt.subplot(131)
    plt.title("D0 = 10")
    plt.imshow(img1, cmap="gray")
    plt.subplot(132)
    plt.title("D0 = 30")
    plt.imshow(img2, cmap="gray")
    plt.subplot(133)
    plt.title("D0 = 60")
    plt.imshow(img3, cmap="gray")
    plt.show()
