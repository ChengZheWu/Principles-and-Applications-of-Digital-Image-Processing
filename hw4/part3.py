import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def Homomorphic_filter(img, D0, rH, rL, c):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    P = dft.shape[0]
    Q = dft.shape[1]
    H = np.zeros((P, Q, 2))

    for u in range(P):
        for v in range(Q):
            distance = ((u - P/2)**2 + (v - Q/2)**2)**(0.5)
            H[u][v] = (rH - rL)*(1 - np.exp(-((c*(distance**2) / D0)))) + rL

    fshift = dft_shift*H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


if __name__ == "__main__":
    img = cv2.imread("./C1HW04_IMG01_2020.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c = 5
    D0 = int(input("D0 = "))
    rH = float(input("rH = "))
    rL = float(input("rL = "))
    img_homo = Homomorphic_filter(img, D0, rH, rL, 5)

    plt.imshow(img_homo, cmap="gray")
    plt.show()

    # img1 = Homomorphic_filter(img, 20, 3, 0.4, c)
    # img2 = Homomorphic_filter(img, 200, 3, 0.4, c)
    # img3 = Homomorphic_filter(img, 2000, 3, 0.4, c)

    # plt.subplot(131)
    # plt.title("D0 = 20")
    # plt.imshow(img1, cmap="gray")
    # plt.subplot(132)
    # plt.title("D0 = 200")
    # plt.imshow(img2, cmap="gray")
    # plt.subplot(133)
    # plt.title("D0 = 2000")
    # plt.imshow(img3, cmap="gray")
    # plt.show()
