from gui3 import Ui_Form
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import matplotlib


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


def segmentation(img, K=2):
    img_tmp = img.reshape((img.shape[0]*img.shape[1], 3))
    img_tmp = img_tmp.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        img_tmp, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    new_img = label.reshape((img.shape[0], img.shape[1]))
    return new_img


def plot(img, K=2):
    new_img = segmentation(img, K)
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.subplot(122)
    plt.imshow(new_img, cmap="gray")
    plt.show()


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()  # 新增做好的前端介面
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.load_data)
        self.ui.pushButton_2.clicked.connect(self.plot_img)
        self.ui.pushButton_3.clicked.connect(self.choose_K)
        self.ui.radioButton.toggled.connect(self.RGB_img)
        self.ui.radioButton_2.toggled.connect(self.HSI_img)
        self.ui.radioButton_3.toggled.connect(self.Lab_img)
        self.show()  # 顯示gui視窗

    def load_data(self):
        filename, filetype = QFileDialog.getOpenFileName(
            self, "開啟檔案", "./", "(*.bmp);;(*.jpg);;(*.jpeg)")
        if filename != "":
            self.ui.label.setText(filename)  # 選好檔案後，把label改成檔案名稱
            self.original_img = cv2.imread(filename)  # 路徑不能為中文
            self.original_img = cv2.cvtColor(
                self.original_img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
            self.img = self.original_img.copy()
            self.K = None

    def RGB_img(self):
        self.img = self.original_img

    def HSI_img(self):
        self.img = RGB2HSI(self.original_img)

    def Lab_img(self):
        self.img = RGB2Lab(self.original_img)

    def choose_K(self):
        text = int(self.ui.lineEdit.text())
        self.K = text

    def plot_img(self):
        if self.K == None:
            plot(self.img)
        else:
            plot(self.img, self.K)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()  # 顯示w
    sys.exit(app.exec_())  # 使用exit或點關閉按鈕退出
