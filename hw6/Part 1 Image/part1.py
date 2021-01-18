from gui1 import Ui_Form
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import matplotlib
import math


def Trapezoidal_Transform(img):
    rows, cols = img.shape
    new_img = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            new_x = int(np.round(3*i/4 + j*i/(cols*rows)))
            new_y = int(np.round(j+i/4 - j*i/(2*cols)))
            new_img[new_x][new_y] = img[i][j]
    return new_img


def Wavy_Transform(img):
    rows, cols = img.shape
    new_img = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            new_x = int(np.round(j - 32*np.sin(i/32)))
            new_y = int(np.round(i - 32*np.sin(j/32)))
            if new_x >= 0 and new_x <= rows-1 and new_y >= 0 and new_y <= cols-1:
                new_img[j][i] = img[new_x][new_y]
    return new_img


def Circular_Transform(img):
    rows, cols = img.shape
    new_img = np.zeros(img.shape, dtype=img.dtype)

    for i in range(cols):
        for j in range(rows):
            d = np.sqrt((rows/2)**2 - (rows/2 - i)**2)
            new_x = np.round((j - cols/2)*cols/(d*2) + cols/2)
            new_y = i
            if new_x >= 0 and new_x <= cols-1 and new_y >= 0 and new_y <= cols-1:
                new_img[i][j] = img[new_y][int(new_x)]
    return new_img


def plot(img, gray=True):
    if gray == True:
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(img)
        plt.axis("off")
        plt.show()


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()  # 新增做好的前端介面
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.load_data)
        self.ui.pushButton_2.clicked.connect(self.plot_img)
        self.ui.radioButton.toggled.connect(self.ori_img)
        self.ui.radioButton_2.toggled.connect(self.tra_img)
        self.ui.radioButton_3.toggled.connect(self.wav_img)
        self.ui.radioButton_4.toggled.connect(self.cir_img)
        self.show()  # 顯示gui視窗

    def load_data(self):
        filename, filetype = QFileDialog.getOpenFileName(
            self, "開啟檔案", "./", "(*.bmp);;(*.jpg);;(*.jpeg)")
        if filename != "":
            self.ui.label.setText(filename)  # 選好檔案後，把label改成檔案名稱
            self.original_img = cv2.imread(filename)  # 路徑不能為中文
            self.original1_img = cv2.cvtColor(
                self.original_img, cv2.COLOR_BGR2RGB)
            self.original2_img = cv2.cvtColor(
                self.original_img, cv2.COLOR_BGR2GRAY)
            self.img = self.original2_img

    def ori_img(self):
        self.img = self.original1_img

    def tra_img(self):
        self.img = Trapezoidal_Transform(self.original2_img)

    def wav_img(self):
        self.img = Wavy_Transform(self.original2_img)

    def cir_img(self):
        self.img = Circular_Transform(self.original2_img)

    def plot_img(self):
        if len(self.img.shape) == 2:
            plot(self.img)
        else:
            plot(self.img, False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()  # 顯示w
    sys.exit(app.exec_())  # 使用exit或點關閉按鈕退出
