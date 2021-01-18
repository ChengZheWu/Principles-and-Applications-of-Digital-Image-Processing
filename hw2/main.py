from gui import Ui_Form
from PyQt5.QtWidgets import *
import sys
import cv2
import numpy as np
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


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.factor = 0
        self.ui.setupUi(self)
        self.ui.lineEdit.setText("1")
        text = self.ui.lineEdit.text()
        self.factor = float(text)
        self.ui.pushButton.clicked.connect(self.load_data)
        self.ui.pushButton_2.clicked.connect(self.show_img)
        self.ui.pushButton_3.clicked.connect(self.plot_hist)
        self.ui.pushButton_4.clicked.connect(self.change)
        self.ui.radioButton.toggled.connect(self.to_original)
        self.ui.radioButton_2.toggled.connect(self.to_grayA)
        self.ui.radioButton_3.toggled.connect(self.to_grayB)
        self.ui.radioButton_4.toggled.connect(self.to_binary)
        self.ui.radioButton_5.toggled.connect(self.to_rescale)
        self.ui.radioButton_6.toggled.connect(self.to_constrast)
        self.ui.radioButton_7.toggled.connect(self.to_brightness)
        self.ui.radioButton_8.toggled.connect(self.to_equal)
        self.show()  # 顯示gui視窗

    def load_data(self):
        filename, filetype = QFileDialog.getOpenFileName(
            self, "開啟檔案", "./", "(*.jpeg);;(*.bmp)")
        if filename != "":
            self.ui.label.setText(filename)  # 選好檔案後，把label改成檔案名稱
            print(filename)
            self.original_img = cv2.imread(filename)  # 路徑不能為中文
            self.original_img = cv2.cvtColor(
                self.original_img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB

    def change(self):
        text = self.ui.lineEdit.text()
        self.factor = float(text)

    def to_original(self):
        self.img = self.original_img

    def to_grayA(self):
        self.img = grayA(self.original_img)

    def to_grayB(self):
        self.img = grayB(self.original_img)

    def to_binary(self):
        self.img = binary(self.original_img, self.factor)

    def to_rescale(self):
        self.img = rescale(self.original_img, self.factor)

    def to_brightness(self):
        self.img = brightness(self.original_img, self.factor)

    def to_constrast(self):
        self.img = constrast(self.original_img, self.factor)

    def to_equal(self):
        self.img = equalized(self.original_img)

    def show_img(self):
        if self.img.shape[2] == 1:
            plt.imshow(self.img, cmap="gray")
            plt.show()
        else:
            plt.imshow(self.img)
            plt.show()

    def plot_hist(self):
        plt.hist(self.img.flatten(), bins=256)
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()  # 顯示w
    sys.exit(app.exec_())  # 使用exit或點關閉按鈕退出
