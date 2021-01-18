import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui
from gui import Ui_MainWindow
from PyQt5 import QtWidgets
from function import *


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.load_data)
        self.ui.pushButton_7.clicked.connect(self.origin_img)
        self.ui.pushButton_2.clicked.connect(self.mask_operator)
        self.ui.pushButton_3.clicked.connect(self.edge_detection)
        self.ui.pushButton_4.clicked.connect(self.median_filter_img)
        self.ui.pushButton_5.clicked.connect(self.max_filter_img)
        self.ui.pushButton_6.clicked.connect(self.min_filter_img)

        self.size1 = 3
        self.ui.horizontalSlider.setMinimum(1)
        self.ui.horizontalSlider.setMaximum(20)
        self.ui.horizontalSlider.setValue(3)
        self.ui.horizontalSlider.valueChanged.connect(self.value_change_1)

        self.sigma = 1
        self.ui.horizontalSlider_2.setMinimum(0)
        self.ui.horizontalSlider_2.setMaximum(20)
        self.ui.horizontalSlider_2.setValue(10)
        self.ui.horizontalSlider_2.valueChanged.connect(self.value_change_2)

        self.size3 = 3
        self.ui.horizontalSlider_3.setMinimum(1)
        self.ui.horizontalSlider_3.setMaximum(20)
        self.ui.horizontalSlider_3.setValue(3)
        self.ui.horizontalSlider_3.valueChanged.connect(self.value_change_3)

        self.size4 = 3
        self.ui.horizontalSlider_4.setMinimum(1)
        self.ui.horizontalSlider_4.setMaximum(20)
        self.ui.horizontalSlider_4.setValue(3)
        self.ui.horizontalSlider_4.valueChanged.connect(self.value_change_4)

        self.size5 = 3
        self.ui.horizontalSlider_5.setMinimum(1)
        self.ui.horizontalSlider_5.setMaximum(20)
        self.ui.horizontalSlider_5.setValue(3)
        self.ui.horizontalSlider_5.valueChanged.connect(self.value_change_5)
        self.show()  # 顯示gui視窗

    def load_data(self):
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, "開啟檔案", "./", "(*.jpg);;(*.jpeg);;(*.bmp)")
        if filename != "":
            self.ui.label_6.setText(filename)  # 選好檔案後，把label改成檔案名稱
            self.original_img = cv2.imread(filename)  # 路徑不能為中文
            self.original_img = cv2.cvtColor(
                self.original_img, cv2.COLOR_BGR2GRAY)  # convert BGR to GRAY
            self.img = self.original_img

    def plot(self):
        plt.subplot(121)
        plt.imshow(self.original_img, cmap="gray")
        plt.subplot(122)
        plt.imshow(self.img, cmap="gray")
        plt.show()

    def origin_img(self):
        self.img = self.original_img
        plt.imshow(self.img, cmap="gray")
        plt.show()

    def value_change_1(self):
        size = self.ui.horizontalSlider.value()
        self.ui.label.setText("Size %d" % size)
        self.size1 = size

    def mask_operator(self):
        self.img, Computation_time = mask_operator(
            self.original_img, self.size1)
        self.plot()
        self.ui.label_9.setText(
            "Computation Time of Mask Operator :" + str(Computation_time))

    def value_change_2(self):
        sigma = self.ui.horizontalSlider_2.value()
        sigma = sigma/10.0
        self.ui.label_2.setText("Sigma %f" % sigma)
        self.sigma = sigma

    def edge_detection(self):
        self.img = Marr_Hildreth(self.original_img, self.sigma)
        self.plot()

    def value_change_3(self):
        size = self.ui.horizontalSlider_3.value()
        self.ui.label_3.setText("Size %d" % size)
        self.size3 = size

    def value_change_4(self):
        size = self.ui.horizontalSlider_4.value()
        self.ui.label_4.setText("Size %d" % size)
        self.size4 = size

    def value_change_5(self):
        size = self.ui.horizontalSlider_5.value()
        self.ui.label_5.setText("Size %d" % size)
        self.size5 = size

    def median_filter_img(self):
        self.img = median_filter(self.original_img, self.size3)
        self.plot()

    def max_filter_img(self):
        self.img = max_filter(self.original_img, self.size4)
        self.plot()

    def min_filter_img(self):
        self.img = min_filter(self.original_img, self.size5)
        self.plot()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
