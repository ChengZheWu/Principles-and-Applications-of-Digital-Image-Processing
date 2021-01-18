from gui2 import Ui_Form
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import matplotlib


def plot(img, colormap="Reds"):
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(img, cmap=colormap)
    plt.colorbar()
    plt.show()


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()  # 新增做好的前端介面
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.load_data)
        self.ui.pushButton_2.clicked.connect(self.plot_img)
        self.ui.pushButton_3.clicked.connect(self.choose_color)
        self.show()  # 顯示gui視窗

    def load_data(self):
        filename, filetype = QFileDialog.getOpenFileName(
            self, "開啟檔案", "./", "(*.bmp);;(*.jpg);;(*.jpeg)")
        if filename != "":
            self.ui.label.setText(filename)  # 選好檔案後，把label改成檔案名稱
            self.original_img = cv2.imread(filename)  # 路徑不能為中文
            self.original_img = cv2.cvtColor(
                self.original_img, cv2.COLOR_BGR2GRAY)  # convert BGR to GRAY
            self.img = self.original_img
            self.color = None

    def choose_color(self):
        text = str(self.ui.lineEdit.text())
        self.color = text

    def plot_img(self):
        if self.color == None:
            plot(self.img)
        else:
            plot(self.img, self.color)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()  # 顯示w
    sys.exit(app.exec_())  # 使用exit或點關閉按鈕退出
