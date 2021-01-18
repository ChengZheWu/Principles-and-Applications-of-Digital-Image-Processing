from gui import Ui_Form
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib


class plot(QDialog):
    def __init__(self, data_path):
        super().__init__()
        f = open(data_path, mode="r")
        data = f.read()
        f.close()
        self.original_img = self.img_generator(data)
        self.add_img = self.add(self.original_img)  # 4-1
        self.mul_img = self.multiply(self.original_img)  # 4-2
        self.mean_img = self.mean(self.add_img, self.mul_img)  # 4-3
        self.g_img = self.g_transform(self.original_img)  # 4-4

    def convert(self, data):
        """
        將32進位的值轉換成 0~255
        input : string
        output : 0~255 integer
        """
        list32 = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
        if data in list32:
            convert_list = {}
            num = 0
            dis = 256/31  # 有 32 個值要對應，所以切 31 等分
            convert_list[list32[0]] = num
            for i in range(1, 32):
                num += dis
                num_int = int(num)
                convert_list[list32[i]] = num_int
            return convert_list[data]
        else:
            return None

    def img_generator(self, data):
        """
        生成最後顯示的影像
        pixel value: 0-255
        image shape: 64x64
        """
        img = []
        for line in data.split("\n"):
            row = []
            for char in line:
                num = self.convert(char)
                if num != None:
                    img.append(num)
        img = np.array(img, dtype=np.uint8).reshape(64, 64)  # reshape圖片大小
        return img

    def add(self, img):
        img = img.astype(np.int32)  # 先轉換為32位元，以免overflow
        img += 100
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        return img

    def multiply(self, img):
        img = img.astype(np.float32)
        img *= 5
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        return img

    def mean(self, img1, img2):
        """
        retrun average image of two input images
        """
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        img = (img1 + img2)//2
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        return img

    def g_transform(self, img):
        img = img.astype(np.int32)
        new_img = np.empty((img.shape), dtype=np.int32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i == 0:
                    # 由於第一行沒有左邊的值，因此減去最後一行
                    new_img[i][j] = new_img[img.shape[0]-1][j]
                else:
                    new_img[i][j] = img[i][j] - img[i-1][j]
        img[img < 0] = 0
        img[img > 255] = 255
        new_img = new_img.astype(np.uint8)
        return new_img

    def plot_img(self, img):
        plt.imshow(img, cmap="gray")
        plt.show()

    def plot_histogram(self, img):
        plt.hist(img.flatten(), bins=64)
        plt.xlabel("pixel value")
        plt.ylabel("num")
        plt.show()


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()  # 新增做好的前端介面
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.load_data)
        self.ui.pushButton_2.clicked.connect(self.plot_img)
        self.ui.pushButton_3.clicked.connect(self.plot_histogram)
        self.ui.radioButton.toggled.connect(self.origin_img)
        self.ui.radioButton_2.toggled.connect(self.add_img)
        self.ui.radioButton_3.toggled.connect(self.mul_img)
        self.ui.radioButton_4.toggled.connect(self.mean_img)
        self.ui.radioButton_5.toggled.connect(self.g_img)
        self.show()  # 顯示gui視窗

    def load_data(self):
        filename, filetype = QFileDialog.getOpenFileName(
            self, "開啟檔案", "./", "Image files(*.64)")
        if filename != "":
            self.ui.label.setText(filename)  # 選好檔案後，把label改成檔案名稱
            self.data_path = filename
            self.plot = plot(self.data_path)

    def origin_img(self):
        self.img = self.plot.original_img

    def add_img(self):
        self.img = self.plot.add_img

    def mul_img(self):
        self.img = self.plot.mul_img

    def mean_img(self):
        self.img = self.plot.mean_img

    def g_img(self):
        self.img = self.plot.g_img

    def plot_img(self):
        self.plot.plot_img(self.img)

    def plot_histogram(self):
        self.plot.plot_histogram(self.img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()  # 顯示w
    sys.exit(app.exec_())  # 使用exit或點關閉按鈕退出
