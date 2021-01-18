import numpy as np
import matplotlib.pyplot as plt


def convert(data):
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


def img_generator(data):
    """
    生成最後顯示的影像
    pixel value: 0-255
    image shape: 64x64
    """
    img = []
    for line in data.split("\n"):
        row = []
        for char in line:
            num = convert(char)
            if num != None:
                img.append(num)
    img = np.array(img, dtype=np.uint8).reshape(64, 64)  # reshape圖片大小
    return img


data_path1 = "./LINCOLN.64"
data_path2 = "./LIBERTY.64"

f = open(data_path1, mode="r")
data1 = f.read()
f.close()

f = open(data_path2, mode="r")
data2 = f.read()
f.close()

img1 = img_generator(data1)
img2 = img_generator(data2)


def mean(img1, img2):
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


img = mean(img1, img2)

plt.imshow(img, cmap="gray")
plt.show()
