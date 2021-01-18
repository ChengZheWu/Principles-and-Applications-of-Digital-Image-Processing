import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./rects.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
row, col = img.shape

edges = cv2.Canny(img, 50, 150)

lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
for i in range(lines.shape[0]):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


plt.subplot(121)
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
