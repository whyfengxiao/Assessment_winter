import cv2
import numpy as np

cnt = 0
# 读取图片
image = cv2.imread(
    '68747470733a2f2f696d616765732e6e756c6c2d7177657274792e746f702f70686f656e69782f636f756e742e706e67.png')

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if (cv2.arcLength(contours[i], True) < 500) and (cv2.arcLength(contours[i], True) > 300):
        # cv2.drawContours(image, contours[i], -1, (0, 0, 255), 3)
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = box.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(image, [box], True, (0, 255, 0), 10)
        cnt += 1
cv2.putText(image, "count=" + str(cnt), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
print(cnt)
cv2.imshow("img", image)
cv2.waitKey(0)
