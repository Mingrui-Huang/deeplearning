import torch
import cv2
import numpy as np

from lenet import LeNet


classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

net = LeNet()
net.load_state_dict(torch.load('LeNet.pth'))

img = cv2.imread('bag.jpg')  # 以灰度图的方式读取要预测的图片
print(img.shape)
cv2.imshow('original_image', img)
cv2.waitKey()
h, w = img.shape[0:2]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为fashionmnist数据集都是灰度图
cv2.imshow('gray_image', img_gray)
cv2.waitKey()

# 颜色反转
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
img = np.zeros((height, width, 1), np.uint8)

for i in range(0, height):
    for j in range(0, width):
        grayPixel = img_gray[i, j]
        img[i, j] = 255-grayPixel

cv2.imshow("dst_image", img)
cv2.waitKey()

img = cv2.equalizeHist(img)  # 直方图均衡化，增加对比度
cv2.imshow('shape_image', img)
cv2.waitKey()

img = cv2.resize(img, (28, 28))
cv2.imshow('suit_the_net_image', img)
cv2.waitKey()
img = np.array(img).astype(np.float32)
img = np.expand_dims(img, 0)  # ndarray:(1, 28, 28)
img = np.expand_dims(img, 0)  # ndarray:(1, 1, 28, 28)
img = torch.from_numpy(img)  # ndarray转为tensor:(1, 1, 28, 28)


with torch.no_grad():  # 不计算梯度
    outputs = net(img)
    predict = torch.max(outputs, dim=1)[1].numpy()
print('predict:' + classes[int(predict)])
