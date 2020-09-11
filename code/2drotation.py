'''
import csv
import numpy as np
import matplotlib.pyplot as plt

csvFile = open("E:/3DPointsDataset/KittiSeg-master/data/examples/um_road_000005.png", "r")
reader = csv.reader(csvFile)
# 建立空列表
datas = []
for data in reader:
	datas.append(data)
datas = np.array(datas)
datas = datas.astype(np.float64)
csvFile.close()
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('E:/3DPointsDataset/KittiSeg-master/data/examples/um_road_000005.png')
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
H = np.float32([[1,0,100],[0,1,50]])
rows,cols = img.shape[:2]

#平移
res = cv2.warpAffine(img,H,(rows,cols)) #需要图像、变换矩阵、变换后的大小
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
plt.show()
#旋转
#第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
#第三个参数：变换后的图像大小
res = cv2.warpAffine(img,M,(rows,cols))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
plt.show()

#缩放
# 插值：interpolation
# None本应该是放图像大小的位置的，后面设置了缩放比例，
#所有就不要了
res1 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
#直接规定缩放大小，这个时候就不需要缩放因子
height,width = img.shape[:2]
res2 = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(res1)
plt.subplot(133)
plt.imshow(res2)
plt.show()